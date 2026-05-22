"""
protocol/messages.py
结构化消息定义
"""

import time
import uuid
import struct
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MessageType(IntEnum):
    """消息类型（用IntEnum，序列化更紧凑）"""
    # 会话管理 (1-9)
    AGENT_HELLO         = 1
    AGENT_BYE           = 2
    CAPABILITY_QUERY    = 3
    CAPABILITY_RESPONSE = 4
    HEARTBEAT           = 5
    
    # 任务协作 (10-19)
    TASK_REQUEST        = 10
    TASK_RESULT         = 11
    TASK_ERROR          = 12
    
    # 状态传递 (20-29)
    STATE_TRANSFER      = 20
    STATE_QUERY         = 21
    
    # 记忆操作 (30-39)
    MEMORY_STORE        = 30
    MEMORY_QUERY        = 31
    MEMORY_RESULT       = 32


@dataclass
class StructuredMessage:
    """
    结构化消息
    
    设计原则：
    1. 字段小而精：每个字段有明确语义
    2. 大块数据用指针：state_offset 指向共享内存，不在消息体内传
    3. 记忆用引用：memory_refs 只传ID，不传内容
    4. 二进制序列化：用 struct + msgpack 混合编码
    """
    
    # ========== 消息头（固定大小，便于快速解析）==========
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    msg_type: MessageType = MessageType.TASK_REQUEST
    sender_id: str = ""
    receiver_id: str = "broadcast"
    timestamp: float = field(default_factory=time.time)
    ttl: int = 30                          # 生存时间（秒）
    
    # ========== 任务上下文 ==========
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 5                      # 1-10，1最高
    
    # ========== 动作指令（结构化，非自然语言）==========
    action: Optional[str] = None           # "plan_task" | "search" | "execute" | "summarize"
    action_type: Optional[str] = None      # "plan" | "retrieve" | "execute" | "summarize"
    parameters: Optional[Dict[str, Any]] = None  # 结构化参数
    
    # ========== 结果数据 ==========
    result_data: Optional[Any] = None      # 小数据直接内联
    result_type: str = "inline"            # "inline" | "shm_pointer" | "memory_refs"
    
    # ========== 状态数据指针（核心创新）==========
    state_offset: Optional[int] = None     # 指向共享内存的偏移量
    state_type: Optional[str] = None       # "sde" | "embedding" | "hidden_state"
    state_size: int = 0                    # 状态数据大小（字节）
    
    # ========== 记忆引用 ==========
    memory_refs: Optional[List[str]] = None  # 只传记忆ID列表
    
    # ========== 元数据 ==========
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ================================================================
    # 序列化方法
    # ================================================================
    
    def serialize(self) -> bytes:
        """
        序列化为紧凑二进制格式
        
        格式布局：
        [header(32B)] [可变长度字段...]
        
        header:
        - magic: 2B  (0x4D41 = "MA")
        - version: 1B
        - msg_type: 1B (IntEnum)
        - priority: 1B
        - ttl: 1B
        - timestamp: 8B (double)
        - body_size: 4B (uint32)
        - state_size: 4B (uint32)
        - state_offset: 8B (uint64, 0表示无)
        - reserved: 2B
        """
        import msgpack
        
        # 构建消息体（可变长度部分）
        body = {
            "id": self.msg_id,
            "s": self.sender_id,
            "r": self.receiver_id,
            "tid": self.task_id,
            "sid": self.session_id,
            "a": self.action,
            "at": self.action_type,
            "p": self.parameters,
            "rd": self.result_data,
            "rt": self.result_type,
            "st": self.state_type,
            "mr": self.memory_refs,
            "m": self.metadata
        }
        
        # 移除None值，减小体积
        body = {k: v for k, v in body.items() if v is not None}
        
        body_bytes = msgpack.packb(body)
        
        # 构建头部
        header = struct.pack(
            ">HBBBBdIIQH",       # 大端序格式
            0x4D41,              # magic
            1,                   # version
            int(self.msg_type),  # msg_type
            self.priority,       # priority
            self.ttl,            # ttl
            self.timestamp,      # timestamp
            len(body_bytes),     # body_size
            self.state_size,     # state_size
            self.state_offset or 0,  # state_offset
            0                    # reserved
        )
        
        return header + body_bytes
    
    @classmethod
    def deserialize(cls, data: bytes) -> "StructuredMessage":
        """
        从二进制数据反序列化
        """
        import msgpack
        
        # 解析头部
        header_size = 32
        header = data[:header_size]
        body_bytes = data[header_size:]
        
        (
            magic, version, msg_type, priority, ttl,
            timestamp, body_size, state_size, state_offset, _
        ) = struct.unpack(">HBBBBdIIQH", header)
        
        if magic != 0x4D41:
            raise ValueError("无效的消息格式（magic不匹配）")
        
        # 解析消息体
        body = msgpack.unpackb(body_bytes)
        
        return cls(
            msg_id=body.get("id", ""),
            msg_type=MessageType(msg_type),
            sender_id=body.get("s", ""),
            receiver_id=body.get("r", "broadcast"),
            timestamp=timestamp,
            ttl=ttl,
            task_id=body.get("tid"),
            session_id=body.get("sid"),
            priority=priority,
            action=body.get("a"),
            action_type=body.get("at"),
            parameters=body.get("p"),
            result_data=body.get("rd"),
            result_type=body.get("rt", "inline"),
            state_offset=state_offset if state_offset > 0 else None,
            state_type=body.get("st"),
            state_size=state_size,
            memory_refs=body.get("mr"),
            metadata=body.get("m", {})
        )
    
    # ================================================================
    # 工具方法
    # ================================================================
    
    def estimate_text_tokens(self) -> int:
        """
        估算等价的纯文本 Token 数
        
        用于性能对比：比较结构化消息 vs 纯文本消息的通信量
        """
        token_count = 0
        
        # 动作和类型
        if self.action:
            token_count += len(self.action.split("_")) * 2
        if self.action_type:
            token_count += 2
        
        # 参数
        if self.parameters:
            import json
            param_str = json.dumps(self.parameters, ensure_ascii=False)
            token_count += len(param_str) // 4
        
        # 结果数据（仅当内联时）
        if self.result_type == "inline" and self.result_data:
            import json
            result_str = json.dumps(self.result_data, ensure_ascii=False)
            token_count += len(result_str) // 4
        
        # 记忆引用
        if self.memory_refs:
            token_count += len(self.memory_refs) * 3  # 每个引用约3个token
        
        return token_count
    
    def actual_size(self) -> int:
        """消息实际大小（字节）"""
        return len(self.serialize())
    
    def __repr__(self) -> str:
        return (
            f"Msg({self.msg_id[:8]}.., {self.msg_type.name}, "
            f"{self.sender_id}→{self.receiver_id}, "
            f"action={self.action}, body={self.actual_size()}B, "
            f"state={self.state_size}B@{self.state_offset})"
        )