"""
消息总线

当前原型阶段：
- 所有 Agent 在同一进程，直接调用 handle_message
- 支持异步并行发送
- 保留跨进程扩展接口

后续可扩展为：
- Socket/TCP 通信
- 消息队列（RabbitMQ/Kafka）
- 共享内存 + 信号量
"""

import asyncio
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field

from src.protocol.messages import StructuredMessage, MessageType


@dataclass
class PendingRequest:
    """一个等待回复的请求"""
    msg_id: str
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)
    timeout: float = 60.0


class MessageBus:
    """
    消息总线
    
    职责：
    1. 将消息从发送方投递到接收方
    2. 支持"发送-等待回复"模式
    3. 支持"发送-不等待"（异步通知）模式
    4. 支持广播
    """
    
    def __init__(self, agents: Dict[str, Any] = None):
        """
        参数:
            agents: Agent 字典 {agent_id: Agent实例}
        """
        self.agents: Dict[str, Any] = agents or {}
        
        # 等待回复的请求
        self._pending: Dict[str, PendingRequest] = {}
        
        # 临时处理器：msg_id → callback（用于处理异步回复）
        self._handlers: Dict[str, Callable] = {}
        
        # 消息日志
        self.message_log: List[StructuredMessage] = []
        
        # 统计
        self.stats = {
            "total_sent": 0,
            "total_received": 0,
            "total_errors": 0,
            "total_timeouts": 0
        }
    
    # ================================================================
    # Agent 管理
    # ================================================================
    
    def register_agent(self, agent_id: str, agent: Any):
        """注册 Agent"""
        self.agents[agent_id] = agent
        print(f"[MessageBus] Agent 注册: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """注销 Agent"""
        self.agents.pop(agent_id, None)
        # 取消该 Agent 的所有等待请求
        for msg_id, pending in list(self._pending.items()):
            if not pending.future.done():
                pending.future.set_exception(
                    RuntimeError(f"Agent {agent_id} 已下线")
                )
        print(f"[MessageBus] Agent 注销: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """获取 Agent"""
        return self.agents.get(agent_id)
    
    # ================================================================
    # 发送消息（三种模式）
    # ================================================================
    
    async def send(
        self,
        msg: StructuredMessage,
        target: str = None,
        timeout: float = 60.0
    ) -> StructuredMessage:
        """
        发送消息并等待回复
        
        这是最常用的模式：发一个请求，等一个回复。
        
        参数:
            msg: 消息
            target: 目标 agent_id
            timeout: 超时时间（秒）
        
        返回:
            回复消息
        """
        # 1. 创建 Future 用于接收回复
        future = asyncio.Future()
        
        # 2. 注册等待
        self._pending[msg.msg_id] = PendingRequest(
            msg_id=msg.msg_id,
            future=future,
            timeout=timeout
        )
        
        # 3. 发送消息
        await self._deliver(msg, target)
        
        # 4. 等待回复
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self.stats["total_timeouts"] += 1
            return StructuredMessage(
                msg_type=MessageType.TASK_ERROR,
                sender_id="message_bus",
                receiver_id=msg.sender_id,
                task_id=msg.task_id,
                result_data={"error": f"等待回复超时 ({timeout}s)"}
            )
        finally:
            self._pending.pop(msg.msg_id, None)
    
    async def send_async(
        self,
        msg: StructuredMessage,
        target: str = None,
        callback: Callable = None
    ):
        """
        发送消息，不等待回复（异步通知）
        
        参数:
            msg: 消息
            target: 目标 agent_id
            callback: 回复时的回调函数（可选）
        """
        if callback:
            self._handlers[msg.msg_id] = callback
        
        await self._deliver(msg, target)
    
    async def broadcast(self, msg: StructuredMessage):
        """
        广播消息给所有 Agent
        
        不等待回复。
        """
        msg.receiver_id = "broadcast"
        
        tasks = []
        for agent_id in self.agents:
            if agent_id != msg.sender_id:
                tasks.append(self._deliver_to(agent_id, msg))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    # ================================================================
    # 回复消息
    # ================================================================
    
    async def reply(self, response: StructuredMessage):
        """
        Agent 通过此方法发送回复消息
        
        会自动匹配到原始请求，唤醒等待者。
        """
        # 回复消息的 in_reply_to 指向原始消息的 msg_id
        original_msg_id = response.metadata.get("in_reply_to", "")
        
        if original_msg_id and original_msg_id in self._pending:
            # 有等待者，直接设置结果
            pending = self._pending[original_msg_id]
            if not pending.future.done():
                pending.future.set_result(response)
            return
        
        if original_msg_id and original_msg_id in self._handlers:
            # 有回调处理器
            handler = self._handlers.pop(original_msg_id)
            handler(response)
            return
        
        # 没有等待者也没有处理器，按正常消息投递
        target = response.receiver_id
        if target and target != "broadcast":
            await self._deliver_to(target, response)
    
    # ================================================================
    # 消息投递（内部）
    # ================================================================
    
    async def _deliver(self, msg: StructuredMessage, target: str = None):
        """
        投递消息到目标 Agent
        
        当前原型：直接调用 agent.handle_message()
        后续可替换为：Socket/TCP 发送
        """
        if target:
            await self._deliver_to(target, msg)
        elif msg.receiver_id and msg.receiver_id != "broadcast":
            await self._deliver_to(msg.receiver_id, msg)
        else:
            # 广播
            await self.broadcast(msg)
    
    async def _deliver_to(self, agent_id: str, msg: StructuredMessage):
        """
        投递消息到指定 Agent
        
        原型阶段：直接调用 handle_message
        """
        agent = self.agents.get(agent_id)
        
        if not agent:
            self.stats["total_errors"] += 1
            # 如果有人在等回复，返回错误
            if msg.msg_id in self._pending:
                pending = self._pending[msg.msg_id]
                if not pending.future.done():
                    pending.future.set_result(StructuredMessage(
                        msg_type=MessageType.TASK_ERROR,
                        sender_id="message_bus",
                        receiver_id=msg.sender_id,
                        task_id=msg.task_id,
                        result_data={"error": f"Agent {agent_id} 不存在"}
                    ))
            return
        
        self.stats["total_sent"] += 1
        self.message_log.append(msg)
        
        try:
            # 直接调用 Agent 的 handle_message
            response = await agent.handle_message(msg)
            
            self.stats["total_received"] += 1
            
            if response:
                # 标记回复
                response.metadata["in_reply_to"] = msg.msg_id
                response.task_id = response.task_id or msg.task_id
                response.receiver_id = response.receiver_id or msg.sender_id
                
                # 通过 reply 方法处理回复
                await self.reply(response)
                
        except Exception as e:
            self.stats["total_errors"] += 1
            
            # 如果有人在等回复，返回错误
            if msg.msg_id in self._pending:
                pending = self._pending[msg.msg_id]
                if not pending.future.done():
                    pending.future.set_result(StructuredMessage(
                        msg_type=MessageType.TASK_ERROR,
                        sender_id=agent_id,
                        receiver_id=msg.sender_id,
                        task_id=msg.task_id,
                        result_data={"error": str(e)}
                    ))
    
    # ================================================================
    # 并行发送
    # ================================================================
    
    async def send_parallel(
        self,
        messages: List[tuple],  # [(msg, target), ...]
        timeout: float = 60.0
    ) -> List[StructuredMessage]:
        """
        并行发送多条消息，等待所有回复
        
        参数:
            messages: [(消息, 目标), ...] 的列表
            timeout: 超时时间
        
        返回:
            回复列表（与输入顺序对应）
        
        用法:
            results = await bus.send_parallel([
                (msg1, "retriever_01"),
                (msg2, "retriever_02"),
            ])
        """
        tasks = [
            self.send(msg, target, timeout)
            for msg, target in messages
        ]
        return await asyncio.gather(*tasks)
    
    # ================================================================
    # 统计与调试
    # ================================================================
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "pending_count": len(self._pending),
            "handler_count": len(self._handlers),
            "log_size": len(self.message_log),
            "agent_count": len(self.agents)
        }
    
    def get_pending_requests(self) -> List[Dict]:
        """获取等待中的请求"""
        return [
            {
                "msg_id": p.msg_id,
                "elapsed": time.time() - p.created_at,
                "timeout": p.timeout
            }
            for p in self._pending.values()
            if not p.future.done()
        ]
    
    def clear_log(self):
        """清除消息日志"""
        self.message_log.clear()
        