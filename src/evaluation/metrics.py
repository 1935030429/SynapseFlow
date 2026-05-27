"""
evaluation/metrics.py
任务指标采集器

负责采集单任务执行过程中的所有性能指标：
- 通信指标：消息数量、Token 消耗、数据传输量
- 状态传递指标：SDE/KV Cache 传递次数和数据量
- 记忆指标：查询次数、命中次数、命中率
- 时间指标：任务总耗时
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.protocol.messages import StructuredMessage


@dataclass
class TaskMetrics:
    """
    单任务指标采集器
    
    用法:
        metrics = TaskMetrics(task_id="task_001", mode="structured")
        metrics.start()
        
        # 任务执行过程中...
        metrics.record_message(msg)
        metrics.record_memory_query(hits=3, total=5)
        metrics.record_state_transfer(size=4800)
        
        metrics.finish()
        print(metrics.to_dict())
    """
    
    task_id: str
    mode: str = "structured"  # "text" 或 "structured"
    
    # ========== 时间 ==========
    start_time: float = 0.0
    end_time: float = 0.0
    
    # ========== 通信指标 ==========
    message_count: int = 0           # 总消息数
    text_tokens_sent: int = 0        # 发送的等价文本 Token
    text_tokens_received: int = 0    # 接收的等价文本 Token
    total_bytes_sent: int = 0        # 发送的总字节数
    total_bytes_received: int = 0    # 接收的总字节数
    
    # ========== 状态传递指标 ==========
    state_transfer_count: int = 0    # 状态传递次数
    state_data_bytes: int = 0        # 状态数据总字节数
    
    # ========== 记忆指标 ==========
    memory_queries: int = 0          # 记忆查询次数
    memory_hits: int = 0             # 记忆命中次数
    memories_created: int = 0        # 新创建的记忆数
    
    # ========== LLM 调用 ==========
    llm_calls: int = 0               # LLM 调用次数
    llm_tokens_generated: int = 0    # LLM 生成的 Token 数
    llm_tokens_prompt: int = 0       # LLM 的 prompt Token 数
    
    # ========== 步骤执行 ==========
    total_steps: int = 0             # 总步骤数
    completed_steps: int = 0         # 已完成步骤数
    failed_steps: int = 0            # 失败步骤数
    
    # ========== 内部状态 ==========
    _messages: List[Dict] = field(default_factory=list)  # 消息记录
    
    # ================================================================
    # 生命周期
    # ================================================================
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def finish(self):
        """结束计时"""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """任务耗时（秒）"""
        if self.end_time == 0:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    # ================================================================
    # 通信指标采集
    # ================================================================
    
    def record_message(self, msg: StructuredMessage, direction: str = "auto"):
        """
        记录一条消息
        
        参数:
            msg: 结构化消息
            direction: "sent" / "received" / "auto"
        """
        self.message_count += 1
        
        # 估算 Token 数
        tokens = msg.estimate_text_tokens()
        
        if direction == "sent" or direction == "auto":
            self.text_tokens_sent += tokens
            self.total_bytes_sent += msg.actual_size()
        if direction == "received" or direction == "auto":
            self.text_tokens_received += tokens
            self.total_bytes_received += msg.actual_size()
        
        # 记录消息概要
        self._messages.append({
            "msg_id": msg.msg_id[:8],
            "msg_type": msg.msg_type.name if msg.msg_type else "?",
            "action": msg.action,
            "tokens": tokens,
            "bytes": msg.actual_size(),
            "has_state": msg.state_offset is not None,
            "has_memory_refs": bool(msg.memory_refs)
        })
    
    # ================================================================
    # 状态传递指标采集
    # ================================================================
    
    def record_state_transfer(self, size_bytes: int):
        """
        记录一次状态传递
        
        参数:
            size_bytes: 状态数据的大小（字节）
        """
        self.state_transfer_count += 1
        self.state_data_bytes += size_bytes
    
    # ================================================================
    # 记忆指标采集
    # ================================================================
    
    def record_memory_query(self, hits: int, total: int):
        """
        记录一次记忆查询
        
        参数:
            hits: 命中数
            total: 查询总数
        """
        self.memory_queries += total
        self.memory_hits += hits
    
    def record_memory_created(self, count: int = 1):
        """记录新创建的记忆"""
        self.memories_created += count
    
    # ================================================================
    # LLM 指标采集
    # ================================================================
    
    def record_llm_call(self, prompt_tokens: int = 0, generated_tokens: int = 0):
        """记录一次 LLM 调用"""
        self.llm_calls += 1
        self.llm_tokens_prompt += prompt_tokens
        self.llm_tokens_generated += generated_tokens
    
    # ================================================================
    # 步骤指标
    # ================================================================
    
    def record_step_completed(self):
        """记录一个步骤完成"""
        self.completed_steps += 1
    
    def record_step_failed(self):
        """记录一个步骤失败"""
        self.failed_steps += 1
    
    def set_total_steps(self, total: int):
        """设置总步骤数"""
        self.total_steps = total
    
    # ================================================================
    # 计算属性
    # ================================================================
    
    @property
    def memory_hit_rate(self) -> float:
        """记忆命中率"""
        if self.memory_queries == 0:
            return 0.0
        return self.memory_hits / self.memory_queries
    
    @property
    def avg_message_size(self) -> float:
        """平均消息大小（字节）"""
        if self.message_count == 0:
            return 0.0
        return (self.total_bytes_sent + self.total_bytes_received) / self.message_count
    
    @property
    def state_compression_ratio(self) -> float:
        """
        状态传递的压缩比（相对于等价的文本 Token）
        
        假设: 1 token ≈ 4 bytes（文本）
        状态数据用二进制传递，所以更紧凑
        """
        if self.state_data_bytes == 0 or self.text_tokens_sent == 0:
            return 0.0
        text_equivalent_bytes = self.text_tokens_sent * 4
        return self.state_data_bytes / max(1, text_equivalent_bytes)
    
    @property
    def step_success_rate(self) -> float:
        """步骤成功率"""
        total = self.completed_steps + self.failed_steps
        if total == 0:
            return 1.0
        return self.completed_steps / total
    
    # ================================================================
    # 输出
    # ================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            # 基本信息
            "task_id": self.task_id,
            "mode": self.mode,
            "duration": round(self.duration, 3),
            
            # 通信
            "message_count": self.message_count,
            "text_tokens_equivalent": self.text_tokens_sent + self.text_tokens_received,
            "text_tokens_sent": self.text_tokens_sent,
            "text_tokens_received": self.text_tokens_received,
            "total_bytes": self.total_bytes_sent + self.total_bytes_received,
            "avg_message_bytes": round(self.avg_message_size, 1),
            
            # 状态传递
            "state_transfer_count": self.state_transfer_count,
            "state_data_bytes": self.state_data_bytes,
            "state_compression_ratio": round(self.state_compression_ratio, 3),
            
            # 记忆
            "memory_queries": self.memory_queries,
            "memory_hits": self.memory_hits,
            "memory_hit_rate": round(self.memory_hit_rate, 3),
            "memories_created": self.memories_created,
            
            # LLM
            "llm_calls": self.llm_calls,
            "llm_tokens_generated": self.llm_tokens_generated,
            "llm_tokens_prompt": self.llm_tokens_prompt,
            
            # 步骤
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "step_success_rate": round(self.step_success_rate, 3),
        }
    
    def to_summary(self) -> str:
        """转为单行摘要"""
        return (
            f"[{self.task_id}] mode={self.mode} "
            f"duration={self.duration:.2f}s "
            f"msgs={self.message_count} "
            f"tokens={self.text_tokens_sent + self.text_tokens_received} "
            f"mem_hits={self.memory_hits}/{self.memory_queries} "
            f"({self.memory_hit_rate:.1%}) "
            f"states={self.state_transfer_count}({self.state_data_bytes}B) "
            f"llm={self.llm_calls}calls/{self.llm_tokens_generated}tok"
        )
    
    def print_report(self):
        """打印详细报告"""
        d = self.to_dict()
        print("=" * 50)
        print(f"  任务指标报告: {self.task_id}")
        print("=" * 50)
        print(f"  模式: {self.mode}")
        print(f"  耗时: {d['duration']}s")
        print()
        print(f"  【通信】")
        print(f"    消息数: {d['message_count']}")
        print(f"    等价Token: {d['text_tokens_equivalent']}")
        print(f"    总字节数: {d['total_bytes']}")
        print(f"    平均消息大小: {d['avg_message_bytes']}B")
        print()
        print(f"  【状态传递】")
        print(f"    传递次数: {d['state_transfer_count']}")
        print(f"    数据量: {d['state_data_bytes']}B")
        print(f"    压缩比: {d['state_compression_ratio']}")
        print()
        print(f"  【记忆】")
        print(f"    查询: {d['memory_queries']}")
        print(f"    命中: {d['memory_hits']}")
        print(f"    命中率: {d['memory_hit_rate']}")
        print(f"    新创建: {d['memories_created']}")
        print()
        print(f"  【LLM】")
        print(f"    调用次数: {d['llm_calls']}")
        print(f"    生成Token: {d['llm_tokens_generated']}")
        print(f"    Prompt Token: {d['llm_tokens_prompt']}")
        print()
        print(f"  【步骤】")
        print(f"    总步骤: {d['total_steps']}")
        print(f"    完成: {d['completed_steps']}")
        print(f"    失败: {d['failed_steps']}")
        print(f"    成功率: {d['step_success_rate']}")
        print("=" * 50)


# ================================================================
# 实验对比工具
# ================================================================

class MetricsComparator:
    """
    对比纯文本模式和结构化模式的指标
    
    用法:
        comparator = MetricsComparator()
        comparator.add_text_result(metrics_list_from_text_mode)
        comparator.add_structured_result(metrics_list_from_structured_mode)
        report = comparator.compare()
    """
    
    def __init__(self):
        self.text_metrics: List[TaskMetrics] = []
        self.structured_metrics: List[TaskMetrics] = []
    
    def add_text_result(self, metrics_list: List[TaskMetrics]):
        """添加纯文本模式的指标"""
        self.text_metrics.extend(metrics_list)
    
    def add_structured_result(self, metrics_list: List[TaskMetrics]):
        """添加结构化模式的指标"""
        self.structured_metrics.extend(metrics_list)
    
    def compare(self) -> Dict[str, Any]:
        """生成对比报告"""
        text_avg = self._average(self.text_metrics)
        struct_avg = self._average(self.structured_metrics)
        
        if not text_avg or not struct_avg:
            return {"error": "缺少对比数据"}
        
        # Token 节省率
        text_tokens = text_avg["text_tokens_equivalent"]
        struct_tokens = struct_avg["text_tokens_equivalent"]
        token_reduction = (text_tokens - struct_tokens) / max(1, text_tokens) * 100
        
        # 时延改善率
        text_duration = text_avg["duration"]
        struct_duration = struct_avg["duration"]
        time_improvement = (text_duration - struct_duration) / max(0.001, text_duration) * 100
        
        # 记忆命中率提升
        memory_improvement = struct_avg["memory_hit_rate"] - text_avg["memory_hit_rate"]
        
        # 消息数减少
        msg_reduction = (text_avg["message_count"] - struct_avg["message_count"]) / max(1, text_avg["message_count"]) * 100
        
        return {
            "text_mode": text_avg,
            "structured_mode": struct_avg,
            "comparison": {
                "token_reduction": f"{token_reduction:.1f}%",
                "token_reduction_abs": text_tokens - struct_tokens,
                "time_improvement": f"{time_improvement:.1f}%",
                "time_improvement_abs": round(text_duration - struct_duration, 3),
                "memory_hit_rate_text": round(text_avg["memory_hit_rate"], 3),
                "memory_hit_rate_structured": round(struct_avg["memory_hit_rate"], 3),
                "memory_improvement": f"{memory_improvement:.1%}",
                "message_reduction": f"{msg_reduction:.1f}%",
                "state_transfer_enabled": struct_avg["state_transfer_count"] > 0
            }
        }
    
    def _average(self, metrics_list: List[TaskMetrics]) -> Optional[Dict]:
        """计算平均值"""
        if not metrics_list:
            return None
        
        n = len(metrics_list)
        return {
            "task_count": n,
            "duration": sum(m.duration for m in metrics_list) / n,
            "message_count": sum(m.message_count for m in metrics_list),
            "text_tokens_equivalent": sum(m.text_tokens_sent + m.text_tokens_received for m in metrics_list),
            "total_bytes": sum(m.total_bytes_sent + m.total_bytes_received for m in metrics_list),
            "state_transfer_count": sum(m.state_transfer_count for m in metrics_list),
            "state_data_bytes": sum(m.state_data_bytes for m in metrics_list),
            "memory_queries": sum(m.memory_queries for m in metrics_list),
            "memory_hits": sum(m.memory_hits for m in metrics_list),
            "memory_hit_rate": sum(m.memory_hit_rate for m in metrics_list) / n,
            "llm_calls": sum(m.llm_calls for m in metrics_list),
            "llm_tokens_generated": sum(m.llm_tokens_generated for m in metrics_list),
        }
    
    def print_report(self):
        """打印对比报告"""
        report = self.compare()
        if "error" in report:
            print(f"错误: {report['error']}")
            return
        
        comp = report["comparison"]
        
        print("\n" + "=" * 60)
        print("  性能对比报告: 纯文本模式 vs 结构化模式")
        print("=" * 60)
        
        print(f"\n  【任务概况】")
        print(f"    纯文本模式任务数: {report['text_mode']['task_count']}")
        print(f"    结构化模式任务数: {report['structured_mode']['task_count']}")
        
        print(f"\n  【通信效率】")
        print(f"    纯文本模式 Token: {report['text_mode']['text_tokens_equivalent']:,}")
        print(f"    结构化模式 Token: {report['structured_mode']['text_tokens_equivalent']:,}")
        print(f"    Token 节省率: {comp['token_reduction']}")
        print(f"    消息数减少: {comp['message_reduction']}")
        
        print(f"\n  【任务时延】")
        print(f"    纯文本模式平均: {report['text_mode']['duration']:.3f}s")
        print(f"    结构化模式平均: {report['structured_mode']['duration']:.3f}s")
        print(f"    时延改善率: {comp['time_improvement']}")
        
        print(f"\n  【记忆复用】")
        print(f"    纯文本模式命中率: {comp['memory_hit_rate_text']:.1%}")
        print(f"    结构化模式命中率: {comp['memory_hit_rate_structured']:.1%}")
        print(f"    命中率提升: {comp['memory_improvement']}")
        
        print(f"\n  【状态传递】")
        print(f"    状态传递次数: {report['structured_mode']['state_transfer_count']}")
        print(f"    状态数据量: {report['structured_mode']['state_data_bytes']:,}B")
        print(f"    状态传递启用: {comp['state_transfer_enabled']}")
        
        print(f"\n  【LLM 调用】")
        print(f"    纯文本模式: {report['text_mode']['llm_calls']}次, {report['text_mode']['llm_tokens_generated']:,} tokens")
        print(f"    结构化模式: {report['structured_mode']['llm_calls']}次, {report['structured_mode']['llm_tokens_generated']:,} tokens")
        
        print("=" * 60)