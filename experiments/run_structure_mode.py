"""
run_structured_mode.py
结构化 + SDE 模式实验运行脚本

执行流程：
1. 加载系统各模块
2. 执行任务列表
3. 采集性能指标
4. 保存结果
"""

import asyncio
import json
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runtime.shared_memory import SharedMemoryManager
from src.runtime.message_bus import MessageBus
from src.runtime.orchestrator import Orchestrator

from src.protocol.messages import StructuredMessage, MessageType
from src.protocol.router import ProtocolRouter

from src.state_transfer.sde_manager import SDEManager

from src.memory.memory_store import MemoryStore
from src.memory.memory_retrieval import HybridRetrieval
from src.memory.memory_graph import MemoryGraph

from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.executor import ExecutorAgent
from src.agents.summarizer import SummarizerAgent

from src.evaluation.metrics import TaskMetrics
from src.evaluation.benchmark import BenchmarkRunner

from experiments.tasks import get_tasks


async def init_system(config):
    """初始化系统各模块"""
    
    print("[INIT] 初始化共享内存...")
    shm = SharedMemoryManager(size_mb=config.get("shm_size_mb", 256))
    
    print("[INIT] 初始化记忆存储...")
    memory_store = MemoryStore(persist_dir=config.get("memory_dir", "./memory_store"))
    
    print("[INIT] 加载模型并初始化SDE管理器...")
    sde_manager = SDEManager(
        model_name=config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        shm_manager=shm,
        quantize_bits=config.get("sde_quantize_bits", 8)
    )
    
    print("[INIT] 初始化记忆检索...")
    memory_retrieval = HybridRetrieval(
        store=memory_store,
        sde_manager=sde_manager
    )
    memory_graph = MemoryGraph(store=memory_store)
    
    print("[INIT] 初始化协议路由器...")
    router = ProtocolRouter()
    
    print("[INIT] 创建Agent...")
    planner = PlannerAgent(
        agent_id="planner_01",
        role="planner",
        router=router,
        shm=shm,
        sde_manager=sde_manager,
        memory=memory_retrieval,
        memory_graph=memory_graph,
        model_config={
            "name": config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            "quantize": config.get("model_quantize", "int8"),
            "max_tokens": config.get("max_tokens", 2048)
        }
    )
    
    retriever = RetrieverAgent(
        agent_id="retriever_01",
        role="retriever",
        router=router,
        shm=shm,
        sde_manager=sde_manager,
        memory=memory_retrieval,
        memory_graph=memory_graph,
        model_config={
            "embedding_model": config.get("embedding_model", "BAAI/bge-m3")
        }
    )
    
    executor = ExecutorAgent(
        agent_id="executor_01",
        role="executor",
        router=router,
        shm=shm,
        sde_manager=sde_manager,
        memory=memory_retrieval,
        memory_graph=memory_graph,
        sandbox_config={
            "type": config.get("sandbox_type", "nsjail"),
            "timeout": config.get("sandbox_timeout", 30),
            "max_memory": config.get("sandbox_memory", "256m")
        }
    )
    
    summarizer = SummarizerAgent(
        agent_id="summarizer_01",
        role="summarizer",
        router=router,
        shm=shm,
        sde_manager=sde_manager,
        memory=memory_retrieval,
        memory_graph=memory_graph,
        model_config={
            "name": config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            "quantize": config.get("model_quantize", "int8"),
            "max_tokens": config.get("max_tokens", 2048)
        }
    )
    
    agents = [planner, retriever, executor, summarizer]
    
    print("[INIT] 初始化消息总线...")
    bus = MessageBus(router=router, agents=agents)
    
    print("[INIT] 初始化调度器...")
    orchestrator = Orchestrator(
        bus=bus,
        shm=shm,
        router=router,
        agents=agents
    )
    
    print("[INIT] 启动所有Agent...")
    for agent in agents:
        await agent.start()
    
    print("[INIT] 系统初始化完成\n")
    
    return orchestrator, agents, shm, memory_store, sde_manager


async def run_task(orchestrator, task, task_index):
    """运行单个任务并采集指标"""
    
    metrics = TaskMetrics(
        task_id=task["id"],
        mode="structured"
    )
    metrics.start()
    
    print(f"[TASK {task_index}] 开始: {task['description'][:60]}...")
    
    # 发送任务到调度器
    result = await orchestrator.submit_task(
        user_input=task["description"],
        context=task.get("context", {}),
        metrics=metrics
    )
    
    metrics.finish()
    
    print(f"[TASK {task_index}] 完成")
    print(f"   耗时: {metrics.duration:.2f}s")
    print(f"   消息数: {metrics.message_count}")
    print(f"   等价Token: {metrics.text_tokens_equivalent}")
    print(f"   SDE传递: {metrics.sde_count}次, {metrics.sde_bytes}字节")
    print(f"   记忆命中: {metrics.memory_hits}/{metrics.memory_queries} "
          f"({metrics.memory_hit_rate:.1%})")
    print()
    
    return metrics


async def run_structured_mode(config):
    """运行结构化 + SDE 模式实验"""
    
    print("=" * 60)
    print("  结构化 + SDE 模式实验")
    print("=" * 60)
    print()
    
    # 初始化系统
    orchestrator, agents, shm, memory_store, sde_manager = await init_system(config)
    
    # 获取任务列表
    tasks = get_tasks()
    
    all_metrics = []
    
    # 顺序执行任务
    for i, task in enumerate(tasks, 1):
        metrics = await run_task(orchestrator, task, i)
        all_metrics.append(metrics)
        
        # 任务间短暂间隔
        await asyncio.sleep(1)
    
    # 汇总统计
    print("=" * 60)
    print("  汇总统计")
    print("=" * 60)
    
    total_duration = sum(m.duration for m in all_metrics)
    total_messages = sum(m.message_count for m in all_metrics)
    total_tokens = sum(m.text_tokens_equivalent for m in all_metrics)
    total_sde_count = sum(m.sde_count for m in all_metrics)
    total_sde_bytes = sum(m.sde_bytes for m in all_metrics)
    total_memory_hits = sum(m.memory_hits for m in all_metrics)
    total_memory_queries = sum(m.memory_queries for m in all_metrics)
    
    avg_duration = total_duration / len(all_metrics)
    avg_messages = total_messages / len(all_metrics)
    avg_memory_hit_rate = total_memory_hits / max(1, total_memory_queries)
    
    print(f"  任务数: {len(all_metrics)}")
    print(f"  总耗时: {total_duration:.2f}s")
    print(f"  平均耗时: {avg_duration:.2f}s")
    print(f"  总消息数: {total_messages}")
    print(f"  平均消息数: {avg_messages:.1f}")
    print(f"  总等价Token: {total_tokens}")
    print(f"  SDE传递次数: {total_sde_count}")
    print(f"  SDE数据总量: {total_sde_bytes}字节 ({total_sde_bytes/1024:.1f}KB)")
    print(f"  记忆命中率: {total_memory_hits}/{total_memory_queries} "
          f"({avg_memory_hit_rate:.1%})")
    print()
    
    # 保存结果
    results = {
        "mode": "structured",
        "config": config,
        "tasks": [
            {
                "task_id": m.task_id,
                "duration": m.duration,
                "message_count": m.message_count,
                "text_tokens_equivalent": m.text_tokens_equivalent,
                "total_bytes": m.total_bytes,
                "sde_count": m.sde_count,
                "sde_bytes": m.sde_bytes,
                "sde_compression_ratio": m.sde_compression_ratio,
                "memory_queries": m.memory_queries,
                "memory_hits": m.memory_hits,
                "memory_hit_rate": m.memory_hit_rate,
                "memories_created": m.memories_created
            }
            for m in all_metrics
        ],
        "summary": {
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_sde_count": total_sde_count,
            "total_sde_bytes": total_sde_bytes,
            "total_memory_hits": total_memory_hits,
            "total_memory_queries": total_memory_queries,
            "avg_memory_hit_rate": avg_memory_hit_rate
        }
    }
    
    result_path = config.get("result_dir", "./results") + "/structured_mode_results.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[DONE] 结果已保存到 {result_path}")
    
    # 关闭系统
    print("[SHUTDOWN] 停止所有Agent...")
    for agent in agents:
        await agent.stop()
    
    shm.close()
    memory_store.close()
    
    return results


if __name__ == "__main__":
    # 默认配置
    default_config = {
        "shm_size_mb": 256,
        "memory_dir": "./memory_store",
        "result_dir": "./results",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_quantize": "int8",
        "embedding_model": "BAAI/bge-m3",
        "max_tokens": 2048,
        "sde_quantize_bits": 8,
        "sde_sparse_threshold": 0.01,
        "sandbox_type": "nsjail",
        "sandbox_timeout": 30,
        "sandbox_memory": "256m"
    }
    
    asyncio.run(run_structured_mode(default_config))