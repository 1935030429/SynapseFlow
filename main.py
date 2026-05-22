"""
main.py
多智能体协作系统 - 主入口

用法：
    python main.py --mode structured  # 仅运行结构化模式
    python main.py --mode text        # 仅运行纯文本模式
    python main.py --mode both        # 运行两种模式并对比
    python main.py --mode interactive # 交互模式（输入单句话执行）
"""

import asyncio
import argparse
import json
import os
import sys

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_text_mode import run_text_mode
from experiments.run_structure_mode import run_structured_mode
from experiments.compare_results import compare_results


def load_config(config_path="config.yaml"):
    """加载配置文件"""
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
    
    # 尝试加载 YAML 配置（如果存在）
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            default_config.update(yaml_config)
        except ImportError:
            print("[WARN] PyYAML未安装，使用默认配置")
        except Exception as e:
            print(f"[WARN] 配置文件加载失败: {e}，使用默认配置")
    
    return default_config


async def run_interactive_mode(config):
    """交互模式：输入一句话，查看执行过程"""
    
    print("=" * 60)
    print("  交互模式 - 结构化 + SDE")
    print("  输入 'quit' 退出")
    print("=" * 60)
    print()
    
    # 初始化系统
    from src.runtime.shared_memory import SharedMemoryManager
    from src.runtime.message_bus import MessageBus
    from src.runtime.orchestrator import Orchestrator
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
    
    print("[INIT] 正在初始化系统...")
    
    shm = SharedMemoryManager(size_mb=config.get("shm_size_mb", 256))
    memory_store = MemoryStore(persist_dir=config.get("memory_dir", "./memory_store"))
    
    print("[INIT] 加载模型（可能需要1-2分钟）...")
    sde_manager = SDEManager(
        model_name=config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        shm_manager=shm
    )
    
    memory_retrieval = HybridRetrieval(store=memory_store, sde_manager=sde_manager)
    memory_graph = MemoryGraph(store=memory_store)
    router = ProtocolRouter()
    
    planner = PlannerAgent(
        agent_id="planner_01", role="planner",
        router=router, shm=shm, sde_manager=sde_manager,
        memory=memory_retrieval, memory_graph=memory_graph,
        model_config={"name": config["model_name"]}
    )
    
    retriever = RetrieverAgent(
        agent_id="retriever_01", role="retriever",
        router=router, shm=shm, sde_manager=sde_manager,
        memory=memory_retrieval, memory_graph=memory_graph,
        model_config={"embedding_model": config["embedding_model"]}
    )
    
    executor = ExecutorAgent(
        agent_id="executor_01", role="executor",
        router=router, shm=shm, sde_manager=sde_manager,
        memory=memory_retrieval, memory_graph=memory_graph
    )
    
    summarizer = SummarizerAgent(
        agent_id="summarizer_01", role="summarizer",
        router=router, shm=shm, sde_manager=sde_manager,
        memory=memory_retrieval, memory_graph=memory_graph,
        model_config={"name": config["model_name"]}
    )
    
    agents = [planner, retriever, executor, summarizer]
    bus = MessageBus(router=router, agents=agents)
    orchestrator = Orchestrator(bus=bus, shm=shm, router=router, agents=agents)
    
    for agent in agents:
        await agent.start()
    
    print("[INIT] 系统就绪！\n")
    
    task_count = 0
    
    while True:
        try:
            user_input = input(">>> 请输入任务: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                break
            
            task_count += 1
            
            # 创建指标采集器
            metrics = TaskMetrics(
                task_id=f"interactive_{task_count}",
                mode="structured"
            )
            metrics.start()
            
            print(f"\n[执行] 处理任务: {user_input}")
            print("-" * 40)
            
            # 执行
            result = await orchestrator.submit_task(
                user_input=user_input,
                context={
                    "user_id": "interactive_user",
                    "timestamp": __import__('time').time()
                },
                metrics=metrics
            )
            
            metrics.finish()
            
            # 打印结果
            print("-" * 40)
            print(f"[结果] {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
            print()
            print(f"[统计] 耗时: {metrics.duration:.2f}s | "
                  f"消息: {metrics.message_count} | "
                  f"等价Token: {metrics.text_tokens_equivalent} | "
                  f"SDE: {metrics.sde_count}次/{metrics.sde_bytes}字节 | "
                  f"记忆命中: {metrics.memory_hits}/{metrics.memory_queries}")
            print()
            
        except KeyboardInterrupt:
            print("\n[EXIT] 用户中断")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
    
    # 关闭
    print("[SHUTDOWN] 停止所有Agent...")
    for agent in agents:
        await agent.stop()
    shm.close()
    memory_store.close()
    print("[SHUTDOWN] 系统已关闭")


async def main():
    parser = argparse.ArgumentParser(
        description="多智能体协作系统 - 低开销通信与共享记忆"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["text", "structured", "both", "interactive"],
        help="运行模式: text(纯文本), structured(结构化+SDE), both(对比), interactive(交互)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="交互模式下直接传入任务（非交互式）"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    print("=" * 60)
    print("  多智能体协作系统")
    print("  低开销通信 · 非文本状态传递 · 共享记忆复用")
    print("=" * 60)
    print(f"  运行模式: {args.mode}")
    print(f"  模型: {config['model_name']}")
    print(f"  共享内存: {config['shm_size_mb']}MB")
    print(f"  沙箱: {config['sandbox_type']}")
    print("=" * 60)
    print()
    
    if args.mode == "text":
        await run_text_mode(config)
        
    elif args.mode == "structured":
        await run_structured_mode(config)
        
    elif args.mode == "both":
        print(">>> 第1阶段：纯文本模式")
        text_results = await run_text_mode(config)
        
        print("\n\n>>> 第2阶段：结构化 + SDE 模式")
        struct_results = await run_structured_mode(config)
        
        print("\n\n>>> 对比分析")
        compare_results(text_results, struct_results)
        
    elif args.mode == "interactive":
        if args.input:
            # 非交互式：直接执行传入的任务
            config_copy = config.copy()
            # 这里可以扩展为单次执行模式
            print(f"[TASK] {args.input}")
        await run_interactive_mode(config)


if __name__ == "__main__":
    asyncio.run(main())