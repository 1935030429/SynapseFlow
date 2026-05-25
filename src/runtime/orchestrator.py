import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from src.runtime.message_bus import MessageBus

from src.protocol.messages import StructuredMessage, MessageType
from src.agents.base_agent import BaseAgent
from src.protocol.router import ProtocolRouter
from src.evaluation.metrics import TaskMetrics
from src.runtime.shared_memory import SharedMemoryManager

@dataclass
class TaskContext:
    """
    单个任务的运行时上下文
    
    一个任务从创建到完成的完整生命周期都记录在这里。
    """
    task_id: str                              # 任务唯一ID
    user_input: str                           # 用户原始输入
    context: Dict[str, Any] = field(default_factory=dict)  # 额外上下文
    
    # 执行状态
    plan: Optional[Dict] = None               # Planner 生成的计划
    step_results: Dict[int, Any] = field(default_factory=dict)  # 每个子步骤的结果
    completed_steps: set = field(default_factory=set)            # 已完成的步骤ID
    failed_steps: set = field(default_factory=set)               # 失败的步骤ID
    
    # 时间追踪
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0
    
    # 指标采集器引用
    metrics: Optional[TaskMetrics] = None
class Orchestrator:
    """主流程"""
    def __init__(self, bus: MessageBus, shm: SharedMemoryManager, router: ProtocolRouter, agents: List[BaseAgent]):
        self.shm = shm
        self.bus = bus
        self.router = router
        
        self.agent_map = {agent.agent_id: agent for agent in agents}
        self.role_map = {agent.role: agent for agent in agents}
        
        # 记录所有活跃任务（task_id → TaskContext）
        self.active_tasks: Dict[str, TaskContext] = {}
        
        # 记录所有已完成任务的历史（用于查询）
        self.task_history: List[TaskContext] = []
        
    async def submit_task(self, user_input: str, context: Dict[str, Any], metrics: TaskMetrics=None):
        task_id = self.__create_task_id()
        
        task_ctx = TaskContext(
            task_id=task_id,
            user_input=user_input,
            context=context or {},
            metrics=metrics
        )
        
        self.active_tasks[task_id] = task_ctx
        
        #执行任务
        try:
            result = await self.__execute_task(task_ctx)
        except Exception as e:
            result = {"error": str(e)}
        
        #完成，移入历史记录
        task_ctx.finished_at = time.time()
        self.task_history.append(task_ctx)
        
        # del self.active_tasks[task_id]
        self.active_tasks.pop(task_id, None)

        if metrics:
            metrics.finish()
        
        return result
        
        # 创建结构化消息
        # msg = StructuredMessage(
        #     msg_type=MessageType.TASK_REQUEST,
        #     action="plan_task",
        #     task_id=task_id,
        #     parameters={
        #         "task": user_input,              # "今天应该吃什么"
        #         "context": {
        #             "user_id": "user_001",
        #             "date": "2026-05-20",
        #             "location": "上海",
        #             "time": "12:00"
        #         }
        #     }
        # )
    def __create_task_id(self):
        import uuid
        timestamp = int(time.time())
        short_uuid = uuid.uuid4().hex[:6]
        return f"task_{timestamp}_{short_uuid}"
    
    async def __execute_task(self, ctx: TaskContext) -> Dict[str, Any]:
        """
        执行单个任务的完整流程
        
        阶段1：规划 → Planner
        阶段2：执行 → Retriever / Executor（按依赖关系）
        阶段3：总结 → Summarizer
        """
        
        # ──── 阶段1：规划 ────
        ctx.plan = await self.__planning_phase(ctx)
        
        if not ctx.plan or not ctx.plan.get("steps"):
            return {"error": "Fail to plan task, cannot generate execute steps"}
        
        # ──── 阶段2：执行子任务 ────
        await self.__execution_phase(ctx)
        
        # ──── 阶段3：总结 ────
        final_result = await self.__summary_phase(ctx)
        
        return final_result
    
    async def __planning_phase(self, ctx: TaskContext) -> Dict:
        """
        向 Planner 发送规划请求，等待计划返回
        
        流程：
        1. 构造规划消息
        2. 找到 Planner Agent
        3. 发送消息并等待回复
        4. 从回复中获取计划（可能在消息体内，也可能在共享内存中）
        """
        
        # 2. 找到 Planner（通过角色查找）
        planner = self.role_map.get("planner")
        if not planner:
            raise RuntimeError("没有可用的 Planner Agent")

        # 1. 构造消息
        plan_msg = StructuredMessage(
            msg_type=MessageType.TASK_REQUEST,
            sender_id="orchestrator",
            receiver_id=planner.agent_id,
            action="plan_task",
            task_id=ctx.task_id,
            parameters={
                "task": ctx.user_input,
                "context": ctx.context
            }
        )
        
        # 3. 发送并等待回复
        # response = await self.__send_and_wait(plan_msg, planner.agent_id)
        response = await self._send(planner, plan_msg)
        
        if response.msg_type == MessageType.TASK_ERROR:
            return None
        
        # 4. 获取计划内容
        plan = self.__extract_data(response)
        
        # 5. 记录指标
        if ctx.metrics:
            ctx.metrics.record_message(plan_msg)
            ctx.metrics.record_message(response)
        
        return plan
    
    # ================================================================
    # 阶段2：执行子任务
    # ================================================================
    
    async def __execution_phase(self, ctx: TaskContext):
        """
        根据计划中的步骤和依赖关系，调度执行
        
        计划的格式：
        {
            "steps": [
                {"id": 1, "action": "search", "query": "...", "depends_on": []},
                {"id": 2, "action": "search", "query": "...", "depends_on": []},
                {"id": 3, "action": "search", "query": "...", "depends_on": [1, 2]},
                {"id": 4, "action": "summarize", "depends_on": [3]}
            ],
            "parallel_groups": [[1, 2]]
        }
        
        执行逻辑：
        1. 找出所有依赖已满足的步骤
        2. 如果可以并行，一起发送
        3. 等待完成，标记状态
        4. 重复直到所有步骤完成或失败
        """
        steps = ctx.plan.get("steps", [])
        parallel_groups = ctx.plan.get("parallel_groups", [])
        
        # 建立步骤索引
        step_map = {s["id"]: s for s in steps}
        pending_steps = set(step_map.keys())
        
        while pending_steps:
            # 1. 找出所有依赖已满足的步骤
            ready_steps = []
            for step_id in list(pending_steps):
                step = step_map[step_id]
                deps = step.get("depends_on", [])
                
                # 检查所有依赖是否已完成
                if all(d in ctx.completed_steps for d in deps):
                    ready_steps.append(step)
            
            if not ready_steps:
                # 没有就绪步骤，但有未完成步骤 → 可能存在循环依赖
                print(f"[ORCH] WARN: Detect possible cycle dependency, remaining steps: {pending_steps}")
                for sid in pending_steps:
                    ctx.failed_steps.add(sid)
                break
            
            # 2. 确定哪些可以并行执行
            parallel_batch = self.__find_parallel_batch(ready_steps, parallel_groups)
            
            # 3. 并行执行这一批
            if len(parallel_batch) > 1:
                tasks = [
                    self.__execute_single_step(ctx, step)
                    for step in parallel_batch
                ]
                await asyncio.gather(*tasks)
            else:
                await self.__execute_single_step(ctx, parallel_batch[0])
            
            # 4. 更新状态
            for step in parallel_batch:
                pending_steps.discard(step["id"])
    
    def __find_parallel_batch(
        self, 
        ready_steps: List[Dict], 
        parallel_groups: List[List[int]]
    ) -> List[Dict]:
        """
        从就绪步骤中找出一批可以并行执行的步骤
        
        如果 Planner 指定了 parallel_groups，
        且某个组的全部步骤都已就绪，就整组并行执行。
        否则，只取第一个就绪步骤串行执行。
        """
        for group in parallel_groups:
            group_ids = set(group)
            ready_ids = {s["id"] for s in ready_steps}
            
            if group_ids.issubset(ready_ids):
                # 这个组的全部步骤都就绪了，一起执行
                return [s for s in ready_steps if s["id"] in group_ids]
        
        # 没有匹配的并行组，只取第一个
        return ready_steps[:1]
    
    async def __execute_single_step(self, ctx: TaskContext, step: Dict):
        """
        执行单个子任务步骤
        
        流程：
        1. 找到能执行这个 action 的 Agent
        3. 构造消息
        4. 发送并等待结果
        5. 存储结果到上下文中
        """
        step_id = step["id"]
        action = step["action"]
        
        # 1. 通过路由表找到能执行此 action 的 Agent
        capable_agents = self.router.route_by_action(action)
        
        if not capable_agents:
            print(f"[ORCH] Step {step_id}: No Agent can execute {action}")
            ctx.failed_steps.add(step_id)
            return
        
        target_agent_id = capable_agents[0]  # 取第一个，多实例时可做负载均衡

        agent = self.agent_map[target_agent_id]
        
        upstream_context = self._collect_upstream_results(ctx, step)
        
        # 3. 构造消息
        msg = StructuredMessage(
            msg_type=MessageType.TASK_REQUEST,
            sender_id="orchestrator",
            receiver_id=target_agent_id,
            task_id=ctx.task_id,
            action=action,
            parameters={
                **step,
                "upstream_results": upstream_context  # 上游结果摘要
            }
        )
        
        # 4. 发送并等待回复
        response = await self._send(agent, msg)
        
        # 5. 记录结果
        ctx.step_results[step_id] = response
        
        if response.msg_type == MessageType.TASK_ERROR:
            ctx.failed_steps.add(step_id)
        else:
            ctx.completed_steps.add(step_id)
        
        # 6. 记录指标
        if ctx.metrics:
            ctx.metrics.record_message(msg)
            ctx.metrics.record_message(response)
    
    # ================================================================
    # 阶段3：总结
    # ================================================================

    def _collect_upstream_results(self, ctx: TaskContext, step: Dict) -> Dict:
        """
        收集上游步骤的结果，作为当前步骤的上下文
        
        只传递摘要信息，不传完整数据（完整数据在共享内存中）。
        """
        upstream = {}
        
        for dep_id in step.get("depends_on", []):
            if dep_id in ctx.step_results:
                response = ctx.step_results[dep_id]
                upstream[dep_id] = {
                    "action": step.get("action", ""),
                    "result_type": response.result_type,
                    "summary": str(response.result_data)[:200] if response.result_data else ""
                }
        
        return upstream
    
    async def __summary_phase(self, ctx: TaskContext) -> Dict[str, Any]:
        """
        将所有子任务结果发送给 Summarizer 进行总结
        
        流程：
        1. 收集所有已完成步骤的 SDE 指针
        2. 构造总结消息
        3. 发送给 Summarizer
        4. 返回最终结果
        """

        summarizer = self.role_map.get("summarizer")
        if not summarizer:
                # 没有 Summarizer，直接返回所有步骤结果
                return {
                    "completed": list(ctx.completed_steps),
                    "failed": list(ctx.failed_steps),
                    "results": {
                        sid: self.__extract_data(ctx.step_results[sid])
                        for sid in ctx.completed_steps
                    }
                }
        
        # 收集所有步骤结果
        step_results_summary = {}
        for step_id in sorted(ctx.completed_steps):
            response = ctx.step_results[step_id]
            step_results_summary[step_id] = self.__extract_data(response)
        
        # 构造总结消息
        msg = StructuredMessage(
            msg_type=MessageType.TASK_REQUEST,
            sender_id="orchestrator",
            receiver_id=summarizer.agent_id,
            task_id=ctx.task_id,
            action="summarize",
            parameters={
                "user_query": ctx.user_input,
                "total_steps": len(ctx.plan.get("steps", [])),
                "completed": len(ctx.completed_steps),
                "failed": len(ctx.failed_steps),
                "strategy": ctx.plan.get("strategy", ""),
                "step_results": step_results_summary
            }
        )
        
        # 发送并等待回复
        response = await self._send(summarizer, msg)
        
        if ctx.metrics:
            ctx.metrics.record_message(msg)
            ctx.metrics.record_message(response)
        
        return self.__extract_data(response)
    
    # ================================================================
    # 通信辅助
    # ================================================================
    
    async def __send_and_wait(
        self,
        msg: StructuredMessage,
        target_id: str,
        timeout: float = 60.0
    ) -> StructuredMessage:
        """
        发送消息并同步等待回复
        
        实现原理：
        1. 创建一个 asyncio.Queue 作为响应容器
        2. 在 MessageBus 上注册一个临时处理器
        3. 当回复到达时，MessageBus 调用这个处理器，把回复放入 Queue
        4. 这里等待 Queue 有数据，拿到回复后返回
        
        这就是"异步发送、同步等待"的实现方式。
        """
        # 创建一次性响应队列
        response_queue = asyncio.Queue(maxsize=1)
        
        # 注册临时处理器：当回复到达时，把回复放入队列
        self.bus.register_temp_handler(
            msg.msg_id,
            lambda resp: response_queue.put_nowait(resp)
        )
        
        # 发送消息
        await self.bus.send(msg, target=target_id)
        
        # 等待回复（带超时）
        try:
            response = await asyncio.wait_for(
                response_queue.get(),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            # 超时，返回错误消息
            return StructuredMessage(
                msg_type=MessageType.TASK_ERROR,
                task_id=msg.task_id,
                result_data={"error": f"等待 {target_id} 回复超时 ({timeout}s)"}
            )
            
    def __extract_data(self, response: StructuredMessage) -> Any:
        """
        从回复消息中提取实际数据
        
        数据可能在两个地方：
        1. result_type == "inline" → 直接从 result_data 取
        2. result_type == "shm_pointer" → 从共享内存读取
        """
        if response.result_type == "shm_pointer":
            offset = response.result_data.get("offset") or \
                     response.result_data.get("plan_offset") or \
                     response.result_data.get("result_offset")
            if offset is not None:
                return self.shm.read_task(offset)
        
        return response.result_data
    
    def get_task_status(self, task_id: str) -> Dict:
        """查询任务状态"""
        ctx = self.active_tasks.get(task_id)
        if not ctx:
            # 尝试从历史中查找
            for h in self.task_history:
                if h.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "user_input": h.user_input[:100],
                        "duration": h.finished_at - h.created_at,
                        "completed_steps": len(h.completed_steps),
                        "failed_steps": len(h.failed_steps)
                    }
            return {"error": "任务不存在"}
        
        return {
            "task_id": task_id,
            "status": "running",
            "user_input": ctx.user_input[:100],
            "elapsed": time.time() - ctx.created_at,
            "total_steps": len(ctx.plan.get("steps", [])) if ctx.plan else 0,
            "completed": len(ctx.completed_steps),
            "failed": len(ctx.failed_steps),
            "pending": len(ctx.plan.get("steps", [])) - len(ctx.completed_steps) - len(ctx.failed_steps) if ctx.plan else 0
        }
        
    def get_history(self, limit: int = 10) -> List[Dict]:
        """获取历史任务摘要"""
        return [
            {
                "task_id": ctx.task_id,
                "user_input": ctx.user_input[:100],
                "completed_steps": len(ctx.completed_steps),
                "failed_steps": len(ctx.failed_steps),
                "duration": ctx.finished_at - ctx.created_at
            }
            for ctx in self.task_history[-limit:]
        ]

    async def _send(self, agent: BaseAgent, msg: StructuredMessage) -> StructuredMessage:
        """
        发送消息给 Agent 并等待回复
        
        原型阶段：直接调用 agent.handle_message()
        后续可替换为：bus.send(msg, target=agent.agent_id)
        """
        # 如果配置了 MessageBus，走总线
        if self.bus:
            return await self.bus.send(msg, target=agent.agent_id)
        
        response = await agent.handle_message(msg)
        
        if response is None:
            # Agent 没有返回，构造错误
            response = StructuredMessage(
                msg_type=MessageType.TASK_ERROR,
                sender_id=agent.agent_id,
                receiver_id="orchestrator",
                task_id=msg.task_id,
                result_data={"error": "Agent 未返回结果"}
            )
        
        return response