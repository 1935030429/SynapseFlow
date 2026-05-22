class PlannerAgent(BaseAgent):
    """规划Agent - 负责任务分解与规划"""
    
    def __init__(self, *args, llm_model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = self._init_llm(llm_model)
        
    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            agent_id=self.agent_id,
            role="planner",
            skills=["task_decomposition", "workflow_planning", "dependency_analysis"],
            supported_actions=["plan_task", "adjust_plan", "evaluate_progress"],
            embedding_model="all-MiniLM-L6-v2"
        )
    
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        """分解任务为子任务序列"""
        # 1. 搜索相关历史记忆
        relevant_memories = await self.search_memory(msg.parameters['task_description'])
        
        # 2. 基于历史经验生成计划
        plan = await self._generate_plan(
            task=msg.parameters['task_description'],
            context=relevant_memories
        )
        
        # 3. 将计划存储为记忆
        plan_embedding = self.state_transfer.encode_state(str(plan))
        memory = MemoryUnit(
            source_agent=self.agent_id,
            task_id=msg.task_id,
            task_theme=msg.parameters.get('task_theme', ''),
            summary=plan['summary'],
            strategy=plan['strategy'],
            embedding=plan_embedding,
            tags=['plan', msg.parameters.get('task_type', 'general')]
        )
        await self.memory.store(memory)
        
        return StructuredMessage(
            msg_id=self._gen_id(),
            msg_type=MessageType.TASK_RESULT,
            sender_id=self.agent_id,
            receiver_id=msg.sender_id,
            timestamp=time.time(),
            action=msg.action,
            result_data=plan,
            result_type="json",
            task_id=msg.task_id
        )
    
    async def _generate_plan(self, task: str, context: List[MemoryUnit]) -> dict:
        """基于LLM生成计划"""
        # 构建精简的上下文
        context_summary = "\n".join([
            f"- [{mem.task_theme}] {mem.summary[:200]}"
            for mem in context[:3]
        ])
        
        prompt = f"""Based on task and historical context, create execution plan.
Task: {task}
Historical Context:
{context_summary}

Output JSON format:
{{"steps": [...], "dependencies": [...], "estimated_tokens": ..., "strategy": "..."}}
"""
        # LLM调用（略）
        return {"steps": [], "strategy": "sequential"}


class RetrieverAgent(BaseAgent):
    """检索Agent - 负责信息检索"""
    
    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            agent_id=self.agent_id,
            role="retriever",
            skills=["semantic_search", "knowledge_retrieval", "fact_checking"],
            supported_actions=["search", "verify", "extract_evidence"],
            embedding_model="all-MiniLM-L6-v2"
        )
    
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        """执行检索任务"""
        query = msg.parameters['query']
        
        # 1. 先搜索共享记忆
        memory_results = await self.search_memory(query)
        
        # 2. 对外部知识源检索
        external_results = await self._external_search(query)
        
        # 3. 合并结果并提取embedding用于传递
        combined = self._merge_results(memory_results, external_results)
        result_embedding = self.state_transfer.encode_state(combined['summary'])
        
        return StructuredMessage(
            msg_id=self._gen_id(),
            msg_type=MessageType.TASK_RESULT,
            sender_id=self.agent_id,
            receiver_id=msg.sender_id,
            timestamp=time.time(),
            action=msg.action,
            result_data={
                "text_summary": combined['summary'],
                "embedding": result_embedding.tolist(),  # 直接传递embedding
                "sources": combined['sources']
            },
            result_type="mixed",  # 混合类型：文本+向量
            task_id=msg.task_id
        )


class ExecutorAgent(BaseAgent):
    """执行Agent - 支持CodeAct模式"""
    
    def __init__(self, *args, sandbox_type: str = "docker", **kwargs):
        super().__init__(*args, **kwargs)
        self.sandbox = CodeSandbox(sandbox_type)
        
    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            agent_id=self.agent_id,
            role="executor",
            skills=["code_execution", "tool_usage", "data_processing"],
            supported_actions=["execute_code", "call_api", "process_data"],
            embedding_model="all-MiniLM-L6-v2"
        )
    
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        """执行代码或工具调用"""
        if msg.action == "execute_code":
            result = await self.sandbox.execute(
                code=msg.parameters['code'],
                timeout=msg.parameters.get('timeout', 30)
            )
        else:
            result = await self._execute_action(msg.action, msg.parameters)
        
        # 生成执行结果的embedding
        result_emb = self.state_transfer.encode_state(str(result))
        
        # 存储执行经验
        memory = MemoryUnit(
            source_agent=self.agent_id,
            task_id=msg.task_id,
            task_theme=msg.parameters.get('context', ''),
            summary=f"Executed: {msg.action}",
            evidence=[str(result)],
            embedding=result_emb,
            tags=['execution', msg.action]
        )
        await self.memory.store(memory)
        
        return StructuredMessage(
            msg_id=self._gen_id(),
            msg_type=MessageType.TASK_RESULT,
            sender_id=self.agent_id,
            receiver_id=msg.sender_id,
            timestamp=time.time(),
            result_data=result,
            result_type="json",
            task_id=msg.task_id
        )


class SummarizerAgent(BaseAgent):
    """总结Agent"""
    
    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            agent_id=self.agent_id,
            role="summarizer",
            skills=["content_summarization", "insight_extraction", "report_generation"],
            supported_actions=["summarize", "extract_key_points", "generate_report"],
            embedding_model="all-MiniLM-L6-v2"
        )
    
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        """总结生成"""
        # 接收可能包含embedding的结果
        if msg.result_type == "mixed" and "embedding" in msg.result_data:
            # 直接使用传递的embedding，无需重新编码
            incoming_emb = np.array(msg.result_data['embedding'])
            text_content = msg.result_data['text_summary']
        else:
            incoming_emb = None
            text_content = str(msg.result_data)
        
        summary = await self._generate_summary(text_content, incoming_emb)
        
        return StructuredMessage(
            msg_id=self._gen_id(),
            msg_type=MessageType.TASK_RESULT,
            sender_id=self.agent_id,
            receiver_id=msg.sender_id,
            timestamp=time.time(),
            result_data=summary,
            result_type="text",
            task_id=msg.task_id
        )