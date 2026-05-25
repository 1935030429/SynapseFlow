import json
from typing import Any, Dict, List, Optional

from Project.SynapseFlow.src.memory.memory_unit import MemoryUnit
from src.agents.base_agent import BaseAgent
from src.protocol.messages import MessageType, StructuredMessage


class PlannerAgent(BaseAgent):
    """规划Agent - 负责任务分解与规划"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_mas = kwargs.get('latent_steps', 50)
        
    def get_capability(self) -> List[str]:
        return ["plan_task", "adjust_plan"]
    
    async def process_task(self, msg: StructuredMessage) -> Optional[StructuredMessage]:
        """分解任务为子任务序列"""
        # 1. 搜索相关历史记忆
        # relevant_memories = await self.search_memory(msg.parameters['task_description'])
        
        # # 2. 基于历史经验生成计划
        # plan = await self._generate_plan(
        #     task=msg.parameters['task_description'],
        #     context=relevant_memories
        # )
        
        # # 3. 将计划存储为记忆
        # plan_embedding = self.state_transfer.encode_state(str(plan))
        # memory = MemoryUnit(
        #     source_agent=self.agent_id,
        #     task_id=msg.task_id,
        #     task_theme=msg.parameters.get('task_theme', ''),
        #     summary=plan['summary'],
        #     strategy=plan['strategy'],
        #     embedding=plan_embedding,
        #     tags=['plan', msg.parameters.get('task_type', 'general')]
        # )
        # await self.memory.store(memory)
        
        # return StructuredMessage(
        #     msg_id=self._gen_id(),
        #     msg_type=MessageType.TASK_RESULT,
        #     sender_id=self.agent_id,
        #     receiver_id=msg.sender_id,
        #     timestamp=time.time(),
        #     action=msg.action,
        #     result_data=plan,
        #     result_type="json",
        #     task_id=msg.task_id
        # )
        action = msg.action
        
        if action == "plan_task":
            return await self._plan_task(msg)
        elif action == "adjust_plan":
            return await self._adjust_plan(msg)
        
        else:
            return self.make_error(
                action=msg.action,
                error_msg=f"Invalid action: Planner do not support{action}",
                receiver_id=msg.sender_id
            )
        
    async def _plan_task(self, msg: StructuredMessage):
        """
        输入:
            msg.parameters = {
                "task": "用户任务描述",
                "context": {...}  # 额外上下文
            }
        
        输出:
            消息中包含计划在共享内存中的偏移量
        """
        task_desc = msg.parameters.get("task", "")
        context = msg.parameters.get("context", {})
        task_id = msg.task_id or self._generate_id()
        
        # Step1: find history memory
        query_text  = f"Task Planning: {task_desc}"
        relevant_memories = await self.query_memory(
            query_text=query_text,
            top_k=5
        )
        
        prompt = self._build_plan_prompt(task_desc=task_desc, context=context, relevant_memories=relevant_memories)
        
        result = await self.generate_text_async(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["```\n"]
        )
        
        plan_text = result["context"]
        
        # Step 4: parse plan
        plan = self.parse_plan(plan_text=plan_text)
        
        # Step 5: store plan
        plan_offset = await self.write_to_shm(plan)
        
        memory_id = await self._store_plan_memory(
            plan=plan, task_desc=task_desc, task_id=task_id
        )
        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="plan_task",
            result_data={
                "plan_offset": plan_offset,
                "plan_summary": {
                    "total_steps": len(plan.get("steps", [])),
                    "strategy": plan.get("strategy", ""),
                    "parallel_groups": len(plan.get("parallel_groups", []))
                },
                "tokens_used": result["tokens_used"],
                "memory_used": len(relevant_memories)
            },
            result_type="shm_pointer",
            memory_refs=[m.memory_id for m in relevant_memories],
            receiver_id=msg.sender_id
        )
        
    async def _store_plan_memory(self, plan: Dict, task_desc: str, task_id: str):
        action_tags = List(set(step.get("action", "") for step in plan.get("steps", [])))
        memory_id = await self.store_memory(
            summary=f"Task Planning: {plan.get('strategy', '')}",
            task_id=task_id,
            evidence=[json.dumps(plan, ensure_ascii=False)],
            strategy=plan.get("strategy", ""),
            conclusion="",
            tags=["plan", "planner"] + action_tags,
            parent_ids=[] # 计划是顶层记忆，没有前置
        )

        return memory_id
        
    def parse_plan(self, plan_text: str) -> Dict[str, Any]:
        """
        从 LLM 输出中提取计划 JSON
        容错处理：
        直接解析
        从 json... 代码块提取
        从 ... 代码块提取
        从 { ... } 提取
        """
        text = plan_text.strip()
        try:
            return self._valiate_plan(json.loads(text))
        except json.JSONDecodeError:
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end > start:
                    try:
                        return self._valiate_plan(json.loads(text[start:end].strip()))
                    except json.JSONDecodeError as e:
                        print(f"Plan JSON Decode Error: {e}")
                        
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                if end > start:
                    try:
                        json_str = text[start:end].strip()
                        # Remove any language specifier like 'javascript', 'python', etc.
                        lines = json_str.split('\n')
                        if lines and not lines[0].startswith('{'):
                            json_str = '\n'.join(lines[1:])
                        return self._validate_plan(json.loads(json_str))
                    except json.JSONDecodeError as e:
                        print(f"Plan JSON Decode Error: {e}")
            else:
                brace_start = text.find("{")
                brace_end = text.rfind("}")
                if brace_start >= 0 and brace_end > brace_start:
                    try:
                        return self._validate_plan(json.loads(text[brace_start:brace_end + 1]))
                    except json.JSONDecodeError:
                        pass
        print(f"[{self.agent_id}] Fail to parse plan, original output: {text[:200]}")
        return {
            "steps": [],
            "parallel_groups": [],
            "strategy": "parse_failed",
            "error": "fail to parse plan",
            "raw_output": text[:500]
        }
        
        
    def get_plan_from_shm(self, offset: int) -> Dict:
        return self.read_from_shm(offset)
        
            
    def _valiate_plan(self, plan: json):
        """
        验证并补全计划结构
        """
        if "steps" not in plan:
            plan["steps"] = []
        if "parallel_groups" not in plan:
            plan["parallel_groups"] = []
        if "strategy" not in plan:
            plan["strategy"] = ""
            
        valid_actions = {"search", "execute_code", "summarize", "verify"}
        
        for step in plan["steps"]:
            if "id" not in step:
                step["id"] = len(plan["steps"])
            if "action" not in step:
                step["action"] = "search"
            if "depends_on" not in step:
                step["depends_on"] = []
            if "agent" not in step:
                action_to_agent = {
                    "search": "retriever",
                    "verify": "retriever",
                    "execute_code": "executor",
                    "summarize": "summarizer"
                }
                step["agent"] = action_to_agent.get(step["action"], "retriever")
            if step["action"] not in valid_actions:
                step["action"] = "search"

        return plan
    
    async def _adjust_plan(self, msg: StructuredMessage) -> StructuredMessage:
        """
        调整已有计划
        
        输入:
            msg.parameters = {
                "task": "Task Description",
                "original_plan": {...},  # Original Plan
                "feedback": "adjustment Suggestions"
            }
        """
        task_desc = msg.parameters.get("task", "")
        original_plan = msg.parameters.get("original_plan", {})
        feedback = msg.parameters.get("feedback", "")
        task_id = msg.task_id or self._generate_id()
        
        # 构建调整 prompt
        prompt = f"""Original Plan:
{json.dumps(original_plan, ensure_ascii=False, indent=2)}

Adjustment Suggestions: {feedback}

Task: {task_desc}

Please output the adjusted plan in JSON format."""
        result = await self.generate_text_async(
            prompt=prompt,
            max_tokens=1024,
            # temperature=0.1,
            # stop=["```\n"]
        )
        plan = self.parse_plan(plan_text=result["context"])
        
        plan_offset = await self.write_to_shm(plan)
        
        memory_id = await self._store_plan_memory(
            plan=plan, task_desc=f"adjust: {task_desc}", task_id=task_id
        )
        
        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="adjust_plan",
            result_data={
                "plan_offset": plan_offset,
                "plan_summary": {
                    "total_steps": len(plan.get("steps", [])),
                    "strategy": plan.get("strategy", "")
                }
            },
            result_type="shm_pointer",
            receiver_id=msg.sender_id
        )

    def _build_plan_prompt(
        self,
        task_desc: str,
        context: Dict[str, Any],
        relevant_memories: List[MemoryUnit]
    ) -> str:
        """
        构建规划 prompt
        
        特点：精简上下文，只用记忆摘要，不传完整记忆内容。
        """
        # 历史记忆摘要（最多3条，每条最多100字）
        history_parts = []
        for i, mem in enumerate(relevant_memories, 1):
            history_parts.append(f"{i}. {mem.summary[:100]}")
        history_text = "\n".join(history_parts) if history_parts else "no related history"

        # 上下文信息
        context_text = ""
        if context:
            context_items = [f"  {k}: {v}" for k, v in context.items()]
            context_text = "\n".join(context_items)
        
        return f"""You are a task planning expert. Decompose user tasks into executable steps.
    # User Task
    {task_desc}
    # Context
    {context_text if context_text else "None"}
    # Historical Reference Plans
    {history_text}
    # Output Format
    Please output the plan in JSON format.
    ```json
    {{
    "steps": [
        {{
            "id": 1,
            "action": "search",
            "agent": "retriever",
            "query": "Retrieval Content Description",
            "depends_on": []
        }},
        {{
            "id": 2,
            "action": "execute_code",
            "agent": "executor",
            "code": "Executable code",
            "depends_on": [1]
        }},
        {{
            "id": 3,
            "action": "summarize",
            "agent": "summarizer",
            "depends_on": [1, 2]
        }}
    ],
    "parallel_groups": [[1]],
    "strategy": "Brief description of execution strategy"
    }}
    # Rule
    1. The action type which is supported: search, execute_code, summarize.
    2. depends_on: List the IDs of the preceding steps that this step depends on.
    3. parallel_groups: List the IDs of the steps that can be executed in parallel.
    4. Step ID starts from 1.
    5.Only output json, not other text. 
    """
    
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








