from typing import List, Dict, Any
from Project.SynapseFlow.src.memory.memory_unit import MemoryUnit
from src.agents.base_agent import BaseAgent
from src.protocol.messages import StructuredMessage, MessageType
import json

class SummarizerAgent(BaseAgent):
    """总结Agent"""

    def __init__(self) -> None:
        super().__init__()
    
    def get_capability(self) -> List[str]:
        return ["summarize", "generate_report"]
    
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        """总结生成"""
        action = msg.action

        if action == "summarize":
            return await self._summarize(msg)
        elif action == "generate_report":
            return await self._generate_report(msg)
        else:
            return self.make_error(
                action=action,
                error_msg=f"Summarizer do not support action: {action}",
                receiver_id=msg.sender_id
            )

    async def _summarize(self, msg: StructuredMessage) -> StructuredMessage:
        """
        对多步骤结果进行总结

        这是 Summarizer 的主要工作。
        接收 Orchestrator 汇总的所有步骤结果，生成最终输出。

        输入:
            msg.parameters = {
                "user_query": "用户原始问题",
                "total_steps": 3,
                "completed": 3,
                "failed": 0,
                "strategy": "先检索后总结",
                "step_results": {
                    1: {...},  # 步骤1的结果
                    2: {...},  # 步骤2的结果
                    3: {...}   # 步骤3的结果
                }
            }
        """
        user_query = msg.parameters.get("user_query", "")
        step_results = msg.parameters.get("step_results", {})
        total_steps = msg.parameters.get("total_steps", 0)
        completed = msg.parameters.get("completed", 0)
        failed = msg.parameters.get("failed", 0)
        task_id = msg.task_id or self._generate_id()

        # ──── 步骤1：收集所有上游结果 ────
        # 从共享内存读取完整数据（消息中只有摘要）
        all_data = self._collect_upstream_data(msg)

        # ──── 步骤2：查询历史记忆 ────
        # 查找相似的历史总结，获取风格偏好和模板
        relevant_memories = await self.query_memory(
            query_text=f"总结: {user_query}",
            top_k=3
        )

        # ──── 步骤3：构建总结 prompt ────
        prompt = self._build_summarize_prompt(
            user_query=user_query,
            data=all_data,
            completed=completed,
            failed=failed,
            relevant_memories=relevant_memories
        )

        # ──── 步骤4：调用 LLM 生成总结 ────
        result = await self.generate_text_async(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.5
        )

        summary_text = result["content"]

        # ──── 步骤5：存储总结到记忆 ────
        memory_id = await self._store_summary_memory(
            user_query=user_query,
            summary=summary_text,
            task_id=task_id,
            step_count=completed,
            parent_ids=[m.memory_id for m in relevant_memories]
        )

        # ──── 步骤6：返回 ────
        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="summarize",
            result_data={
                "summary": summary_text,
                "tokens_used": result["tokens_used"],
                "memory_used": len(relevant_memories),
                "steps_completed": completed,
                "steps_failed": failed
            },
            result_type="inline",
            memory_refs=[m.memory_id for m in relevant_memories],
            receiver_id=msg.sender_id
        )

    async def _generate_report(self, msg: StructuredMessage) -> StructuredMessage:
        """
        生成结构化报告

        与 summarize 类似，但输出更结构化的格式。
        适用于需要分章节、分要点的正式报告。
        """
        user_query = msg.parameters.get("user_query", "")
        step_results = msg.parameters.get("step_results", {})
        report_format = msg.parameters.get("format", "markdown")
        task_id = msg.task_id or self._generate_id()

        # 收集数据
        all_data = self._collect_upstream_data(msg)

        # 查询历史报告模板
        relevant_memories = await self.query_memory(
            query_text=f"报告: {user_query}",
            top_k=3,
            filter_tags=["report", "summary"]
        )

        # 构建报告 prompt
        prompt = self._build_report_prompt(
            user_query=user_query,
            data=all_data,
            report_format=report_format,
            relevant_memories=relevant_memories
        )

        # 调用 LLM
        result = await self.generate_text_async(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.1
        )

        # 存储
        await self._store_summary_memory(
            user_query=user_query,
            summary=result["content"],
            task_id=task_id,
            parent_ids=[m.memory_id for m in relevant_memories],
            tags=["report"]
        )

        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="generate_report",
            result_data={
                "report": result["content"],
                "format": report_format,
                "tokens_used": result["tokens_used"]
            },
            result_type="inline",
            receiver_id=msg.sender_id
        )

    def _collect_upstream_data(self, msg: StructuredMessage) -> Dict[int, Any]:
        """
        收集所有上游步骤的完整数据

        消息中的 step_results 可能只包含摘要，
        如果数据在共享内存中（result_type="shm_pointer"），
        需要从共享内存读取完整内容。
        """
        step_results: Dict = msg.parameters.get("step_results", {})
        collected = {}

        for step_id, result in step_results.items():
            step_id = int(step_id) if isinstance(step_id, str) else step_id

            if isinstance(result, dict):
                # 检查是否需要在共享内存中读取
                if result.get("result_type") == "shm_pointer":
                    offset = (
                        result.get("offset") or
                        result.get("plan_offset") or
                        result.get("result_offset")
                    )
                    if offset:
                        try:
                            collected[step_id] = self.read_from_shm(offset)
                        except:
                            collected[step_id] = result.get("summary", str(result))
                    else:
                        collected[step_id] = result
                else:
                    collected[step_id] = result
            else:
                collected[step_id] = result

        return collected

    def _build_summarize_prompt(
        self,
        user_query: str,
        data: Dict[int, Any],
        completed: int,
        failed: int,
        relevant_memories: List
    ) -> str:
        """构建总结 prompt"""

        # 格式化上游数据
        data_text = self._format_collected_data(data)

        # 历史记忆摘要
        memory_text = self._format_memory_context(relevant_memories)

        return f"""You are an expert in summarizing information. Please answer the user's question based on the following information.

## User Question
{user_query}

## Data Collected
{data_text}

## Execution Status
- Total Steps: {completed + failed}
- Completed: {completed}
- Failed: {failed}

## Historical Context
{memory_text if memory_text else "None"}

## Requirements
1. Directly answer the user's question
2. Information is accurate and based on the collected data
3. If data is insufficient, clearly point out the lack of information
4. Use clear and concise language
5. If multiple sources are available, synthesize them to form a conclusion
6. Do not fabricate information not found in the data

Directly output summary content."""

    def _build_report_prompt(
        self,
        user_query: str,
        data: Dict[int, Any],
        report_format: str,
        relevant_memories: List
    ) -> str:
        """构建报告 prompt"""
        data_text = self._format_collected_data(data)
        memory_text = self._format_memory_context(relevant_memories)

        return f"""Please generate a structured report based on the following information.

## Report Topic
{user_query}

## Data Source
{data_text}

## Historical Report Template (Reference Format)
{memory_text if memory_text else "None"}

## Output Format
Please use {report_format} format, including the following sections:
1. Overview
2. Detailed Analysis
3. Key Findings
4. Suggestions (if applicable)
5. Data Source

Please directly output report content."""

    def _format_collected_data(self, data: Dict[int, Any]) -> str:
        """将收集的数据格式化为文本"""
        if not data:
            return "无数据"

        parts = []
        for step_id, content in sorted(data.items()):
            parts.append(f"\n### 步骤 {step_id} 的结果")

            if isinstance(content, dict):
                # 提取关键字段
                if "summary" in content:
                    parts.append(f"摘要: {content['summary']}")
                if "items" in content:
                    parts.append(f"条目数: {len(content['items'])}")
                    for i, item in enumerate(content["items"][:5], 1):
                        parts.append(
                            f"  {i}. {item.get('title', '')}: "
                            f"{str(item.get('content', ''))[:200]}"
                        )
                if "stdout" in content:
                    parts.append(f"执行输出:\n```\n{content['stdout'][:500]}\n```")
                if "exit_code" in content:
                    parts.append(f"退出码: {content['exit_code']}")
                # 兜底：序列化整个内容
                if not any(k in content for k in ["summary", "items", "stdout"]):
                    parts.append(json.dumps(content, ensure_ascii=False)[:500])

            elif isinstance(content, str):
                parts.append(content[:1000])
            else:
                parts.append(str(content)[:500])

        return "\n".join(parts)

    def _format_memory_context(self, memories: List[MemoryUnit]) -> str:
        """格式化历史记忆为上下文"""
        if not memories:
            return ""

        parts = []
        for i, mem in enumerate(memories[:3], 1):
            parts.append(f"{i}. {mem.summary[:150]}")
            if mem.strategy:
                parts.append(f"    Strategy: {mem.strategy[:100]}")
        return "\n".join(parts)
    
    async def _store_summary_memory(
        self,
        user_query: str,
        summary: str,
        task_id: str,
        step_count: int = 0,
        parent_ids: List[str] = None,
        tags: List[str] = None
    ) -> str:
        """存储总结到记忆库"""
        return await self.store_memory(
            summary=f"总结: {user_query[:80]}",
            task_id=task_id,
            evidence=[summary],
            strategy=f"综合 {step_count} 个步骤的结果生成总结",
            conclusion=summary[:200],
            tags=tags or ["summary", "final"],
            parent_ids=parent_ids or []
        )
