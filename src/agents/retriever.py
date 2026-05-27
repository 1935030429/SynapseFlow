from typing import List, Dict, Any
import json
from src.agents.base_agent import BaseAgent
from src.protocol.messages import StructuredMessage, MessageType
from src.memory.memory_unit import MemoryUnit

class RetrieverAgent(BaseAgent):
    """检索Agent - 负责信息检索"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_capability(self) -> List[str]:
        return ["search", "verify"]
    
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        """执行检索任务"""
        action = msg.action

        if action == "search":
            return await self._search(msg)
        elif action == "verify":
            return await self._verify(msg)
        else:
            return self.make_error(
                action=action,
                error_msg=f"Retriever 不支持此动作: {action}",
                receiver_id=msg.sender_id
            )

    async def _search(self, msg: StructuredMessage) -> StructuredMessage:
        """
        执行检索任务

        输入:
            msg.parameters = {
                "query": "检索内容",
                "top_k": 5,             # 可选，返回数量
                "upstream_results": {}  # 上游步骤的结果（Orchestrator 传入）
            }

        检索优先级：
        1. 共享记忆（最快，语义匹配）
        2. 外部知识库（模拟，实际可接搜索引擎/向量数据库）
        3. LLM 整理（去重、排序、摘要）
        """
        query = msg.parameters.get("query", "")
        top_k = msg.parameters.get("top_k", 5)
        task_id = msg.task_id or self._generate_id()

        # 检查是否是纯文本模式（无结构化参数，query 本身就是整段文本）
        if not query and msg.parameters.get("task"):
            query = msg.parameters.get("task", "")

        # ──── 步骤1：搜索共享记忆 ────
        memory_results = await self.query_memory(
            query_text=query,
            top_k=top_k
        )

        # ──── 步骤2：判断是否需要外部检索 ────
        external_results = []
        memory_sufficient = False

        # 如果记忆命中足够多且相关度高，可以跳过外部检索
        if len(memory_results) >= top_k:
            memory_sufficient = True
        else:
            # 需要外部检索（模拟）
            external_results = await self._external_search(query, top_k - len(memory_results))

        # ──── 步骤3：合并并整理结果 ────
        all_results = self._merge_results(memory_results, external_results)

        # ──── 步骤4：调用 LLM 整理 ────
        formatted_results = await self._format_results(query, all_results, top_k)

        # ──── 步骤5：写入共享内存 ────
        result_offset = self.write_to_shm(formatted_results)

        # ──── 步骤6：存储检索经验到记忆 ────
        memory_id = await self._store_search_memory(
            query=query,
            results=formatted_results,
            task_id=task_id,
            from_memory=len(memory_results),
            from_external=len(external_results)
        )

        # ──── 步骤7：构造返回消息 ────
        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="search",
            result_data={
                "result_offset": result_offset,
                "summary": {
                    "total_results": len(formatted_results.get("items", [])),
                    "from_memory": len(memory_results),
                    "from_external": len(external_results),
                    "memory_sufficient": memory_sufficient
                }
            },
            result_type="inline",
            memory_refs=[m.memory_id for m in memory_results],
            receiver_id=msg.sender_id
        )

    async def _verify(self, msg: StructuredMessage) -> StructuredMessage:
        """
        事实验证

        输入:
            msg.parameters = {
                "claim": "需要验证的声明",
                "evidence": "已有的证据（可选）"
            }
        """
        claim = msg.parameters.get("claim", "")
        evidence = msg.parameters.get("evidence", "")
        task_id = msg.task_id or self._generate_id()

        # 1. 搜索相关记忆（看之前是否验证过类似声明）
        similar_checks = await self.query_memory(
            query_text=f"验证: {claim}",
            top_k=3
        )

        # 2. 构建验证 prompt
        prompt = self._build_verify_prompt(claim, evidence, similar_checks)

        # 3. 调用 LLM
        result = await self.generate_text_async(
            prompt=prompt,
            max_tokens=512
        )

        # 4. 解析验证结果
        verification = self._parse_verification(result["content"])

        # 5. 存储验证经验
        await self._store_search_memory(
            query=f"验证: {claim}",
            results=verification,
            task_id=task_id
        )

        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="verify",
            result_data=verification,
            result_type="inline",
            receiver_id=msg.sender_id
        )

    async def _external_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        外部知识库检索（模拟）

        实际实现时，这里可以接：
        - 向量数据库（ChromaDB/Milvus）
        - 搜索引擎 API（Elasticsearch/Bing/Google）
        - 本地文档库
        - Web 爬虫

        当前用 LLM 模拟外部知识。
        """
        # 调用 LLM 生成模拟的检索结果
        # 实际场景中，这里应该查询真正的知识库
        prompt = f"""你是一个知识检索系统。请为以下查询提供 {top_k} 条相关信息。

查询: {query}

请以 JSON 格式返回：
```json
{{
    "items": [
        {{
            "title": "标题",
            "content": "内容摘要",
            "source": "来源",
            "relevance": 0.95
        }}
    ]
}}
```"""

        result = await self.generate_text_async(
            prompt=prompt,
            max_tokens=1024
        )

        # 解析结果
        try:
            parsed = self._parse_json(result["content"])
            return parsed.get("items", [])
        except:
            return []

    def _merge_results(
        self,
        memory_results: List[MemoryUnit],
        external_results: List[Dict]
    ) -> List[Dict]:
        """
        合并记忆结果和外部检索结果

        记忆结果优先（已经过验证，置信度高）。
        """
        merged = []

        # 先添加记忆结果
        for mem in memory_results:
            merged.append({
                "title": mem.task_theme or "记忆条目",
                "content": mem.summary,
                "source": f"Shared Memory (Agent: {mem.source_agent})",
                "relevance": mem.confidence,
                "from": "memory",
                "memory_id": mem.memory_id
            })

        # 再添加外部结果
        for ext in external_results:
            ext["from"] = "external"
            merged.append(ext)

        return merged

    async def _format_results(
        self,
        query: str,
        results: List[Dict],
        top_k: int
    ) -> Dict[str, Any]:
        """
        调用 LLM 整理检索结果

        整理包括：
        - 去重
        - 按相关性排序
        - 生成摘要
        """
        if not results:
            return {"items": [], "summary": "未找到相关信息"}

        # 如果结果少，不需要 LLM 整理
        if len(results) <= 3:
            return {
                "items": results[:top_k],
                "summary": f"找到 {len(results)} 条相关信息",
                "query": query
            }

        # 调用 LLM 整理
        results_text = json.dumps(results, ensure_ascii=False, indent=2)
        prompt = f"""Please 整理以下检索结果，去重并按相关性排序。

Original Query: {query}

Retrieval Results:
{results_text[:2000]}

Please return the formatted results in a JSON format:
```json
{{
    "items": [
        {{
            "title": "Title",
            "content": "Content",
            "source": "Source",
            "relevance": 0.95,
            "from": "memory/external"
        }}
    ],
    "summary": "Summary"
}}
```"""

        result = await self.generate_text_async(prompt=prompt, max_tokens=1024)

        try:
            return self._parse_json(result["content"])
        except:
            return {"items": results[:top_k], "summary": "整理失败，返回原始结果"}

    # ================================================================
    # 记忆存储
    # ================================================================

    async def _store_search_memory(
        self,
        query: str,
        results: Any,
        task_id: str,
        from_memory: int = 0,
        from_external: int = 0
    ) -> str:
        """存储检索经验到记忆库"""
        items = results.get("items", []) if isinstance(results, dict) else []

        summary = f"检索: {query[:80]}"
        if items:
            summary += f" → 找到 {len(items)} 条结果"

        evidence = []
        for item in items[:3]:  # 只存前3条作为证据
            evidence.append(
                f"[{item.get('source', 'unknown')}] {item.get('title', '')}: {item.get('content', '')[:200]}"
            )

        return await self.store_memory(
            summary=summary,
            task_id=task_id,
            evidence=evidence,
            strategy=f"Retrieval Strategy: Memory Hits {from_memory}, External Retrieval {from_external}",
            tags=["search", "retrieval"],
            parent_ids=[]
        )

    # ================================================================
    # Prompt 构建
    # ================================================================

    def _build_verify_prompt(
        self,
        claim: str,
        evidence: str,
        similar_checks: List[MemoryUnit]
    ) -> str:
        """构建验证 prompt"""
        history = ""
        if similar_checks:
            history = "\n".join(
                f"- {m.summary[:100]}" for m in similar_checks[:3]
            )

        return f"""Please verify the truth of the following claim.

Claim: {claim}

Evidence: {evidence if evidence else "None"}

History:
{history if history else "None"}

Please return the verification result in a JSON format:
```json
{{
    "claim": "Original Claim",
    "verdict": "true/false/uncertain",
    "confidence": 0.9,
    "reasoning": "Reasoning Process",
    "supporting_evidence": ["Evidence 1", "Evidence 2"]
}}
```"""

    # ================================================================
    # 解析工具
    # ================================================================

    def _parse_json(self, text: str) -> Dict:
        """从 LLM 输出中提取 JSON"""
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return {"items": [], "summary": "Parsing Failed"}

    def _parse_verification(self, text: str) -> Dict:
        """解析验证结果"""
        result = self._parse_json(text)
        return {
            "claim": result.get("claim", ""),
            "verdict": result.get("verdict", "uncertain"),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", ""),
            "supporting_evidence": result.get("supporting_evidence", [])
        }