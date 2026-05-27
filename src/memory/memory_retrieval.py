"""
memory/memory_retrieval.py
混合检索器

提供统一的记忆检索接口，组合三种检索方式：
1. 关键词检索（SQLite LIKE）
2. 标签检索（SQLite IN）
3. 语义向量检索（ChromaDB）

检索策略：
1. 优先语义向量粗筛（Top-2k）
2. 关键词/标签过滤
3. 按相似度排序返回 Top-K
"""

from typing import List, Dict, Optional, Any
import numpy as np

from src.memory.memory_store import MemoryStore
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_graph import MemoryGraph


class HybridRetrieval:
    """
    混合检索器

    所有 Agent 通过此类查询和存储记忆。
    内部调用 MemoryStore 完成实际的数据库操作。
    """

    def __init__(
        self,
        store: MemoryStore,
        memory_graph: MemoryGraph=None
    ):
        """
        参数:
            store: MemoryStore 实例
            memory_graph: MemoryGraph 实例（可选，用于证据链查询）
        """
        self._store = store
        self.memory_graph = memory_graph

        # 统计
        self.stats = {
            "total_queries": 0,
            "total_hits": 0,
            "total_stores": 0
        }

    # ================================================================
    # 检索
    # ================================================================

    async def search(
        self,
        query_text: str,
        query_vector: Any = None,
        top_k: int = 5,
        filter_tags: List[str] = None,
        min_confidence: float = 0.0,
        strategy: str = "hybrid"
    ) -> List[MemoryUnit]:
        """
        混合检索

        参数:
            query_text: 查询文本
            query_vector: 查询向量（可选，如果不传则只用关键词检索）
            top_k: 返回数量
            filter_tags: 标签过滤
            min_confidence: 最低置信度
            strategy: 检索策略
                - "hybrid": 语义 + 关键词（默认）
                - "semantic": 仅语义向量
                - "keyword": 仅关键词
                - "tag": 仅标签

        返回:
            MemoryUnit 列表
        """
        self.stats["total_queries"] += 1

        results = []

        if strategy == "hybrid":
            # 混合策略：语义粗筛 + 关键词过滤
            results = await self._search_hybrid(
                query_text, query_vector, top_k, filter_tags
            )
        elif strategy == "semantic" and query_vector is not None:
            # 纯语义检索
            results = await self._search_semantic(
                query_vector, top_k, filter_tags
            )
        elif strategy == "keyword":
            # 纯关键词检索
            results = await self._search_keyword(query_text, top_k)
        elif strategy == "tag" and filter_tags:
            # 纯标签检索
            results = await self._search_by_tags(filter_tags, top_k)
        else:
            # 兜底：关键词检索
            results = await self._search_keyword(query_text, top_k)

        # 过滤低置信度
        results = [r for r in results if r.confidence >= min_confidence]

        # 更新访问计数
        for mem in results:
            self._store.update_access(mem.memory_id)

        if results:
            self.stats["total_hits"] += len(results)

        return results

    async def search_by_vector(
        self,
        query_vector,
        top_k: int = 5,
        filter_tags: List[str] = None
    ) -> List[MemoryUnit]:
        """
        纯向量检索（Agent 直接调用）
        """
        return await self._search_semantic(query_vector, top_k, filter_tags)

    # ================================================================
    # 存储
    # ================================================================

    async def store(self, memory: MemoryUnit) -> str:
        """
        存储一条记忆

        参数:
            memory: MemoryUnit 实例

        返回:
            memory_id
        """
        self.stats["total_stores"] += 1

        # 存入 MemoryStore
        memory_id = self._store.store(memory)

        return memory_id

    # ================================================================
    # 内部检索实现
    # ================================================================

    async def _search_hybrid(
        self,
        query_text: str,
        query_vector: Any,
        top_k: int,
        filter_tags: List[str] = None
    ) -> List[MemoryUnit]:
        """
        混合检索：语义粗筛 + 关键词精排

        1. 如果有向量，用向量检索 Top-2k
        2. 用关键词在结果中过滤
        3. 按相似度排序
        """
        # 1. 向量粗筛
        if query_vector is not None:
            vector_results = self._store.search_by_vector(
                query_vector=query_vector,
                top_k=top_k * 4,  # 多取一些
                filter_tags=filter_tags
            )
            candidate_ids = [r["memory_id"] for r in vector_results]
            candidate_scores = {r["memory_id"]: r["score"] for r in vector_results}
        else:
            candidate_ids = None
            candidate_scores = {}

        # 2. 关键词检索
        keyword_results = self._store.search_by_keyword(
            keyword=query_text,
            limit=top_k * 4
        )

        # 3. 合并排序
        merged = {}
        for mem_dict in keyword_results:
            mem_id = mem_dict["memory_id"]
            score = candidate_scores.get(mem_id, 0.5) + 0.5  # 关键词匹配加分
            merged[mem_id] = (score, mem_dict)

        if candidate_ids:
            for mem_id in candidate_ids:
                if mem_id not in merged:
                    mem_dict = self._store.get_by_id(mem_id)
                    if mem_dict:
                        merged[mem_id] = (candidate_scores.get(mem_id, 0.5), mem_dict)

        # 4. 排序
        sorted_items = sorted(merged.values(), key=lambda x: -x[0])
        top_items = sorted_items[:top_k]

        return [self._dict_to_memory(item[1]) for item in top_items]

    async def _search_semantic(
        self,
        query_vector,
        top_k: int,
        filter_tags: List[str] = None
    ) -> List[MemoryUnit]:
        """纯语义向量检索"""
        results = self._store.search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            filter_tags=filter_tags
        )

        memories = []
        for r in results:
            mem_dict = self._store.get_by_id(r["memory_id"])
            if mem_dict:
                memory = self._dict_to_memory(mem_dict)
                memory.confidence = r.get("score", memory.confidence)
                memories.append(memory)

        return memories

    async def _search_keyword(
        self,
        query_text: str,
        top_k: int
    ) -> List[MemoryUnit]:
        """关键词检索"""
        results = self._store.search_by_keyword(
            keyword=query_text,
            limit=top_k
        )
        return [self._dict_to_memory(r) for r in results]

    async def _search_by_tags(
        self,
        tags: List[str],
        top_k: int
    ) -> List[MemoryUnit]:
        """标签检索"""
        results = self._store.search_by_tags(
            tags=tags,
            limit=top_k
        )
        return [self._dict_to_memory(r) for r in results]

    # ================================================================
    # 批量操作
    # ================================================================

    async def get_by_task(self, task_id: str) -> List[MemoryUnit]:
        """获取某个任务的所有记忆"""
        results = self._store.get_by_task(task_id)
        return [self._dict_to_memory(r) for r in results]

    async def get_by_agent(self, agent_id: str, limit: int = 100) -> List[MemoryUnit]:
        """获取某个 Agent 的所有记忆"""
        results = self._store.get_by_agent(agent_id, limit)
        return [self._dict_to_memory(r) for r in results]

    async def get_by_id(self, memory_id: str) -> Optional[MemoryUnit]:
        """根据 ID 获取记忆"""
        result = self._store.get_by_id(memory_id)
        if result:
            return self._dict_to_memory(result)
        return None

    # ================================================================
    # 记忆维护
    # ================================================================

    async def update_confidence(self, memory_id: str, confidence: float):
        """更新置信度"""
        self._store.update_confidence(memory_id, confidence)

    async def delete(self, memory_id: str):
        """删除记忆"""
        self._store.delete(memory_id)
        # 同时从关联图中移除
        if self.memory_graph:
            self.memory_graph.remove_node(memory_id)

    # ================================================================
    # 统计
    # ================================================================

    def get_stats(self) -> Dict:
        """获取统计信息"""
        store_stats = self._store.get_stats()
        return {
            **self.stats,
            "store_stats": store_stats,
            "hit_rate": (
                self.stats["total_hits"] / max(1, self.stats["total_queries"])
            )
        }

    # ================================================================
    # 工具
    # ================================================================

    def _dict_to_memory(self, d: Dict) -> MemoryUnit:
        """将存储字典转为 MemoryUnit"""
        return MemoryUnit(
            memory_id=d.get("memory_id", ""),
            source_agent=d.get("source_agent", ""),
            task_id=d.get("task_id", ""),
            task_theme=d.get("task_theme", ""),
            summary=d.get("summary", ""),
            evidence=d.get("evidence", []),
            strategy=d.get("strategy", ""),
            conclusion=d.get("conclusion", ""),
            vector=None,  # 从存储读取时不加载向量，节省内存
            parent_ids=d.get("parent_ids", []),
            tags=d.get("tags", []),
            created_at=d.get("created_at", 0),
            access_count=d.get("access_count", 0),
            confidence=d.get("confidence", 1.0),
            version=d.get("version", 1)
        )