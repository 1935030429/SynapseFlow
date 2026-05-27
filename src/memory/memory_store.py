import sqlite3


"""
memory/memory_store.py
记忆持久化存储层

双层存储：
- SQLite:  记忆元数据（ID、来源、时间、标签、文本摘要）
- ChromaDB: 语义向量（用于语义相似度检索）
"""

import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.memory.memory_unit import MemoryUnit

import chromadb
from chromadb.config import Settings


class MemoryStore:
    """
    记忆持久化存储

    双层存储架构：
    1. SQLite: 结构化元数据 + 文本内容
    2. ChromaDB: 语义向量索引
    """

    def __init__(self, persist_dir: str = "Project/SynapseFlow/memory_store"):
        """
        参数:
            persist_dir: 持久化目录
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # ──── SQLite ────
        self.sqlite_path = os.path.join(persist_dir, "metadata.db")
        self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_sqlite()

        # ──── ChromaDB ────
        self.chroma_path = os.path.join(persist_dir, "vectors")
        self._chroma_client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name="agent_memories",
            metadata={"hnsw:space": "cosine"}
        )

    # ================================================================
    # SQLite 初始化
    # ================================================================

    def _init_sqlite(self):
        """创建 SQLite 表"""
        cursor = self._conn.cursor()

        # 记忆元数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                source_agent TEXT NOT NULL,
                task_id TEXT,
                task_theme TEXT,
                summary TEXT,
                strategy TEXT,
                conclusion TEXT,
                evidence_json TEXT,
                tags_json TEXT,
                parent_ids_json TEXT,
                created_at REAL,
                access_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                version INTEGER DEFAULT 1
            )
        """)

        # 索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_agent
            ON memories(source_agent)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_id
            ON memories(task_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON memories(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_count
            ON memories(access_count)
        """)

        self._conn.commit()

    # ================================================================
    # 存储
    # ================================================================

    def store(self, memory_unit: MemoryUnit) -> str:
        """
        存储一条记忆

        参数:
            memory_unit: MemoryUnit 实例

        返回:
            memory_id
        """
        memory_id = memory_unit.memory_id

        # 1. 写入 SQLite
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (memory_id, source_agent, task_id, task_theme,
             summary, strategy, conclusion,
             evidence_json, tags_json, parent_ids_json,
             created_at, access_count, confidence, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            memory_unit.source_agent,
            memory_unit.task_id,
            memory_unit.task_theme,
            memory_unit.summary,
            memory_unit.strategy,
            memory_unit.conclusion,
            json.dumps(memory_unit.evidence, ensure_ascii=False),
            json.dumps(memory_unit.tags, ensure_ascii=False),
            json.dumps(memory_unit.parent_ids, ensure_ascii=False),
            memory_unit.created_at,
            memory_unit.access_count,
            memory_unit.confidence,
            memory_unit.version
        ))
        self._conn.commit()

        # 2. 写入 ChromaDB（如果有向量）
        if memory_unit.vector is not None:
            # 确保向量是 float 列表
            if isinstance(memory_unit.vector, np.ndarray):
                vector_list = memory_unit.vector.astype(float).tolist()
            else:
                vector_list = list(memory_unit.vector)

            # 检查是否已存在
            existing = self._collection.get(ids=[memory_id])
            if existing and existing["ids"]:
                self._collection.update(
                    ids=[memory_id],
                    embeddings=[vector_list],
                    metadatas=[{
                        "task_theme": memory_unit.task_theme or "",
                        "source_agent": memory_unit.source_agent,
                        "tags": ",".join(memory_unit.tags)
                    }]
                )
            else:
                self._collection.add(
                    ids=[memory_id],
                    embeddings=[vector_list],
                    metadatas=[{
                        "task_theme": memory_unit.task_theme or "",
                        "source_agent": memory_unit.source_agent,
                        "tags": ",".join(memory_unit.tags)
                    }]
                )

        return memory_id

    # ================================================================
    # 查询
    # ================================================================

    def get_by_id(self, memory_id: str) -> Optional[Dict]:
        """根据 ID 获取记忆"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)
        return None

    def get_by_task(self, task_id: str) -> List[Dict]:
        """获取某个任务的所有记忆"""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM memories WHERE task_id = ? ORDER BY created_at",
            (task_id,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_by_agent(self, agent_id: str, limit: int = 100) -> List[Dict]:
        """获取某个 Agent 创建的所有记忆"""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM memories WHERE source_agent = ? ORDER BY created_at DESC LIMIT ?",
            (agent_id, limit)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ================================================================
    # 关键词检索
    # ================================================================

    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict]:
        """关键词检索（SQLite LIKE）"""
        cursor = self._conn.cursor()
        pattern = f"%{keyword}%"
        cursor.execute("""
            SELECT * FROM memories
            WHERE summary LIKE ?
               OR task_theme LIKE ?
               OR conclusion LIKE ?
               OR strategy LIKE ?
            ORDER BY access_count DESC, created_at DESC
            LIMIT ?
        """, (pattern, pattern, pattern, pattern, limit))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict]:
        """标签检索"""
        cursor = self._conn.cursor()
        pattern = f'%"{tag}"%'
        cursor.execute("""
            SELECT * FROM memories
            WHERE tags_json LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (pattern, limit))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[Dict]:
        """多标签检索（AND）"""
        if not tags:
            return []

        cursor = self._conn.cursor()
        conditions = " AND ".join(["tags_json LIKE ?" for _ in tags])
        patterns = [f'%"{t}"%' for t in tags]

        cursor.execute(f"""
            SELECT * FROM memories
            WHERE {conditions}
            ORDER BY created_at DESC
            LIMIT ?
        """, (*patterns, limit))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ================================================================
    # 向量检索
    # ================================================================

    def search_by_vector(
        self,
        query_vector,
        top_k: int = 10,
        filter_tags: List[str] = None
    ) -> List[Dict]:
        """
        语义向量检索（ChromaDB）

        参数:
            query_vector: 查询向量 (numpy array 或 list)
            top_k: 返回数量
            filter_tags: 标签过滤（可选）

        返回:
            [{"memory_id": ..., "score": ..., "metadata": ...}, ...]
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.astype(float).tolist()

        # 构建过滤条件
        where_filter = None
        if filter_tags:
            where_filter = {
                "tags": {"$contains": ",".join(filter_tags)}
            }

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances"]
        )

        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                score = 1.0 - results["distances"][0][i]  # distance → similarity
                formatted.append({
                    "memory_id": mem_id,
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })

        return formatted

    # ================================================================
    # 更新
    # ================================================================

    def update_access(self, memory_id: str):
        """更新访问计数"""
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE memories
            SET access_count = access_count + 1
            WHERE memory_id = ?
        """, (memory_id,))
        self._conn.commit()

    def update_confidence(self, memory_id: str, confidence: float):
        """更新置信度"""
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE memories
            SET confidence = ?
            WHERE memory_id = ?
        """, (max(0.0, min(1.0, confidence)), memory_id))
        self._conn.commit()

    def update_summary(self, memory_id: str, summary: str, version: int = None):
        """更新摘要（创建新版本）"""
        cursor = self._conn.cursor()
        if version is not None:
            cursor.execute("""
                UPDATE memories
                SET summary = ?, version = ?
                WHERE memory_id = ?
            """, (summary, version, memory_id))
        else:
            cursor.execute("""
                UPDATE memories
                SET summary = ?, version = version + 1
                WHERE memory_id = ?
            """, (summary, memory_id))
        self._conn.commit()

    # ================================================================
    # 删除
    # ================================================================

    def delete(self, memory_id: str):
        """删除记忆（SQLite + ChromaDB）"""
        # SQLite
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        self._conn.commit()

        # ChromaDB
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass

    # ================================================================
    # 统计
    # ================================================================

    def count(self) -> int:
        """总记忆数"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        cursor = self._conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT source_agent, COUNT(*) as cnt "
            "FROM memories GROUP BY source_agent"
        )
        by_agent = {row["source_agent"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(DISTINCT task_id) FROM memories")
        unique_tasks = cursor.fetchone()[0]

        return {
            "total_memories": total,
            "by_agent": by_agent,
            "unique_tasks": unique_tasks,
            "chroma_collection": self._collection.count()
        }

    # ================================================================
    # 工具
    # ================================================================

    def _row_to_dict(self, row) -> Dict:
        """SQLite Row → 字典"""
        d = dict(row)
        # 反序列化 JSON 字段
        for field in ["evidence_json", "tags_json", "parent_ids_json"]:
            if field in d and d[field]:
                try:
                    d[field.replace("_json", "")] = json.loads(d[field])
                except json.JSONDecodeError:
                    d[field.replace("_json", "")] = []
            else:
                d[field.replace("_json", "")] = []
            d.pop(field, None)
        return d

    def close(self):
        """关闭连接"""
        self._conn.close()

    def clear(self):
        """清空所有记忆（慎用）"""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories")
        self._conn.commit()

        # 清空 ChromaDB
        try:
            self._chroma_client.delete_collection("agent_memories")
            self._collection = self._chroma_client.get_or_create_collection(
                name="agent_memories",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass