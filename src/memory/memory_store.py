import sqlite3
import chromadb
from typing import List, Tuple

class SharedMemoryModule:
    """共享记忆存储与检索"""
    
    def __init__(self, persist_dir: str = "./memory_store"):
        # 关系型数据库存储元数据
        self.sqlite_conn = sqlite3.connect(f"{persist_dir}/metadata.db")
        self._init_sqlite()
        
        # 向量数据库存储embedding
        self.vector_db = chromadb.PersistentClient(path=f"{persist_dir}/vectors")
        self.collection = self.vector_db.get_or_create_collection(
            name="agent_memories",
            metadata={"hnsw:space": "cosine"}
        )
        
    def _init_sqlite(self):
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                source_agent TEXT,
                task_id TEXT,
                task_theme TEXT,
                summary TEXT,
                conclusion TEXT,
                created_at REAL,
                access_count INTEGER,
                confidence REAL,
                version INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT,
                tag TEXT,
                FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
            )
        """)
        self.sqlite_conn.commit()
    
    async def store(self, memory: MemoryUnit):
        """存储记忆"""
        # 存储元数据
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.memory_id, memory.source_agent, memory.task_id,
            memory.task_theme, memory.summary, memory.conclusion,
            memory.created_at, memory.access_count, 
            memory.confidence, memory.version
        ))
        
        # 存储标签
        for tag in memory.tags:
            cursor.execute(
                "INSERT INTO memory_tags VALUES (?, ?)",
                (memory.memory_id, tag)
            )
        self.sqlite_conn.commit()
        
        # 存储向量
        if memory.embedding is not None:
            self.collection.add(
                ids=[memory.memory_id],
                embeddings=[memory.embedding.tolist()],
                metadatas=[{"task_theme": memory.task_theme}]
            )
    
    async def search_by_keyword(self, keyword: str, limit: int = 5) -> List[MemoryUnit]:
        """关键词检索"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            SELECT m.* FROM memories m
            WHERE m.summary LIKE ? OR m.task_theme LIKE ?
            ORDER BY m.access_count DESC
            LIMIT ?
        """, (f"%{keyword}%", f"%{keyword}%", limit))
        
        return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    async def search_by_similarity(self, query_embedding: np.ndarray, 
                                    limit: int = 5) -> List[Tuple[MemoryUnit, float]]:
        """语义相似度检索"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=limit
        )
        
        memories = []
        for i, memory_id in enumerate(results['ids'][0]):
            memory = await self.get_by_id(memory_id)
            score = 1 - results['distances'][0][i]  # 转换为相似度
            memories.append((memory, score))
        
        return memories
    
    async def hybrid_search(self, query: str, embedding: np.ndarray, 
                           limit: int = 5) -> List[MemoryUnit]:
        """混合检索：结合关键词和语义"""
        keyword_results = await self.search_by_keyword(query, limit * 2)
        semantic_results = await self.search_by_similarity(embedding, limit * 2)
        
        # 合并去重
        seen_ids = set()
        merged = []
        
        for mem in keyword_results:
            if mem.memory_id not in seen_ids:
                merged.append(mem)
                seen_ids.add(mem.memory_id)
                
        for mem, score in semantic_results:
            if mem.memory_id not in seen_ids and score > 0.7:
                merged.append(mem)
                seen_ids.add(mem.memory_id)
                
        return merged[:limit]
    
    async def update_access(self, memory_id: str):
        """更新访问计数"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            UPDATE memories SET access_count = access_count + 1
            WHERE memory_id = ?
        """, (memory_id,))
        self.sqlite_conn.commit()