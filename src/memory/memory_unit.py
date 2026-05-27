from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import List, Optional, Dict, Any
import uuid

import numpy as np

@dataclass
class MemoryUnit:
    """共享记忆单元"""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str = ""
    task_id: str = ""
    task_theme: str = ""
    
    # 内容
    summary: str = ""
    evidence: List[str] = field(default_factory=list)
    strategy: str = ""
    conclusion: str = ""

    # ========== 关联关系（证据链） ==========
    parent_ids: List[str] = field(default_factory=list)   # 前置记忆 ID
    child_ids: List[str] = field(default_factory=list)    # 后继记忆 ID
    
    # 向量表示
    embedding: Optional[np.ndarray] = None
    vector: Optional[np.ndarray] = None
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    # 关联
    related_memories: List[str] = field(default_factory=list)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于 JSON 序列化）"""
        return {
            "memory_id": self.memory_id,
            "source_agent": self.source_agent,
            "task_id": self.task_id,
            "task_theme": self.task_theme,
            "summary": self.summary,
            "evidence": self.evidence,
            "strategy": self.strategy,
            "conclusion": self.conclusion,
            "vector": self.vector.tolist() if self.vector is not None else None,
            "parent_ids": self.parent_ids,
            "tags": self.tags,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "confidence": self.confidence,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryUnit':
        """从字典恢复"""
        vector = data.get("vector")
        if vector is not None:
            vector = np.array(vector)
        return cls(
            memory_id=data.get("memory_id", ""),
            source_agent=data.get("source_agent", ""),
            task_id=data.get("task_id", ""),
            task_theme=data.get("task_theme", ""),
            summary=data.get("summary", ""),
            evidence=data.get("evidence", []),
            strategy=data.get("strategy", ""),
            conclusion=data.get("conclusion", ""),
            vector=vector,
            parent_ids=data.get("parent_ids", []),
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            access_count=data.get("access_count", 0),
            confidence=data.get("confidence", 1.0),
            version=data.get("version", 1)
        )

    # ================================================================
    # 工具方法
    # ================================================================

    def increment_access(self):
        """访问计数 +1"""
        self.access_count += 1

    def update_confidence(self, new_confidence: float):
        """更新置信度（限制在 0~1）"""
        self.confidence = max(0.0, min(1.0, new_confidence))

    def decay_confidence(self, decay_rate: float = 0.01):
        """置信度衰减（随时间推移）"""
        self.confidence = max(0.1, self.confidence - decay_rate)

    def bump_version(self):
        """版本号 +1"""
        self.version += 1

    def add_parent(self, parent_id: str):
        """添加前置记忆"""
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)

    def add_child(self, child_id: str):
        """添加后继记忆"""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)

    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_evidence(self, evidence_text: str):
        """添加证据"""
        if evidence_text not in self.evidence:
            self.evidence.append(evidence_text)

    # ================================================================
    # 摘要
    # ================================================================

    def get_brief(self) -> str:
        """生成简短摘要（用于日志和调试）"""
        return (
            f"Memory({self.memory_id[:8]}..) "
            f"agent={self.source_agent} "
            f"theme={self.task_theme[:30]} "
            f"summary={self.summary[:50]} "
            f"tags={self.tags} "
            f"conf={self.confidence:.2f}"
        )

    def get_full_summary(self) -> str:
        """生成完整摘要"""
        parts = [
            f"记忆 ID: {self.memory_id}",
            f"来源: {self.source_agent}",
            f"任务: {self.task_theme}",
            f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))}",
            f"摘要: {self.summary}",
        ]
        if self.strategy:
            parts.append(f"策略: {self.strategy}")
        if self.conclusion:
            parts.append(f"结论: {self.conclusion}")
        if self.evidence:
            parts.append(f"证据数: {len(self.evidence)}")
        if self.tags:
            parts.append(f"标签: {', '.join(self.tags)}")
        parts.append(f"置信度: {self.confidence:.2f}")
        parts.append(f"访问次数: {self.access_count}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"MemoryUnit({self.memory_id[:8]}.., {self.task_theme[:20]}, conf={self.confidence:.2f})"