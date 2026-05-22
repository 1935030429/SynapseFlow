from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import List, Optional
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
    
    # 向量表示
    embedding: Optional[np.ndarray] = None
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    # 关联
    related_memories: List[str] = field(default_factory=list)
    version: int = 1
    
    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        if d['embedding'] is not None:
            d['embedding'] = d['embedding'].tolist()
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryUnit':
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)