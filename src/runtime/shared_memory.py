# class SharedMemoryManager:
#     """
#     三块共享内存区域：
#     - task_board:    任务黑板（结构化任务数据）
#     - state_vectors: 状态向量区（SDE / Embedding）
#     - memory_index:  记忆索引区（高频向量常驻）
#     """

#     def __init__(self, size_mb=256):
#         # shm_open 创建或打开 /dev/shm/ma_system
#         # mmap 映射到进程地址空间
#         # 初始化三段内存布局：
#         #   [0, 64MB)         → task_board
#         #   [64MB, 192MB)     → state_vectors
#         #   [192MB, 256MB)    → memory_index
#         pass

#     def write_task(self, task_id, data_bytes):
#         # 在 task_board 区分配空间
#         # 写入 data_bytes
#         # 返回偏移量 offset（8字节整数）
#         pass

#     def read_task(self, offset):
#         # 直接从 task_board + offset 读取
#         # 零拷贝，内存布局已在协议中约定
#         pass

#     def write_state_vector(self, data_bytes):
#         # 写入 state_vectors 区
#         # 返回 offset
#         pass

#     def read_state_vector(self, offset):
#         # 直接从 state_vectors + offset 读取
#         pass

#     def write_memory_index(self, vector_id, vector):
#         # 将高频访问的记忆向量写入 memory_index 区
#         pass

"""
runtime/shared_memory.py
共享内存管理器

使用 mmap 创建共享内存区域，所有 Agent 进程/线程可以直接读写。
避免数据序列化/反序列化的开销，实现零拷贝传递。

三块区域：
- task_board:     任务黑板（结构化任务数据、计划、结果）
- state_vectors:  状态向量区（SDE / KV Cache / Embedding）
- memory_index:   记忆索引区（高频访问的记忆向量常驻）
"""

import mmap
import os
import struct
import time
import threading
from typing import Dict, Any, Optional


class SharedMemoryManager:
    """
    共享内存管理器
    
    内存布局（默认 256MB）：
    ┌──────────────────────────────────────────┐
    │ Header (64B)                              │
    │   - magic: 4B                             │
    │   - version: 4B                           │
    │   - total_size: 8B                        │
    │   - task_board_offset: 8B                 │
    │   - task_board_size: 8B                   │
    │   - state_vector_offset: 8B               │
    │   - state_vector_size: 8B                 │
    │   - memory_index_offset: 8B               │
    │   - memory_index_size: 8B                 │
    ├──────────────────────────────────────────┤
    │ Task Board (64MB)                         │
    │   任务黑板：结构化任务数据                 │
    │   ┌─ Entry Header (16B) ─┐               │
    │   │ - offset: 8B          │               │
    │   │ - size: 4B            │               │
    │   │ - flags: 4B           │               │
    │   ├───────────────────────┤               │
    │   │ Data (variable)       │               │
    │   └───────────────────────┘               │
    │   ...                                     │
    ├──────────────────────────────────────────┤
    │ State Vectors (128MB)                     │
    │   状态向量区：SDE / KV Cache / Embedding   │
    │   同样采用 Entry Header + Data 结构       │
    ├──────────────────────────────────────────┤
    │ Memory Index (64MB)                       │
    │   记忆索引区：高频记忆向量                 │
    └──────────────────────────────────────────┘
    """
    
    # 常量
    MAGIC = 0x4D415348  # "MASH" = Multi-Agent Shared Memory
    VERSION = 1
    HEADER_SIZE = 64
    
    DEFAULT_SIZES = {
        "task_board": 64 * 1024 * 1024,      # 64MB
        "state_vectors": 128 * 1024 * 1024,  # 128MB
        "memory_index": 64 * 1024 * 1024     # 64MB
    }
    
    def __init__(self, name: str = "ma_system", size_mb: int = 256):
        """
        参数:
            name: 共享内存名称（用于 /dev/shm/ 下的文件）
            size_mb: 总大小（MB）
        """
        self.name = name
        total_size = size_mb * 1024 * 1024
        
        # 分配各区域大小
        self.region_sizes = {
            "task_board": self.DEFAULT_SIZES["task_board"],
            "state_vectors": self.DEFAULT_SIZES["state_vectors"],
            "memory_index": min(
                self.DEFAULT_SIZES["memory_index"],
                total_size - self.HEADER_SIZE 
                - self.DEFAULT_SIZES["task_board"] 
                - self.DEFAULT_SIZES["state_vectors"]
            )
        }
        
        # 创建或打开共享内存
        self._init_shm(total_size)
        
        # 各区域的空闲空间管理器
        self._allocators = {}
        for region in ["task_board", "state_vectors", "memory_index"]:
            offset = self._read_header(f"{region}_offset")
            size = self.region_sizes[region]
            self._allocators[region] = _RegionAllocator(self._mm, offset, size)
        
        # 并发锁
        self._lock = threading.Lock()
        
        # 统计
        self.stats = {
            "total_writes": 0,
            "total_reads": 0,
            "total_bytes_written": 0,
            "total_bytes_read": 0,
            "allocation_failures": 0
        }
    
    # ================================================================
    # 初始化
    # ================================================================
    
    def _init_shm(self, total_size: int):
        """初始化共享内存"""
        shm_path = f"/dev/shm/{self.name}"
        
        # 检查是否已存在
        if os.path.exists(shm_path):
            # 打开已有
            self._fd = os.open(shm_path, os.O_RDWR)
            self._mm = mmap.mmap(self._fd, total_size, access=mmap.ACCESS_WRITE)
            
            # 验证 magic
            magic = struct.unpack(">I", self._mm[0:4])[0]
            if magic != self.MAGIC:
                raise RuntimeError(f"共享内存 magic 不匹配: 期望 {self.MAGIC}, 实际 {magic}")
        else:
            # 创建新的
            self._fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(self._fd, total_size)
            self._mm = mmap.mmap(self._fd, total_size, access=mmap.ACCESS_WRITE)
            
            # 写入头部
            self._write_initial_header(total_size)
    
    def _write_initial_header(self, total_size: int):
        """写入初始头部"""
        # 计算各区域偏移
        task_board_offset = self.HEADER_SIZE
        state_vector_offset = task_board_offset + self.region_sizes["task_board"]
        memory_index_offset = state_vector_offset + self.region_sizes["state_vectors"]
        
        header = struct.pack(
            ">IIQQQQQQQ",
            self.MAGIC,
            self.VERSION,
            total_size,
            task_board_offset,
            self.region_sizes["task_board"],
            state_vector_offset,
            self.region_sizes["state_vectors"],
            memory_index_offset,
            self.region_sizes["memory_index"]
        )
        self._mm[0:len(header)] = header
        self._mm.flush()
    
    def _read_header(self, field: str) -> int:
        """读取头部字段"""
        fields = {
            "magic": (0, "I"),
            "version": (4, "I"),
            "total_size": (8, "Q"),
            "task_board_offset": (16, "Q"),
            "task_board_size": (24, "Q"),
            "state_vector_offset": (32, "Q"),
            "state_vector_size": (40, "Q"),
            "memory_index_offset": (48, "Q"),
            "memory_index_size": (56, "Q")
        }
        offset, fmt = fields[field]
        return struct.unpack(f">{fmt}", self._mm[offset:offset + struct.calcsize(fmt)])[0]
    
    # ================================================================
    # 写入
    # ================================================================
    
    def write_task(self, task_id: str, data: bytes) -> int:
        """
        写入任务数据到 task_board 区域
        
        参数:
            task_id: 任务 ID（用于日志和调试）
            data: 要写入的数据
        
        返回:
            偏移量（后续读取时使用）
        """
        return self._write("task_board", data)
    
    def write_state_vector(self, data: bytes) -> int:
        """
        写入状态向量到 state_vectors 区域
        
        返回:
            偏移量
        """
        return self._write("state_vectors", data)
    
    def write_memory_index(self, data: bytes) -> int:
        """写入记忆索引"""
        return self._write("memory_index", data)
    
    def _write(self, region: str, data: bytes) -> int:
        """写入指定区域"""
        with self._lock:
            allocator = self._allocators[region]
            offset = allocator.allocate(len(data))
            
            if offset < 0:
                self.stats["allocation_failures"] += 1
                # 尝试整理碎片后重试
                allocator.compact()
                offset = allocator.allocate(len(data))
                if offset < 0:
                    raise MemoryError(f"{region} 区域空间不足，需要 {len(data)} 字节")
            
            # 写入 Entry Header + Data
            region_start = allocator.region_start
            entry_header = struct.pack(">II", len(data), 0)  # size, flags
            self._mm[region_start + offset : region_start + offset + 8] = entry_header
            self._mm[region_start + offset + 8 : region_start + offset + 8 + len(data)] = data
            
            self.stats["total_writes"] += 1
            self.stats["total_bytes_written"] += len(data)
            
            return region_start + offset
    
    # ================================================================
    # 读取
    # ================================================================
    
    def read_task(self, offset: int) -> bytes:
        """从 task_board 读取"""
        return self._read(offset)
    
    def read_state_vector(self, offset: int) -> bytes:
        """从 state_vectors 读取"""
        return self._read(offset)
    
    def read_memory_index(self, offset: int) -> bytes:
        """从 memory_index 读取"""
        return self._read(offset)
    
    def _read(self, offset: int) -> bytes:
        """从指定偏移量读取"""
        # 读取 Entry Header
        header = self._mm[offset:offset + 8]
        size, flags = struct.unpack(">II", header)
        
        # 读取数据
        data = self._mm[offset + 8 : offset + 8 + size]
        
        self.stats["total_reads"] += 1
        self.stats["total_bytes_read"] += size
        
        return bytes(data)
    
    # ================================================================
    # 释放
    # ================================================================
    
    def free_task(self, offset: int):
        """释放 task_board 中的空间"""
        region_start = self._allocators["task_board"].region_start
        self._allocators["task_board"].free(offset - region_start)
    
    def free_state_vector(self, offset: int):
        """释放 state_vectors 中的空间"""
        region_start = self._allocators["state_vectors"].region_start
        self._allocators["state_vectors"].free(offset - region_start)
    
    # ================================================================
    # 统计与调试
    # ================================================================
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = dict(self.stats)
        for name, allocator in self._allocators.items():
            stats[f"{name}_used"] = allocator.used_size
            stats[f"{name}_free"] = allocator.free_size
            stats[f"{name}_fragments"] = allocator.fragment_count
        return stats
    
    def get_region_info(self) -> Dict:
        """获取各区域信息"""
        return {
            name: {
                "total": self.region_sizes[name],
                "used": allocator.used_size,
                "free": allocator.free_size,
                "writes": allocator.write_count
            }
            for name, allocator in self._allocators.items()
        }
    
    # ================================================================
    # 清理
    # ================================================================
    
    def close(self):
        """关闭共享内存"""
        self._mm.flush()
        self._mm.close()
        os.close(self._fd)
    
    def destroy(self):
        """销毁共享内存（删除文件）"""
        self.close()
        shm_path = f"/dev/shm/{self.name}"
        if os.path.exists(shm_path):
            os.unlink(shm_path)


# ================================================================
# 区域内的简易分配器（First-Fit）
# ================================================================

class _RegionAllocator:
    """
    区域内的简易内存分配器
    
    使用 First-Fit 策略管理空闲块。
    每个块的头部：size(4B) + flags(4B)，最高位标记是否空闲
    """
    
    ENTRY_HEADER_SIZE = 8
    FLAG_FREE = 0x80000000
    
    def __init__(self, mm: mmap.mmap, region_start: int, region_size: int):
        self.mm = mm
        self.region_start = region_start
        self.region_size = region_size
        self.write_count = 0
        
        # 初始化：整个区域是一个大空闲块
        self._write_entry(0, region_size - self.ENTRY_HEADER_SIZE, self.FLAG_FREE)
    
    @property
    def used_size(self) -> int:
        """已用大小"""
        return self._calculate_used()
    
    @property
    def free_size(self) -> int:
        """空闲大小"""
        return self.region_size - self.used_size - self.ENTRY_HEADER_SIZE
    
    @property
    def fragment_count(self) -> int:
        """碎片数量（空闲块数量）"""
        return self._count_free_blocks()
    
    def allocate(self, size: int) -> int:
        """
        分配空间（First-Fit）
        
        返回: 块内偏移量（相对于 region_start），-1 表示失败
        """
        pos = 0
        while pos < self.region_size - self.ENTRY_HEADER_SIZE:
            block_size, flags = self._read_entry(pos)
            
            if flags & self.FLAG_FREE and block_size >= size:
                # 找到足够大的空闲块
                remaining = block_size - size
                
                if remaining >= self.ENTRY_HEADER_SIZE + 16:
                    # 分割：前部分分配，后部分保持空闲
                    self._write_entry(pos, size, 0)  # 分配
                    self._write_entry(
                        pos + self.ENTRY_HEADER_SIZE + size,
                        remaining - self.ENTRY_HEADER_SIZE,
                        self.FLAG_FREE
                    )
                else:
                    # 整块分配
                    self._write_entry(pos, block_size, 0)
                
                self.write_count += 1
                return pos
            
            # 跳到下一块
            pos += self.ENTRY_HEADER_SIZE + block_size
        
        return -1
    
    def free(self, offset: int):
        """释放空间"""
        _, flags = self._read_entry(offset)
        if not (flags & self.FLAG_FREE):
            # 标记为空闲
            size, _ = self._read_entry(offset)
            self._write_entry(offset, size, flags | self.FLAG_FREE)
            self.write_count -= 1
    
    def compact(self):
        """整理碎片：合并相邻空闲块"""
        pos = 0
        prev_free_offset = -1
        
        while pos < self.region_size - self.ENTRY_HEADER_SIZE:
            size, flags = self._read_entry(pos)
            
            if flags & self.FLAG_FREE:
                if prev_free_offset >= 0:
                    # 合并到前一个空闲块
                    prev_size, _ = self._read_entry(prev_free_offset)
                    self._write_entry(prev_free_offset, prev_size + self.ENTRY_HEADER_SIZE + size, self.FLAG_FREE)
                else:
                    prev_free_offset = pos
            else:
                prev_free_offset = -1
            
            pos += self.ENTRY_HEADER_SIZE + size
    
    def _read_entry(self, offset: int) -> tuple:
        """读取块头部 (size, flags)"""
        abs_offset = self.region_start + offset
        return struct.unpack(">II", self.mm[abs_offset:abs_offset + 8])
    
    def _write_entry(self, offset: int, size: int, flags: int):
        """写入块头部"""
        abs_offset = self.region_start + offset
        self.mm[abs_offset:abs_offset + 8] = struct.pack(">II", size, flags)
    
    def _calculate_used(self) -> int:
        """计算已用空间"""
        used = 0
        pos = 0
        while pos < self.region_size - self.ENTRY_HEADER_SIZE:
            size, flags = self._read_entry(pos)
            if not (flags & self.FLAG_FREE):
                used += size + self.ENTRY_HEADER_SIZE
            pos += self.ENTRY_HEADER_SIZE + size
        return used
    
    def _count_free_blocks(self) -> int:
        """计算空闲块数量"""
        count = 0
        pos = 0
        while pos < self.region_size - self.ENTRY_HEADER_SIZE:
            _, flags = self._read_entry(pos)
            if flags & self.FLAG_FREE:
                count += 1
            pos += self.ENTRY_HEADER_SIZE + struct.unpack(">I", self.mm[
                self.region_start + pos : self.region_start + pos + 4
            ])[0]
        return count