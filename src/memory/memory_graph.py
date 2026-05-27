from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from src.memory.memory_unit import MemoryUnit

class MemoryGraph:
    def __init__(self, memory_store):
        """
        参数:
            memory_store: MemoryStore 实例（可选，用于验证记忆是否存在）
        """
        self._children: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # 逆邻接表：child_id → [(parent_id, relation_type)]
        self._parents: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # 所有节点（记忆 ID 集合）
        self._nodes: Set[str] = set()

        # 关系类型的反向映射
        self._relation_reverse = {
            "evidence": "supported_by",
            "version": "previous_version",
            "causal": "caused_by"
        }

        # MemoryStore 引用（用于验证和获取记忆）
        self._store = memory_store

        # 统计
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "edge_types": defaultdict(int)
        }
    
    def add_edge(self, parent_id: str, child_id: str, relation: str):
        """
        添加一条有向边：parent → child

        参数:
            parent_id: 前置记忆 ID
            child_id: 后继记忆 ID
            relation: 关系类型
                - "evidence":  child 的结论基于 parent 提供的证据
                - "version":   child 是 parent 的新版本
                - "causal":    parent 的策略/动作导致了 child 的结果

        返回:
            是否添加成功

        示例:
            # 检索结果的证据支持了总结的结论
            graph.add_edge("mem_search_001", "mem_summary_001", "evidence")

            # 第二次总结是第一次的更新版本
            graph.add_edge("mem_summary_001", "mem_summary_002", "version")
        """
        # 参数校验
        if parent_id == child_id:
            return False

        if relation not in ("evidence", "version", "causal"):
            return False

        # 检查是否重复
        existing = self._children.get(parent_id, [])
        if any(cid == child_id and rel == relation for cid, rel in existing):
            return False

        # 检查是否会形成环（DAG 约束）
        if self._would_create_cycle(parent_id, child_id):
            return False

        # 添加边
        self._children[parent_id].append((child_id, relation))
        self._parents[child_id].append((parent_id, relation))
        self._nodes.add(parent_id)
        self._nodes.add(child_id)

        # 更新统计
        self.stats["total_edges"] += 1
        self.stats["total_nodes"] = len(self._nodes)
        self.stats["edge_types"][relation] += 1

        return True
    
    def remove_edge(self, parent_id: str, child_id: str, relation: str):
        """
        移除一条边

        如果指定 relation，只移除该类型的边；
        否则移除两者之间所有类型的边。
        """
        removed = False

        # 从 children 中移除
        if parent_id in self._children:
            old_list = self._children[parent_id]
            if relation:
                new_list = [(c, r) for c, r in old_list
                           if not (c == child_id and r == relation)]
            else:
                new_list = [(c, r) for c, r in old_list if c != child_id]

            if len(new_list) < len(old_list):
                self._children[parent_id] = new_list
                removed = True
                self.stats["total_edges"] -= (len(old_list) - len(new_list))
                if relation:
                    self.stats["edge_types"][relation] -= 1

        # 从 parents 中移除
        if child_id in self._parents:
            old_list = self._parents[child_id]
            if relation:
                new_list = [(p, r) for p, r in old_list
                           if not (p == parent_id and r == relation)]
            else:
                new_list = [(p, r) for p, r in old_list if p != parent_id]
            self._parents[child_id] = new_list

        return removed

    def get_children(
        self,
        memory_id: str,
        relation: str = None
    ) -> List[Tuple[str, str]]:
        """
        获取某个记忆的所有后继（子节点）

        返回:
            [(child_id, relation_type), ...]
        """
        children = self._children.get(memory_id, [])
        if relation:
            return [(c, r) for c, r in children if r == relation]
        return children

    def get_parents(
        self,
        memory_id: str,
        relation: str = None
    ) -> List[Tuple[str, str]]:
        """
        获取某个记忆的所有前驱（父节点）

        返回:
            [(parent_id, relation_type), ...]
        """
        parents = self._parents.get(memory_id, [])
        if relation:
            return [(p, r) for p, r in parents if r == relation]
        return parents

    def get_evidence_chain(
        self,
        memory_id: str,
        max_depth: int = 10
    ) -> List[List[str]]:
        """
        追溯证据链：从当前记忆一直追溯到原始证据

        返回:
            多条路径，每条路径是从源头到目标的一系列记忆 ID

        示例:
            graph.get_evidence_chain("mem_summary_001")
            → [
                ["mem_search_001", "mem_summary_001"],
                ["mem_search_002", "mem_analysis_001", "mem_summary_001"]
            ]
        """
        chains = []
        self._dfs_upstream(memory_id, [], chains, max_depth, "evidence")
        return chains

    def get_reasoning_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 10
    ) -> Optional[List[str]]:
        """
        查找从 start 到 end 的推理路径

        返回:
            路径上的记忆 ID 列表（包含两端），或 None
        """
        visited = set()
        path = []
        return self._bfs_path(start_id, end_id, visited, path, max_depth)

    def get_all_ancestors(self, memory_id: str) -> Set[str]:
        """获取某个记忆的所有祖先节点"""
        ancestors = set()
        self._collect_upstream(memory_id, ancestors)
        return ancestors

    def get_all_descendants(self, memory_id: str) -> Set[str]:
        """获取某个记忆的所有后代节点"""
        descendants = set()
        self._collect_downstream(memory_id, descendants)
        return descendants

    def get_subgraph(
        self,
        root_ids: List[str],
        max_depth: int = 5
    ) -> Dict:
        """
        获取以 root_ids 为根的子图

        返回:
            {
                "nodes": [memory_id, ...],
                "edges": [(parent, child, relation), ...]
            }
        """
        nodes = set()
        edges = []

        queue = [(rid, 0) for rid in root_ids]
        visited = set()

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            nodes.add(current)

            for child_id, relation in self.get_children(current):
                edges.append((current, child_id, relation))
                queue.append((child_id, depth + 1))

        return {
            "nodes": list(nodes),
            "edges": edges
        }

    # ================================================================
    # 图的维护
    # ================================================================

    def remove_node(self, memory_id: str):
        """
        移除一个节点及其所有关联边
        """
        # 移除出边
        for child_id, _ in self.get_children(memory_id):
            self.remove_edge(memory_id, child_id)

        # 移除入边
        for parent_id, _ in self.get_parents(memory_id):
            self.remove_edge(parent_id, memory_id)

        # 移除节点
        self._children.pop(memory_id, None)
        self._parents.pop(memory_id, None)
        self._nodes.discard(memory_id)

        self.stats["total_nodes"] = len(self._nodes)

    def merge_similar_nodes(self, id_a: str, id_b: str):
        """
        合并两个相似节点

        将 id_b 的所有边转移到 id_a，然后删除 id_b。
        """
        # 将 id_b 的父节点连接到 id_a
        for parent_id, relation in self.get_parents(id_b):
            self.add_edge(parent_id, id_a, relation)

        # 将 id_b 的子节点连接到 id_a
        for child_id, relation in self.get_children(id_b):
            self.add_edge(id_a, child_id, relation)

        # 删除 id_b
        self.remove_node(id_b)

    # ================================================================
    # 环检测（DAG 约束）
    # ================================================================

    def _would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        """
        检查添加 parent → child 边是否会形成环

        如果 child 已经是 parent 的祖先，则会形成环。
        """
        if parent_id == child_id:
            return True
        # BFS 从 child 向上游查找，看是否会遇到 parent
        return parent_id in self.get_all_ancestors(child_id)

    # ================================================================
    # 图遍历（内部）
    # ================================================================

    def _dfs_upstream(
        self,
        current: str,
        current_path: List[str],
        all_chains: List[List[str]],
        max_depth: int,
        relation: str = None
    ):
        """DFS 向上游追溯"""
        if len(current_path) >= max_depth:
            return

        parents = self.get_parents(current, relation)

        if not parents:
            # 到达源头，保存路径
            all_chains.append(current_path + [current])
            return

        for parent_id, _ in parents:
            self._dfs_upstream(
                parent_id,
                current_path + [current],
                all_chains,
                max_depth,
                relation
            )

    def _bfs_path(
        self,
        start: str,
        end: str,
        visited: Set[str],
        path: List[str],
        max_depth: int
    ) -> Optional[List[str]]:
        """BFS 查找最短路径"""
        from collections import deque

        queue = deque([(start, [start])])

        while queue:
            current, current_path = queue.popleft()

            if current == end:
                return current_path

            if len(current_path) > max_depth:
                continue

            for child_id, _ in self.get_children(current):
                if child_id not in current_path:
                    queue.append((child_id, current_path + [child_id]))

        return None

    def _collect_upstream(self, current: str, collected: Set[str]):
        """收集所有祖先"""
        for parent_id, _ in self.get_parents(current):
            if parent_id not in collected:
                collected.add(parent_id)
                self._collect_upstream(parent_id, collected)

    def _collect_downstream(self, current: str, collected: Set[str]):
        """收集所有后代"""
        for child_id, _ in self.get_children(current):
            if child_id not in collected:
                collected.add(child_id)
                self._collect_downstream(child_id, collected)

    # ================================================================
    # 统计与导出
    # ================================================================

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "edge_types": dict(self.stats["edge_types"]),
            "roots": self._find_roots(),
            "leaves": self._find_leaves()
        }

    def _find_roots(self) -> List[str]:
        """查找根节点（没有父节点的节点）"""
        return [n for n in self._nodes if not self._parents.get(n)]

    def _find_leaves(self) -> List[str]:
        """查找叶节点（没有子节点的节点）"""
        return [n for n in self._nodes if not self._children.get(n)]

    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            "nodes": list(self._nodes),
            "edges": [
                (p, c, r)
                for p in self._children
                for c, r in self._children[p]
            ]
        }

    def clear(self):
        """清空图"""
        self._children.clear()
        self._parents.clear()
        self._nodes.clear()
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "edge_types": defaultdict(int)
        }