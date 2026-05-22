"""
protocol/router.py
协议路由器

职责：
1. 维护 Agent 的能力注册表
2. 根据 action 查找能执行它的 Agent 列表
3. 支持 Agent 上下线（注册/注销）
"""

from typing import Dict, List, Optional
import time


class ProtocolRouter:
    """
    协议路由器
    
    核心数据结构：route_table
    一个从 action（能力名）到 agent_id 列表的映射。
    
    例如：
    {
        "plan_task":      ["planner_01"],
        "search":         ["retriever_01"],
        "execute_code":   ["executor_01"],
        "summarize":      ["summarizer_01"],
    }
    """
    
    def __init__(self):
        # 路由表：action → [agent_id, ...]
        self.route_table: Dict[str, List[str]] = {}
        
        # Agent 详细信息：agent_id → {capabilities, role, registered_at, ...}
        self.agent_info: Dict[str, dict] = {}
        
        # 统计
        self.stats = {
            "total_lookups": 0,
            "total_misses": 0,   # 查找失败次数
        }
    
    # ================================================================
    # Agent 注册与注销
    # ================================================================
    
    def register_agent(self, agent_id: str, capabilities: List[str], **extra_info):
        """
        注册一个 Agent 及其能力
        
        参数:
            agent_id: Agent 的唯一标识，如 "planner_01"
            capabilities: 能力列表，如 ["plan_task", "adjust_plan"]
            extra_info: 额外信息，如 role="planner"
        
        示例:
            router.register_agent("planner_01", ["plan_task", "adjust_plan"], role="planner")
            router.register_agent("retriever_01", ["search", "verify"], role="retriever")
        """
        # 1. 记录 Agent 信息
        self.agent_info[agent_id] = {
            "capabilities": capabilities,
            "registered_at": time.time(),
            "last_seen": time.time(),
            **extra_info
        }
        
        # 2. 更新路由表：把 agent_id 加到每个能力的列表中
        for capability in capabilities:
            if capability not in self.route_table:
                self.route_table[capability] = []
            self.route_table[capability].append(agent_id)
    
    def unregister_agent(self, agent_id: str):
        """
        注销一个 Agent
        
        当 Agent 下线时调用，从路由表中移除。
        """
        # 1. 获取该 Agent 的能力列表
        info = self.agent_info.pop(agent_id, None)
        if not info:
            return
        
        # 2. 从每个能力的列表中移除该 agent_id
        for capability in info["capabilities"]:
            if capability in self.route_table:
                agent_list = self.route_table[capability]
                if agent_id in agent_list:
                    agent_list.remove(agent_id)
                
                # 如果该能力下没有 Agent 了，删除这个能力条目
                if not agent_list:
                    del self.route_table[capability]
    
    def heartbeat(self, agent_id: str):
        """
        更新 Agent 的心跳时间
        
        用于健康检查，判断 Agent 是否存活。
        """
        if agent_id in self.agent_info:
            self.agent_info[agent_id]["last_seen"] = time.time()
    
    # ================================================================
    # 路由查找
    # ================================================================
    
    def route_by_action(self, action: str) -> List[str]:
        """
        根据 action 查找能执行它的 Agent 列表
        
        这是 Orchestrator 调用的核心方法。
        
        参数:
            action: 动作名，如 "search", "plan_task"
        
        返回:
            能执行该 action 的 agent_id 列表，如 ["retriever_01"]
            如果没有，返回空列表 []
        
        示例:
            agents = router.route_by_action("search")
            # → ["retriever_01"]
            
            agents = router.route_by_action("unknown_action")
            # → []
        """
        self.stats["total_lookups"] += 1
        
        result = self.route_table.get(action, [])
        
        if not result:
            self.stats["total_misses"] += 1
        
        return result
    
    def route_by_role(self, role: str) -> Optional[str]:
        """
        根据角色名查找 Agent
        
        用于 Orchestrator 直接找 Planner 或 Summarizer。
        
        参数:
            role: 角色名，如 "planner", "summarizer"
        
        返回:
            该角色的 agent_id，如 "planner_01"
            如果没有，返回 None
        """
        for agent_id, info in self.agent_info.items():
            if info.get("role") == role:
                return agent_id
        return None
    
    # ================================================================
    # 查询接口
    # ================================================================
    
    def get_agent_info(self, agent_id: str) -> Optional[dict]:
        """获取某个 Agent 的详细信息"""
        return self.agent_info.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, dict]:
        """获取所有已注册的 Agent"""
        return self.agent_info.copy()
    
    def get_capabilities(self, agent_id: str) -> List[str]:
        """获取某个 Agent 的能力列表"""
        info = self.agent_info.get(agent_id)
        return info["capabilities"] if info else []
    
    def get_all_capabilities(self) -> List[str]:
        """获取系统中所有已注册的能力"""
        return list(self.route_table.keys())
    
    def has_capability(self, action: str) -> bool:
        """检查系统是否支持某个能力"""
        return action in self.route_table and len(self.route_table[action]) > 0
    
    def get_stats(self) -> dict:
        """获取路由统计"""
        return {
            **self.stats,
            "total_agents": len(self.agent_info),
            "total_capabilities": len(self.route_table),
            "hit_rate": 1 - (self.stats["total_misses"] / max(1, self.stats["total_lookups"]))
        }