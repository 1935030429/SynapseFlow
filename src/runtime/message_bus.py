import asyncio
from typing import Callable, Dict, List

from src.protocol.messages import MessageType, StructuredMessage

class MessageBus:
    """
    消息总线，负责消息的发送和接收
    """
    def __init__(self, router, agents: List):
        self.router = router
        self.agents = {a.gent_id: a for a in agents}
        
        self._temp_handlers: Dict[str, Callable] = {}
        
        self.message_log: List[StructuredMessage] = []
        
        self.stats = {
            "total_sent": 0,
            "total_delivered": 0,
            "total_failed": 0
        }
        
    def register_temp_handler(self, msg_id: str, handler: Callable):
        """
        注册临时响应处理器
        
        当某个消息的回复到达时，调用对应的handler。
        用于实现"发送-等待回复"模式。
        """
        self._temp_handlers[msg_id] = handler
    async def send(self, msg: StructuredMessage, target: str = None, timeout: float = 30.0):
        """
        发送消息
        
        参数:
            msg: 要发送的消息
            target: 目标Agent的ID
                    - 指定ID: 点对点发送
                    - None/"broadcast": 广播给所有Agent
            timeout: 超时时间（秒）
        """
        self.stats["total_sent"] += 1
        self.message_log.append(msg)
        
        if target and target != "broadcast":
            await self.__send_to_one(msg, target)
        else:
            await self.__broadcast(msg)
            
    async def __broadcast(self, msg: StructuredMessage):
        """广播给所有Agent"""
        tasks = []
        for agent_id, agent in self.agents.items():
            if agent_id != msg.sender_id:  # 不发给发送方自己
                tasks.append(agent.inbox.put(msg))
        
        if tasks:
            await asyncio.gather(*tasks)
            self.stats["total_delivered"] += len(tasks)
            
    async def __send_to_one(self, msg: StructuredMessage, target_id: str):
        """点对点发送"""
        target_agent = self.agents.get(target_id)
        
        if not target_agent:
            print(f"[BUS] 目标Agent {target_id} 不存在")
            self.stats["total_failed"] += 1
            
            # 如果有临时处理器，返回错误
            handler = self._temp_handlers.pop(msg.msg_id, None)
            if handler:
                error_msg = StructuredMessage(
                    msg_type=MessageType.TASK_ERROR,
                    task_id=msg.task_id,
                    result_data={"error": f"Agent {target_id} 不可用"}
                )
                handler(error_msg)
            return
        # 投递到目标Agent的收件箱
        try:
            await asyncio.wait_for(target_agent.inbox.put(msg), timeout=30.0)
            self.stats["total_delivered"] += 1
        except asyncio.TimeoutError:
            print(f"[BUS] 向Agent {target_id} 发送消息超时")
            self.stats["total_failed"] += 1
            # 处理超时情况
            handler = self._temp_handlers.pop(msg.msg_id, None)
            if handler:
                error_msg = StructuredMessage(
                    msg_type=MessageType.TASK_ERROR,
                    task_id=msg.task_id,
                    result_data={"error": f"Agent {target_id} 响应超时"}
                )
                handler(error_msg)
        except Exception as e:
            print(f"[BUS] 向Agent {target_id} 发送消息出错: {e}")
            self.stats["total_failed"] += 1
        