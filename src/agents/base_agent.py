import asyncio
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, agent_id: str, role: str, 
                 protocol: ProtocolHandler,
                 state_transfer: StateTransferModule,
                 memory: SharedMemoryModule):
        self.agent_id = agent_id
        self.role = role
        self.protocol = protocol
        self.state_transfer = state_transfer
        self.memory = memory
        self.inbox = asyncio.Queue()
        
    async def register(self):
        """注册到系统"""
        hello_msg = StructuredMessage(
            msg_id=self._gen_id(),
            msg_type=MessageType.HELLO,
            sender_id=self.agent_id,
            receiver_id="router",
            timestamp=time.time(),
            parameters=self.get_capability().__dict__
        )
        await self.send_message(hello_msg)
        
    @abstractmethod
    def get_capability(self) -> AgentCapability:
        pass
    
    @abstractmethod
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        pass
    
    async def send_message(self, msg: StructuredMessage):
        """发送结构化消息"""
        data = msg.serialize()
        # 通过IPC/Socket发送
        await self.protocol.dispatch(msg)
    
    async def search_memory(self, query: str) -> List[MemoryUnit]:
        """搜索相关记忆"""
        query_emb = self.state_transfer.encode_state(query)
        return await self.memory.hybrid_search(query, query_emb)
    
    def _gen_id(self) -> str:
        return f"{self.agent_id}_{uuid.uuid4().hex[:8]}"