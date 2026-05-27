import asyncio
from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional
import uuid

from src.protocol.messages import StructuredMessage, MessageType
from src.state_transfer.latent_mas_manager import LatentMASManager
from src.memory.memory_unit import MemoryUnit
from src.protocol.router import ProtocolRouter
from src.runtime.modelManager import ModelManager
from src.memory.memory_retrieval import HybridRetrieval
from src.memory.memory_graph import MemoryGraph
from src.runtime.shared_memory import SharedMemoryManager

class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, agent_id: str, role: str, 
                 router: ProtocolRouter,
                 state_transfer = None,
                 shm: SharedMemoryManager = None,
                 model_manager: ModelManager = None,
                 memory: HybridRetrieval = None,
                 memory_graph: Optional[MemoryGraph] = None,
                 model_config = None,
                 latent_mas: Any = None,
                 latent_steps: int = 50
                 ):
        self.agent_id = agent_id
        self.role = role
        
        self.router = router
        self.shm = shm
        self.state_transfer = state_transfer
        self.memory = memory
        self.memory_graph = memory_graph
        
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.model_manager = model_manager
        
        self.inbox = asyncio.Queue()
        self.running = False
        self.current_task_id: Optional[str] = None
        self.latent_mas: LatentMASManager = None
        
        self.metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "text_tokens_generated": 0,
            "text_tokens_received": 0,
            "memory_queries": 0,
            "memory_hits": 0,
            "memories_created": 0
        }
        
    async def start(self):
        """Start Agent"""
        self.running = True
        
        Capability = self.get_capability()
        self.router.register_agent(
            agent_id=self.agent_id,
            role=self.role,
            capability=Capability,
            # register_callback=self.register,
            inbox=self.inbox
        )
        hello_msg = StructuredMessage(
            msg_type=MessageType.AGENT_HELLO,
            sender_id=self.agent_id,
            receiver_id="broadcast",
            timestamp=time.time(),
            parameters={"role": self.role, "capability": Capability}
        )
        
        await self._route_message(hello_msg)
        asyncio.create_task(self._message_loop())
        
        print(f"Agent [{self.agent_id}] started, Role: {self.role}")
        
    async def stop(self):
        self.running = False
        
        bye_msg = StructuredMessage(
            msg_type=MessageType.AGENT_BYE,
            sender_id=self.agent_id,
            receiver_id="broadcast"
        )
        
        await self._route_message(bye_msg)
        self.router.unregister_agent(self.agent_id)
        print(f"Agent [{self.agent_id}] has stopped")
        
    async def _message_loop(self):
        while self.running:
            try:
                msg: StructuredMessage = await asyncio.wait_for(self.inbox.get(), timeout=1)
                self.metrics["messages_received"] += 1
                
                if msg.task_id:
                    self.current_task_id = msg.task_id
                    
                response = await self.handle_message(msg)
                
                if response:
                    response.metadata["in_reply_to"] = msg.msg_id
                    response.task_id = msg.task_id
                    await self._route_message(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[{self.agent_id}] 处理消息出错: {e}")
                
    async def handle_message(self, msg: StructuredMessage) -> Optional[StructuredMessage]:
        if msg.msg_type == MessageType.TASK_REQUEST:
            return await self.process_task(msg)
        elif msg.msg_type == MessageType.CAPABILITY_QUERY:
            return StructuredMessage(
                msg_type=MessageType.CAPABILITY_RESPONSE,
                sender_id=self.agent_id,
                receiver_id=msg.sender_id,
                parameters={"role": self.role, "capability": self.get_capability()}
            )
        elif msg.msg_type == MessageType.MEMORY_RESULT:
            return None
        elif msg.msg_type == MessageType.TASK_ERROR:
            print(f"\033[31m[{self.agent_id}] get error: {msg.result_data}\033[0m")
            return None
        
        return None
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
    def get_capability(self) -> List[str]:
        """
        返回 Agent 的能力列表
        
        Planner:    ["plan_task", "adjust_plan"]
        Retriever:  ["search", "verify"]
        Executor:   ["execute_code"]
        Summarizer: ["summarize"]
        """
        pass
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用模型生成文本
        
        注意：这是同步方法。vLLM 的 LLM.generate 是同步的。
        如果需要异步，可以用 asyncio.to_thread 包装。
        """
        self.metrics["llm_calls"] += 1
        if stop is not None and len(stop) == 0:
            stop = None
        
        result = self.model_manager.generate_single(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs
        )
        
        self.metrics["text_tokens_generated"] += result["tokens"]
        
        return {
            "content": result.get("output", ""),
            "context": result.get("output", ""),
            "full_text": result.get("text", ""),
            "tokens_used": result.get("tokens", ""),
            "prompt_tokens": result.get("prompt_tokens", 0)
        }
        
    async def generate_text_async(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stop: List[str] = None
    ) -> Dict[str, Any]:
        """
        异步生成文本（用线程池包装同步调用）
        """
        return await asyncio.to_thread(
            lambda: self.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
        )
        
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Chat 格式生成"""
        self.metrics["llm_calls"] += 1
        
        result = self.model_manager.generate_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.metrics["text_tokens_generated"] += result["tokens"]
        
        return {
            "content": result["output"],
            "full_text": result["text"],
            "tokens_used": result["tokens"]
        }
    
    def encode_text(self, text: str) -> Any:
        """文本编码为向量"""
        return self.model_manager.encode_single(text)
    
    def count_tokens(self, text: str) -> int:
        """计算 Token 数"""
        return self.model_manager.count_tokens(text)
    
    async def query_memory(
        self, 
        query_text: str, 
        top_k: int = 5, 
        filter_tags: List[str] = None
        ) -> List[MemoryUnit]:
        if "llm_calls" not in self.metrics:
            self.metrics["llm_calls"] = 0
        self.metrics["llm_calls"] += 1
        
        results = await self.memory.search(
            query_text=query_text,
            top_k=top_k,
            filter_tags=filter_tags
        )
        
        if results:
            self.metrics["memory_hits"] += len(results)
            
        return results
    
    async def store_memory(
        self,
        summary: str,
        task_id: str,
        evidence: List[str] = None,
        strategy: str = "",
        conclusion: str = "",
        tags: List[str] = None,
        parent_ids: List[str] = None
    ) -> str:
        vector = self.encode_text(summary)
        
        memory = MemoryUnit(
            source_agent=self.agent_id,
            task_id=task_id,
            task_theme=summary[:50],
            summary=summary,
            evidence=evidence or [],
            strategy=strategy,
            conclusion=conclusion,
            vector=vector,
            tags=tags or [],
            parent_ids=parent_ids or []
        )
        
        memory_id = await self.memory.store(memory)
        
        if self.memory_graph and parent_ids:
            for parent_id in parent_ids:
                await self.memory_graph.add_edge(
                    parent_id=parent_id,
                    child_id=memory_id,
                    relation="evidance"
                )
                
        self.metrics["memories_created"] += 1
        
        return memory_id
    
    """结构化通信"""
    def make_response(
        self, 
        msg_type: MessageType,
        action: str,
        result_data: Any,
        result_type: str = "inline",
        memory_refs: List[str] = None,
        state_offset: int = None,
        state_type: str = None,
        receiver_id: str = None
    ) -> StructuredMessage:
        return StructuredMessage(
            msg_type=msg_type,
            sender_id=self.agent_id,
            receiver_id=receiver_id or "broadcast",
            action=action,
            result_data=result_data,
            result_type=result_type,
            memory_refs=memory_refs or [],
            state_offset=state_offset,
            state_type=state_type
        )
        
    def make_error(self, action: str, error_msg: str, receiver_id: str = None) -> StructuredMessage:
        return self.make_response(
            msg_type=MessageType.TASK_ERROR,
            action=action,
            result_data={"error": error_msg},
            receiver_id=receiver_id
        )
        
    async def _route_message(self, msg: StructuredMessage):
        msg.sender_id = self.agent_id
        
        targets = []
        if msg.receiver_id and msg.receiver_id != "broadcast":
            targets = [msg.receiver_id]
        elif msg.action:
            targets = self.router.route_by_action(msg.action)
        
        for target_id in targets:
            if target_id == self.agent_id:
                continue
            target_info = self.router.get_agent_info(target_id)
            if target_info and "inbox" in target_info:
                try:
                    target_info["inbox"].put_nowait(msg)
                    self.metrics["messages_sent"] += 1
                except asyncio.QueueFull:
                    print(f"[{self.agent_id}] {target_id} inbox is full")
                    
    """Shared Memory"""
    def write_to_shm(self, data: Any) -> int:
        import msgpack
        data_bytes = msgpack.packb(data)
        if self.shm:
            return self.shm.write_task(self.current_task_id or "unknown", data_bytes)
        return data
    
    def read_from_shm(self, offset: int) -> Any:
        import msgpack
        data_bytes = self.shm.read_task(offset)
        return msgpack.unpackb(data_bytes)
    
    @abstractmethod
    async def process_task(self, msg: StructuredMessage) -> StructuredMessage:
        pass
        # if msg.msg_type == MessageType.STATE_TRANSFER and msg.state_offset is not None:
        #     upstream = self.latent_mas.consume(msg.state_offset)
            
        # offset = self.latent_mas.produce(
        #     input_text=msg.parameters.get("task", ""),
        #     cache_key=msg.task_id
        # )
        
        # return StructuredMessage(
        #     state_offset=offset,
        #     state_type=MessageType.STATE_TRANSFER
        # )
    
    async def send_message(self, msg: StructuredMessage):
        """发送结构化消息"""
        data = msg.serialize()
        # 通过IPC/Socket发送
        await self.protocol.dispatch(msg)
    
    async def search_memory(self, query: str) -> List[MemoryUnit]:
        """搜索相关记忆"""
        query_emb = self.state_transfer.encode_state(query)
        return await self.memory.hybrid_search(query, query_emb)
    
    def _generate_id(self) -> str:
        import uuid
        timestamp = int(time.time())
        short_uuid = uuid.uuid4().hex[:6]
        return f"task_{timestamp}_{short_uuid}"
    
    def get_metrics(self) -> Dict[str, int]:
        return self.metrics.copy()