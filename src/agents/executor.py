"""

角色：代码执行与工具调用（CodeAct 模式）
模型：不需要 LLM 推理，只需要沙箱执行环境

职责：
1. 接收 TASK_REQUEST（action="execute_code" 或 "call_api"）
2. 在沙箱中安全执行代码
3. 收集执行结果（stdout/stderr/exit_code）
4. 结果写入共享内存
5. 存储执行经验到记忆库
6. 返回 TASK_RESULT

CodeAct 模式说明：
- LLM 生成可执行代码，Executor 在沙箱中运行
- 沙箱提供安全隔离：内存限制、网络禁用、文件系统隔离
- 支持 Docker 或 nsjail 两种沙箱后端
"""

import json
import time
from typing import List, Dict, Optional, Any

from src.agents.base_agent import BaseAgent
from src.protocol.messages import StructuredMessage, MessageType


class ExecutorAgent(BaseAgent):
    """
    执行Agent

    不需要 LLM 推理，只负责在沙箱中执行代码。
    """

    def __init__(self, sandbox_config: Dict = None, **kwargs):
        """
        参数:
            sandbox_config: 沙箱配置
                {
                    "type": "nsjail",       # "nsjail" 或 "docker"
                    "timeout": 30,          # 超时时间（秒）
                    "max_memory": "256m",   # 最大内存
                    "readonly": True,       # 只读文件系统
                    "network": False        # 禁用网络
                }
        """
        super().__init__(**kwargs)
        
        self.sandbox_config = sandbox_config or {
            "type": "nsjail",
            "timeout": 30,
            "max_memory": "256m",
            "readonly": True,
            "network": False
        }
        
        # 初始化沙箱
        self.sandbox = self._create_sandbox()

    def _create_sandbox(self):
        """创建沙箱实例"""
        sandbox_type = self.sandbox_config.get("type", "nsjail")

        if sandbox_type == "docker":
            from src.sandbox.docker_sandbox import DockerSandbox
            return DockerSandbox(self.sandbox_config)
        elif sandbox_type == "nsjail":
            from src.sandbox.nsjail_sandbox import NsjailSandbox
            return NsjailSandbox(self.sandbox_config)
        else:
            # 默认使用子进程（最简单的隔离方式）
            from src.sandbox.process_sandbox import ProcessSandbox
            return ProcessSandbox(self.sandbox_config)

    def get_capabilities(self) -> List[str]:
        return ["execute_code", "call_api", "process_data"]

    # ================================================================
    # 任务处理入口
    # ================================================================

    async def process_task(self, msg: StructuredMessage) -> Optional[StructuredMessage]:
        """处理任务请求"""
        action = msg.action

        if action == "execute_code":
            return await self._execute_code(msg)
        elif action == "call_api":
            return await self._call_api(msg)
        elif action == "process_data":
            return await self._process_data(msg)
        else:
            return self.make_error(
                action=action,
                error_msg=f"Executor 不支持此动作: {action}",
                receiver_id=msg.sender_id
            )

    # ================================================================
    # 代码执行（核心方法）
    # ================================================================

    async def _execute_code(self, msg: StructuredMessage) -> StructuredMessage:
        """
        在沙箱中执行代码

        输入:
            msg.parameters = {
                "code": "print('hello')",          # 要执行的代码
                "language": "python",              # 语言（默认 python）
                "timeout": 30,                     # 超时（可选）
                "env": {"VAR": "value"},           # 环境变量（可选）
                "files": {"input.txt": "content"}, # 输入文件（可选）
                "upstream_results": {}             # 上游步骤结果
            }
        """
        code = msg.parameters.get("code", "")
        language = msg.parameters.get("language", "python")
        timeout = msg.parameters.get("timeout", self.sandbox_config["timeout"])
        env = msg.parameters.get("env", {})
        files = msg.parameters.get("files", {})
        task_id = msg.task_id or self._generate_id()

        # 如果没有代码，尝试从上游结果中获取
        if not code:
            upstream = msg.parameters.get("upstream_results", {})
            for step_id, result in upstream.items():
                if isinstance(result, dict) and "code" in result:
                    code = result["code"]
                    break

        if not code:
            return self.make_error(
                action="execute_code",
                error_msg="没有可执行的代码",
                receiver_id=msg.sender_id
            )

        # ──── 步骤1：查询记忆，看是否执行过相同代码 ────
        code_hash = self._hash_code(code)
        cached_results = await self.query_memory(
            query_text=f"执行代码: {code_hash}",
            top_k=1
        )

        # 如果命中缓存，直接返回（避免重复执行）
        if cached_results and cached_results[0].confidence > 0.95:
            cached = self._extract_cached_result(cached_results[0])
            if cached:
                return self.make_response(
                    msg_type=MessageType.TASK_RESULT,
                    action="execute_code",
                    result_data={
                        "execution": cached,
                        "from_cache": True
                    },
                    result_type="inline",
                    memory_refs=[cached_results[0].memory_id],
                    receiver_id=msg.sender_id
                )

        # ──── 步骤2：在沙箱中执行 ────
        start_time = time.time()
        exec_result = await self.sandbox.execute(
            code=code,
            language=language,
            timeout=timeout,
            env=env,
            files=files
        )
        elapsed = time.time() - start_time

        # ──── 步骤3：格式化结果 ────
        formatted = self._format_execution_result(exec_result, elapsed, code)

        # ──── 步骤4：写入共享内存 ────
        result_offset = self.write_to_shm(formatted)

        # ──── 步骤5：存储执行经验 ────
        memory_id = await self._store_execution_memory(
            code=code,
            code_hash=code_hash,
            result=formatted,
            task_id=task_id
        )

        # ──── 步骤6：返回 ────
        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="execute_code",
            result_data={
                "result_offset": result_offset,
                "summary": {
                    "exit_code": formatted["exit_code"],
                    "elapsed": round(elapsed, 3),
                    "output_size": len(formatted.get("stdout", "")),
                    "from_cache": False
                }
            },
            result_type="shm_pointer",
            memory_refs=[memory_id],
            receiver_id=msg.sender_id
        )

    # ================================================================
    # API 调用
    # ================================================================

    async def _call_api(self, msg: StructuredMessage) -> StructuredMessage:
        """
        调用外部 API（模拟）

        输入:
            msg.parameters = {
                "api_name": "weather_api",
                "method": "GET",
                "url": "https://api.example.com/...",
                "headers": {},
                "body": {}
            }
        """
        api_name = msg.parameters.get("api_name", "unknown")
        task_id = msg.task_id or self._generate_id()

        # 模拟 API 调用结果
        # 实际实现时，这里应该真正发起 HTTP 请求
        result = {
            "api_name": api_name,
            "status": "success",
            "data": {"message": f"模拟 {api_name} 调用成功"},
            "timestamp": time.time()
        }

        result_offset = self.write_to_shm(result)

        # 存储经验
        await self._store_execution_memory(
            code=f"API调用: {api_name}",
            code_hash=f"api_{api_name}",
            result=result,
            task_id=task_id
        )

        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="call_api",
            result_data={
                "result_offset": result_offset,
                "summary": {"status": "success"}
            },
            result_type="shm_pointer",
            receiver_id=msg.sender_id
        )

    # ================================================================
    # 数据处理
    # ================================================================

    async def _process_data(self, msg: StructuredMessage) -> StructuredMessage:
        """
        处理数据（如统计分析、格式转换）

        本质上也是执行代码，但更侧重数据处理。
        """
        # 如果有 code，走 _execute_code
        if "code" in msg.parameters:
            return await self._execute_code(msg)

        # 否则根据 operation 执行预定义操作
        operation = msg.parameters.get("operation", "")
        data = msg.parameters.get("data", {})

        if operation == "count":
            result = {"count": len(data) if isinstance(data, list) else 0}
        elif operation == "summarize":
            result = {"summary": str(data)[:500]}
        else:
            result = {"processed": data}

        result_offset = self.write_to_shm(result)

        return self.make_response(
            msg_type=MessageType.TASK_RESULT,
            action="process_data",
            result_data={"result_offset": result_offset},
            result_type="shm_pointer",
            receiver_id=msg.sender_id
        )

    # ================================================================
    # 结果格式化
    # ================================================================

    def _format_execution_result(
        self,
        exec_result: Dict,
        elapsed: float,
        code: str
    ) -> Dict[str, Any]:
        """
        格式化执行结果

        沙箱返回的原始格式:
        {
            "exit_code": 0,
            "stdout": "hello\n",
            "stderr": "",
            "timed_out": False,
            "memory_used": 1024000
        }

        格式化后:
        {
            "exit_code": 0,
            "stdout": "hello\n",
            "stderr": "",
            "elapsed": 0.123,
            "timed_out": False,
            "success": True,
            "code_snippet": "print('hello')",
            "code_length": 14
        }
        """
        exit_code = exec_result.get("exit_code", -1)

        return {
            "exit_code": exit_code,
            "stdout": exec_result.get("stdout", ""),
            "stderr": exec_result.get("stderr", ""),
            "elapsed": round(elapsed, 4),
            "timed_out": exec_result.get("timed_out", False),
            "memory_used": exec_result.get("memory_used", 0),
            "success": exit_code == 0 and not exec_result.get("timed_out", False),
            "code_snippet": code[:200],
            "code_length": len(code)
        }

    # ================================================================
    # 缓存
    # ================================================================

    def _hash_code(self, code: str) -> str:
        """计算代码的哈希值（用于缓存）"""
        import hashlib
        return hashlib.md5(code.encode()).hexdigest()[:12]

    def _extract_cached_result(self, memory) -> Optional[Dict]:
        """从记忆中提取缓存的执行结果"""
        if memory.evidence:
            try:
                return json.loads(memory.evidence[0])
            except:
                pass
        return None

    # ================================================================
    # 记忆存储
    # ================================================================

    async def _store_execution_memory(
        self,
        code: str,
        code_hash: str,
        result: Dict,
        task_id: str
    ) -> str:
        """存储执行经验到记忆库"""
        success = result.get("success", False)
        exit_code = result.get("exit_code", -1)

        summary = f"执行代码: {code[:80]}..."
        if success:
            summary += " → 成功"
        else:
            summary += f" → 失败 (exit_code={exit_code})"

        return await self.store_memory(
            summary=summary,
            task_id=task_id,
            evidence=[
                json.dumps(result, ensure_ascii=False),  # 执行结果
                code  # 原始代码
            ],
            strategy=f"沙箱执行, 耗时{result.get('elapsed', 0)}s",
            conclusion="成功" if success else f"失败: exit_code={exit_code}",
            tags=["execution", "codeact", "success" if success else "failed"],
            parent_ids=[]
        )