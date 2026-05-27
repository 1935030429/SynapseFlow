"""
runtime/model_manager.py
模型管理器（单例）

使用 vLLM 的 LLM 类直接加载 Qwen3-8B 进行推理。
所有 Agent 共享同一个 LLM 实例。
"""

import numpy as np
from typing import Dict, Any, List, Optional
from vllm import LLM, SamplingParams
import yaml


class ModelManager:
    """
    模型管理器（单例）
    
    使用 vLLM Python API 直接推理，不需要启动 HTTP 服务。
    所有 Agent 共享同一个 LLM 实例。
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_path: str = None,
        tensor_parallel_size: int = 8,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 4096
    ):
        if self._initialized:
            return
        config = self.get_config()
        
        self.model_path = model_path or config.get("model", {"path": {}}).get("path", "") or "model/Qwen/Qwen3-8B"
        
        print(f"[ModelManager] 加载模型: {self.model_path}")
        
        # vLLM 直接加载
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        
        # 获取 tokenizer（vLLM 内部有）
        self.tokenizer = self.llm.get_tokenizer()
        
        self._initialized = True
        print(f"[ModelManager] 模型加载完成")

    def get_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        返回:
            模型配置字典
        """
        _path = "Project/SynapseFlow/config.yaml"
        with open(_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    # ================================================================
    # 文本生成
    # ================================================================
    
    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量生成文本
        
        参数:
            prompts: 输入文本列表
            temperature: 温度
            max_tokens: 最大生成长度
            top_p: nucleus sampling
            stop: 停止词列表
        
        返回:
            [
                {
                    "text": 生成的完整文本（含输入）,
                    "output": 仅生成的部分,
                    "tokens": Token 数量
                },
                ...
            ]
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            full_text = output.outputs[0].text
            results.append({
                "text": output.prompt + full_text,
                "output": full_text,
                "tokens": len(output.outputs[0].token_ids),
                "prompt_tokens": len(output.prompt_token_ids)
            })
        
        return results
    
    def generate_single(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: List[str] = None
    ) -> Dict[str, Any]:
        """
        单条生成（便捷方法）
        
        参数:
            prompt: 输入文本
            temperature: 温度
            max_tokens: 最大生成长度
            stop: 停止词
        
        返回:
            {"text": ..., "output": ..., "tokens": ...}
        """
        results = self.generate(prompts=[prompt], temperature=temperature, max_tokens=max_tokens, stop=stop)
        return results[0]
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Chat 格式生成
        
        参数:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            temperature: 温度
            max_tokens: 最大生成长度
        
        返回:
            {"text": ..., "output": ..., "tokens": ...}
        """
        # 用 tokenizer 的 chat template 转成 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return self.generate_single(prompt, temperature, max_tokens)
    
    # ================================================================
    # Embedding
    # ================================================================
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本编码为向量（使用模型的 hidden state）
        
        参数:
            texts: 文本列表
        
        返回:
            numpy 数组，shape (len(texts), hidden_dim)
        """
        # vLLM 的 LLM 类直接调用 generate 无法获取 hidden states
        # 如果需要 embedding，可以用 encode 方法或单独加载 embedding 模型
        # 这里使用 tokenizer + 简单编码作为替代
        
        # 方法1：使用 vLLM 的 embed 方法（如果可用）
        if hasattr(self.llm, 'encode'):
            outputs = self.llm.encode(texts)
            return np.array([o.outputs.embedding for o in outputs])
        
        # 方法2：用 tokenizer 的输入 ID 平均值作为简单向量
        # 这只是原型阶段的临时方案
        vectors = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            # 创建一个简单的 bag-of-tokens 向量
            vec = np.zeros(self.tokenizer.vocab_size)
            for t in tokens:
                if t < len(vec):
                    vec[t] = 1.0
            vectors.append(vec)
        
        return np.array(vectors)
    
    def encode_single(self, text: str) -> np.ndarray:
        """单条文本编码"""
        return self.encode([text])[0]
    
    # ================================================================
    # 工具
    # ================================================================
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 Token 数"""
        return len(self.tokenizer.encode(text))
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """批量计算 Token 数"""
        return [len(self.tokenizer.encode(t)) for t in texts]