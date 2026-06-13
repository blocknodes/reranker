#!/usr/bin/env python
import uvicorn
import logging
import math
import gc
import sys
import re
import json
import time
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt
import torch
from log import init_logger

# 初始化日志
logger = init_logger("rerank.log")
__all__ = ['logger', 'init_logger']

# 初始化FastAPI应用
app = FastAPI(title="Rerank Service (vllm backend)")

# ========== 数据模型 ==========
class QADocs(BaseModel):
    query: Optional[str]
    documents: Optional[List[str]]
    filenames: Optional[List[str]] = None  # 新增可选字段
    instruction: Optional[str] = None      # 新增可选字段
    flag: Optional[int] = 0                # 新增标志位(0:qd 1:qq)

# ========== 全局变量 ==========
tokenizer = None
model = None
suffix_tokens = None
true_token = None
false_token = None
sampling_params = None
max_length = 8192  # 最大序列长度
MODEL_PATH = None   # 模型路径(通过命令行传入)

# ========== 单例模式保持(用于兼容原有结构) ==========
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# ========== 重排序核心类(基于vllm) ==========
class ReRanker(metaclass=Singleton):
    def preprocess(self, text: str) -> str:
        """文本预处理，保留原有的清洗逻辑"""
        if not text:
            return ""
        text = re.sub(r'!\[\]\([^)]*\)', '', text)  # 去除图片链接
        text = re.sub(r'\s+', ' ', text)            # 合并空格
        text = re.sub(r'([^a-zA-Z0-9\s])\1{1,}', r'\1', text)  # 去除重复标点
        return text.strip()

    def clear_and_duplicate(self, q_d: QADocs) -> (List[str], List[int]):
        """文档去重并保留原始索引，保持原有逻辑"""
        seen = set()
        unique_docs = []
        original_indices = []

        for idx, doc in enumerate(q_d.documents):
            cleaned_doc = self.preprocess(doc)
            if cleaned_doc not in seen:
                seen.add(cleaned_doc)
                unique_docs.append(cleaned_doc)
                original_indices.append([idx])
            else:
                for i, u_doc in enumerate(unique_docs):
                    if u_doc == cleaned_doc:
                        original_indices[i].append(idx)
        return unique_docs, original_indices

    def format_instruction(self, instruction, query, doc):
        """格式化指令为模型输入格式"""
        return [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]

    def process_inputs(self, pairs, instruction):
        """处理输入为vllm所需格式"""
        global tokenizer, max_length, suffix_tokens

        messages = [self.format_instruction(instruction, query, doc) for query, doc in pairs]
        messages = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False
        )
        # 截断并添加后缀
        messages = [ele[:max_length - len(suffix_tokens)] + suffix_tokens for ele in messages]
        return [TokensPrompt(prompt_token_ids=ele) for ele in messages]

    def compute(self, query: str, documents: List[str], instruction: str):
        """使用vllm模型计算相关性分数"""
        global model, sampling_params, true_token, false_token

        if not query or not documents:
            return []

        # 构建查询-文档对
        pairs = [(query, doc) for doc in documents]
        # 处理输入
        inputs = self.process_inputs(pairs, instruction)

        try:
            outputs = model.generate(inputs, sampling_params, use_tqdm=False)
            scores = []

            for output in outputs:
                # 获取最后一个token的logprobs
                final_logits = output.outputs[0].logprobs[-1]

                # 获取"yes"和"no"的log概率
                true_logit = final_logits[true_token].logprob if true_token in final_logits else -10.0
                false_logit = final_logits[false_token].logprob if false_token in final_logits else -10.0

                # 计算概率并归一化
                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                score = true_score / (true_score + false_score) if (true_score + false_score) > 0 else 0.0

                scores.append(score)

            return scores
        except Exception as e:
            logger.error(f"模型计算失败: {str(e)}")
            raise

# 实例化重排序器
reranker = ReRanker()

# ========== 服务生命周期管理 ==========
@app.on_event("startup")
def load_model():
    """启动时加载vllm模型和分词器"""
    global tokenizer, model, suffix_tokens, true_token, false_token, sampling_params, MODEL_PATH

    try:
        if not MODEL_PATH:
            raise ValueError("未指定模型路径，请通过命令行参数传入")

        logger.info(f"从路径加载模型: {MODEL_PATH}")

        # 确定GPU数量
        number_of_gpu = torch.cuda.device_count()
        logger.info(f"检测到 {number_of_gpu} 个GPU")

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        # 加载vllm模型
        model = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=number_of_gpu,
            max_model_len=max_length,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8
        )

        # 准备后缀和特殊token
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

        # 配置采样参数
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[true_token, false_token],
        )

        # 模型预热
        test_query = "test query"
        test_docs = ["test document"]
        test_instruction = "Test instruction"
        test_scores = reranker.compute(test_query, test_docs, test_instruction)
        logger.info("模型预热成功")

    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        destroy_model_parallel()
        gc.collect()
        raise

@app.on_event("shutdown")
def shutdown_model():
    """关闭时释放模型资源"""
    global model
    if model is not None:
        destroy_model_parallel()
        gc.collect()
        logger.info("模型资源已释放")

# ========== API 路由 ==========
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post('/rerank')
async def handle_post_request(docs: QADocs):
    start_time = time.time()
    try:
        logger.info("收到重排序请求")

        # 验证输入
        if not docs.query or not docs.documents:
            logger.warning("查询或文档列表为空")
            return {"code": 0, "data": []}

        # 文档去重
        unique_docs, original_indices = reranker.clear_and_duplicate(docs)
        if not unique_docs:
            logger.info("去重后文档为空")
            return {"code": 0, "data": []}

        # 获取指令(使用默认值如果未提供)
        instruction = docs.instruction or "Given a web search query, retrieve relevant passages that answer the query"

        # 计算分数
        scores = reranker.compute(docs.query, unique_docs, instruction)

        # 映射分数到原始索引
        score_map = {}
        for idx_list, score in zip(original_indices, scores):
            for idx in idx_list:
                score_map[idx] = float(score)
        restored_scores = [score_map[i] for i in range(len(docs.documents))]

        # 构建结果
        results = [
            {"index": i, "relevance_score": score}
            for i, score in enumerate(restored_scores)
        ]
        response_data = {
            "code": 0,
            "data": [{
                "value": results,
                "status": 0,
                "detail": "",
                "msg": ""
            }]
        }

        # 记录处理时间
        elapsed_time = time.time() - start_time
        logger.info(f"请求处理成功，耗时 {elapsed_time:.3f} 秒. 请求内容: {json.dumps(docs.dict(), ensure_ascii=False)}. 响应内容: {response_data}")
        return response_data

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.exception(f"请求处理失败，耗时 {elapsed_time:.3f} 秒. 请求内容: {json.dumps(docs.dict(), ensure_ascii=False)}. 错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# ========== 启动服务 ==========
if __name__ == "__main__":
    try:
        # 解析命令行参数
        if len(sys.argv) < 2:
            logger.error("请提供模型路径作为第一个命令行参数")
            sys.exit(1)

        MODEL_PATH = sys.argv[1]
        # 端口配置(默认8080)
        PORT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8080

        logger.info(f"启动服务，端口: {PORT}，模型路径: {MODEL_PATH}")
        uvicorn.run(app, host='0.0.0.0', port=PORT, reload=False, timeout_keep_alive=75)
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        print(f"API启动失败！\n报错：\n{e}")