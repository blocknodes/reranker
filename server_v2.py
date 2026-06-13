#!/usr/bin/env python
import uvicorn
import logging
import math
import gc
import sys
import re
import json
import time
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt
import torch
from log import init_logger
import random
import requests
import asyncio
import aiohttp
import os

# 初始化日志
logger = init_logger("rerank.log")
__all__ = ['logger', 'init_logger']

# 初始化FastAPI应用
app = FastAPI(title="Rerank Service (vllm backend)")

# ===================== 新增：BM25 客户端 =====================
class precise_matchingClient:
    """precise_matching接口客户端（调用独立部署的precise_matching FastAPI服务）"""
    def __init__(
        self,
        base_url: str = "http://10.18.231.45:32563",
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {"Content-Type": "application/json"}

    def get_precise_matching_scores(
        self,
        query: str,
        documents: List[str],
        k1: float = 0.2,
        b: float = 0.75
    ) -> List[float]:
        """
        同步调用precise_matching接口获取归一化得分
        :param query: 查询语句
        :param documents: 文档列表
        :param k1: precise_matching参数k1
        :param b: precise_matching参数b
        :return: 每个文档的归一化得分列表
        """
        try:
            response = requests.post(
                url=f"{self.base_url}/precise_matching/rank",
                json={
                    "query": query,
                    "documents": documents,
                    "k1": k1,
                    "b": b
                },
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["normalized_scores"]
        except Exception as e:
            logger.error(f"调用precise_matching接口失败: {str(e)}")
            # 降级策略：返回全1分（不影响最终融合结果）
            return [1.0 for _ in documents]

    async def async_get_precise_matching_scores(
        self,
        query: str,
        documents: List[str],
        meta_data: List[dict],
        k1: float = 0.2,
        b: float = 0.75
    ) -> List[float]:
        """
        异步调用precise_matching接口获取归一化得分（适配FastAPI异步环境）
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    url=f"{self.base_url}/precise_matching/rank",
                    json={
                        "query": query,
                        "documents": documents,
                        "meta_data": meta_data,
                        "k1": k1,
                        "b": b
                    },
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["normalized_scores"],result['weights']['alpha']
        except Exception as e:
            logger.error(f"异步调用precise_matching接口失败 [{type(e).__name__}]: {str(e)} | base_url={self.base_url}")
            return [1.0 for _ in documents], 0

# ===================== 原有代码（仅修改precise_matching相关部分） =====================
class AsyncChatCompletionClient:
    """异步调用Chat Completions API的客户端（适配FastAPI异步环境）"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        default_model: str = "filter"  # 默认模型名称
    ):
        """
        初始化客户端

        Args:
            base_url: API基础地址
            timeout: 请求超时时间（秒）
            headers: 自定义请求头
            default_model: 默认使用的模型名称（批量请求时自动使用）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.headers = headers or {"Content-Type": "application/json"}
        self.default_model = default_model

    async def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        异步创建单个聊天补全请求

        Args:
            model: 模型名称（如filter）
            messages: 消息列表，格式为[{"role": "...", "content": "..."}]
            **kwargs: 其他可选参数（如temperature、max_tokens等）

        Returns:
            API响应数据（字典格式）

        Raises:
            aiohttp.ClientError: HTTP请求错误
            asyncio.TimeoutError: 请求超时
            ValueError: 响应JSON解析失败
        """
        # 构建请求体
        payload = {
            "model": model,
            "messages": messages,** kwargs
        }
        async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
            try:
                async with session.post(
                    url=f"{self.base_url}/v1/chat/completions",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    try:
                        return await response.json()
                    except ValueError as e:
                        raise ValueError(f"响应JSON解析失败: {e}") from e
            except aiohttp.ClientError as e:
                raise aiohttp.ClientError(f"HTTP请求失败: {e}") from e
            except asyncio.TimeoutError as e:
                raise asyncio.TimeoutError(f"请求超时（{self.timeout.total}秒）") from e

    async def batch_requests(
        self,
        query: str,
        content_list: List[str],  # 仅接收content字符串列表
        model: Optional[str] = None,** kwargs: Any
    ) -> List[Union[Dict[str, Any], Exception]]:
        """
        异步批量执行多个聊天补全请求（仅需传入content列表）

        Args:
            content_list: 内容字符串列表，每个元素对应一个请求的content
            model: 可选，指定模型名称（默认使用客户端初始化的default_model）
            **kwargs: 其他请求参数（如temperature、max_tokens等）

        Returns:
            按请求顺序返回的结果列表，每个元素为响应字典或异常对象
        """
        # 使用指定模型或默认模型
        target_model = model or self.default_model
        requests = [
            {
                "model": target_model,
                "messages": [{"role": "user", "content": content}],** kwargs
            }
            for content in [query]+content_list
        ]
        tasks = [
            self.create_chat_completion(**req)
            for req in requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [item['choices'][0]['message']['content'] for item in results]
        print(results)
        if results[0] == '无':
            return {'scores':[1 * len(content_list)]}
        else:
            scores = []
            for item in results[1:]:
                if item == results[0] or item == '无':
                    scores.append(1)
                else:
                    scores.append(0)
            return  {'scores': scores}

class QADocs(BaseModel):
    query: Optional[str]
    documents: Optional[List[str]]
    meta_data: Optional[List[dict]] = None  # 新增可选字段
    instruction: Optional[str] = None      # 新增可选字段

class RerankClient:
    def __init__(self, base_url: str = "https://inner-apisix-test.hisense.com/kbp/rerank",
                 user_key: str = "sxwox9bfz6hrunpeo4dsndfjtniaqmj3"):
        self.base_url = base_url
        self.user_key = user_key
        self.rerank_endpoint = f"{self.base_url}/rerank"
        logging.info(f"Rerank client initialized with endpoint: {self.rerank_endpoint}")

    def rerank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> Dict:
        if not documents:
            logging.warning("文档列表为空，直接返回空结果")
            return {"scores": []}
        payload = {
            "query": query,
            "documents": documents,
            "instruction": instruction
        }
        params = {
            "user_key": self.user_key
        }
        try:
            logging.info(f"发送排序请求，查询: {query[:50]}..., 文档数量: {len(documents)}")
            response = requests.post(
                self.rerank_endpoint,
                json=payload,
                params=params,
                headers={"Content-Type": "application/json"},
                timeout=1
            )
            response.raise_for_status()
            result = response.json()
            logging.info(f"排序请求成功，返回{len(result.get('scores', []))}个分数")
            return result
        except requests.exceptions.RequestException as e:
            logging.error(f"排序请求失败: {str(e)}")
            return {"scores": []}

class ReRanker():
    def __init__(self, base_url):
        self.base_url = base_url
        self.filter_client = AsyncChatCompletionClient()
        # 初始化precise_matching客户端（指定独立部署的precise_matching服务地址）
        addr = os.getenv("PRECISE_SERVER",None)
        self.precise_matching_client = precise_matchingClient(
            base_url=f"http://{addr}",  # 替换为实际的precise_matching服务地址
            timeout=15
        )

    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'!\[\]\([^)]*\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([^a-zA-Z0-9\s])\1{1,}', r'\1', text)
        return text.strip()

    def clear_and_duplicate(self, q_d: QADocs) -> (List[str], List[int]):
        unique_docs = []
        original_indices = []
        for idx, doc in enumerate(q_d.documents):
            cleaned_doc = self.preprocess(doc)
            unique_docs.append(cleaned_doc)
            original_indices.append([idx])
        return unique_docs, original_indices

    async def compute(self, query: str, documents: List[str], instruction: str, type:str='doc', meta_data:dict=None):
        if not query or not documents:
            return []

        # 原有rerank逻辑
        addr = os.getenv("RERANK_SERVER",None)
        client_configs = {
            'doc': [
                #("http://10.18.231.31:30287/", "reranker_q"),
                #("http://10.18.231.46:30642/", "reranker-filter"),
                (f"http://{addr}/", "reraner-copy"),

            ],
            'qna': [
                #("http://10.18.231.31:30287/", "reranker_q"),
                #("http://10.18.231.46:30642/", "reranker-filter"),
                (f"http://{addr}/", "reraner-copy"),
            ]
        }
        configs = client_configs.get(type, client_configs['doc'])
        for idx, (base_url, client_name) in enumerate(configs):
            logger.info(f"尝试使用{client_name}客户端 (地址: {base_url})，序号: {idx+1}")
            client = RerankClient(base_url=base_url)
            ranked_results = client.rerank(query, documents, instruction)
            print(f'#######{ranked_results} {len(ranked_results)}')
            if 'data' in ranked_results:
                logger.info(f"成功使用{client_name}客户端获取结果")
                break

        if type not in ['doc', 'qna']:
            ranked_results = await self.filter_client.batch_requests(
                query = query,
                content_list=documents
            )

        if 'scores' not in ranked_results:
            ranked_results['scores']=[item['relevance_score'] for item in ranked_results['data'][0]['value']]


        # 1. 拼接文档（保留原有逻辑）
        if type=='doc' and meta_data:
            #print(f'############{meta_data}')
            docs_for_precise_matching=[]
            for i in range(len(documents)):
                if meta_data[i]['fileName']:
                    docs_for_precise_matching.append(documents[i] + meta_data[i]['fileName'])
                else:
                    docs_for_precise_matching.append(documents[i])

            #docs_for_precise_matching = [documents[i] + meta_data[i]['fileName'] for i in range(len(documents)) if meta_data[i]['fileName'] else documents[i]]
        else:
            docs_for_precise_matching = documents.copy()


        precise_matching_scores,alpha = await self.precise_matching_client.async_get_precise_matching_scores(
            query=query,
            documents=docs_for_precise_matching,
            meta_data=meta_data)
        logger.info(f"precise_matching得分: {precise_matching_scores}")

        # 3. 分数融合（原有逻辑）

        beta=1-alpha
        for i in range(len(ranked_results['scores'])):
            if precise_matching_scores[i]==0:
                ranked_results['scores'][i]=0
            else:
                ranked_results['scores'][i] = alpha * precise_matching_scores[i] + beta * ranked_results['scores'][i]

        # 4. 子串完全匹配硬规则：query 作为完整子串出现在文档中，直接打最高分
        #    （忽略空白差异，避免 "xx pro"/"xxpro" 之类的空格扰动）
        def _norm(s: str) -> str:
            return re.sub(r'\s+', '', s or '').lower()

        norm_query = _norm(query)
        if norm_query:
            for i in range(len(documents)):
                if norm_query in _norm(documents[i]):
                    ranked_results['scores'][i] = 1.0
                    logger.info(f"文档[{i}]命中query完整子串，直接置为最高分1.0")

        return ranked_results['scores']

# ========== API 路由 ==========
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post('/rerank')
async def handle_post_request(docs: QADocs):
    start_time = time.time()
    try:
        logger.info("收到重排序请求")
        meta_data=docs.meta_data
        logger.info(f"meta_data is {meta_data}")

        type='doc'
        if meta_data:
            type='qna'
            for item in meta_data:
                if item['kind'] != 'qna':
                    type = 'doc'
                    break

        if not docs.query or not docs.documents:
            logger.warning("查询或文档列表为空")
            return {"code": 0, "data": []}

        reranker = ReRanker("http://10.18.231.47:30373/")
        unique_docs, original_indices = reranker.clear_and_duplicate(docs)
        if not unique_docs:
            logger.info("去重后文档为空")
            return {"code": 0, "data": []}

        instruction = docs.instruction or "Given a web search query, retrieve relevant passages that answer the query"
        scores = await reranker.compute(docs.query, unique_docs, instruction, type, meta_data)

        score_map = {}
        for idx_list, score in zip(original_indices, scores):
            for idx in idx_list:
                score_map[idx] = float(score)
        restored_scores = [score_map[i] for i in range(len(docs.documents))]

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
        logger.info(f"启动服务")
        uvicorn.run(app, host='0.0.0.0', port=8080, reload=False)
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        print(f"API启动失败！\n报错：\n{e}")