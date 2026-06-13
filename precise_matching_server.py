import jieba
import math
import re
from collections import defaultdict
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import asyncio
import aiohttp
import json
import os



# 初始化FastAPI应用
app = FastAPI(title="Chinese Precise Matching Text Retrieval API", version="1.0")

# ===================== 原有代码（仅修改BM25相关部分） =====================
class AsyncChatCompletionClient:
    """异步调用Chat Completions API的客户端（适配FastAPI异步环境）"""

    def __init__(
        self,
        base_url: str = "http://10.18.231.45:30642",
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
        return json.loads(results[0])


# 正则模式定义
pattern = '(?!-)[A-Za-z0-9/-]*[A-Za-z0-9/](?<!-)'

# -------------------------- 工具函数 --------------------------
# 型号中常见的分隔符：空格、斜杠、连字符、下划线、反斜杠、点、中文间隔号等
_MODEL_SEP_PATTERN = re.compile(r'[\s/\\\-_.·•·]+')

def normalize_model(s: str) -> str:
    """型号归一化：转小写并剥离分隔符，使 'E8S Pro'/'E8S-Pro'/'E8S/Pro'/'E8SPro' 等价"""
    if not s:
        return ""
    return _MODEL_SEP_PATTERN.sub('', s.lower())

# 仅删除“字母/数字之间”的分隔符（空格 / - _ . \），用于分词前预处理：
# 'E8S Pro电视' -> 'E8SPro电视'，'E8S-Pro' -> 'E8SPro'；中文之间的空格不受影响。
_ALNUM_SEP_PATTERN = re.compile(r'(?<=[0-9A-Za-z])[\s/\\\-_.]+(?=[0-9A-Za-z])')

def merge_alnum_separators(text: str) -> str:
    """折叠字母数字型号串内部的分隔符，使 BM25 分词对 'E8S Pro'/'E8SPro' 一致"""
    if not text:
        return text
    return _ALNUM_SEP_PATTERN.sub('', text)

# 规则型号提取：型号一般是“字母+数字”混合的连续串（如 U5Q / E8SPro / WF100E5Q / KFR35GW）
_MODEL_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9]+')

def extract_model_by_rule(query: str) -> Optional[str]:
    """
    filter 接口不稳定时的降级方案：从 query 中按规则提取型号。
    规则：先折叠分隔符，再取“同时含字母和数字”的连续字母数字串；
    若有多个，取最长的（最具体），长度相同取最先出现的。无则返回 None。
    """
    if not query:
        return None
    merged = merge_alnum_separators(query)
    candidates = []
    for tok in _MODEL_TOKEN_PATTERN.findall(merged):
        has_digit = any(c.isdigit() for c in tok)
        has_alpha = any(c.isalpha() for c in tok)
        if has_digit and has_alpha:
            candidates.append(tok)
    if not candidates:
        return None
    # 最长优先；长度相同保持出现顺序（max 对稳定序列取首个最大）
    return max(candidates, key=len)

def split_num_alpha(s: str) -> List[str]:
    """拆分数字+字母组合（如85U8N → 85、U8N）"""
    pattern = r'^(\d+)([A-Za-z].*)$'
    match = re.match(pattern, s)
    if match:
        return [match.group(1), match.group(2)]
    return [s]

def clean_and_split(words: List[str]) -> List[str]:
    """清理空值+拆分混合字符串"""
    cleaned = []
    for word in words:
        stripped_word = word.strip()
        if not stripped_word:
            continue
        split_parts = split_num_alpha(stripped_word)
        cleaned.extend(split_parts)
    return cleaned

# -------------------------- 精准匹配核心类 --------------------------
class ChinesePreciseMatching:
    def __init__(self, documents: List[str], k1: float = 0.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        jieba.suggest_freq(("激光", "电视"), True)
        jieba.add_word("激光电视", freq=1000)
        jieba.add_word("强制恢复", freq=1000)
        self.stop_words = set(['的', '了', '是', '在', '和', '有', '我', '也', '很', '就', '/','pdf','vip'])
        self.whitelist = set(['折扣','强制恢复','恢复','制冷','制热','实时','开机','关机','电视','空调','激光电视','进水', '排水', '洗衣机', '洗碗机','油烟机','冰箱','空调','电视','平板电视','洗衣机','冷柜','洗碗机','变温柜','电热水器','燃气灶','投影'])
        self.conflict_map={'空调':['油烟机'],'恢复':['强制恢复']}
        self.productlist = set(['油烟机','冰箱','空调','电视','洗衣机','冷柜','洗碗机','变温柜','电热水器','燃气灶','投影','微波炉'])
        self._preprocess()
        self._calc_avgdl()
        self._calc_idf()
        self.model_blacklist=['vip']
        self.model_cutoff=-0.25



    def _preprocess(self):
        """预处理：分词、过滤、统计文档频率"""
        self.corpus = []
        self.word_count = defaultdict(int)

        for doc in self.documents:
            words = jieba.lcut(merge_alnum_separators(doc), cut_all=True)
            filtered_words = clean_and_split(words)
            final_words = [
                word.lower() for word in filtered_words
                if word not in self.stop_words and (word in self.whitelist or re.search(pattern, word))
            ]
            print(f'final words: {final_words}')
            self.corpus.append(final_words)

            # 更新文档频率
            for word in set(final_words):
                self.word_count[word] += 1

    def _calc_avgdl(self):
        """计算平均文档长度"""
        total_length = sum(len(doc) for doc in self.corpus)
        self.avgdl = total_length / len(self.corpus) if len(self.corpus) > 0 else 0

    def _calc_idf(self):
        """计算IDF值"""
        self.idf = {}
        N = len(self.documents)
        for word, df in self.word_count.items():
            self.idf[word] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def filter(self, query: str) -> float:
        query_product_set = set()
        doc_product_set_list = []
        for i in range(len(self.documents)):
            doc_product_set_list.append(set())

        for word in self.productlist:
            if word in query:
                query_product_set.add(word)
            for i in range(len(self.documents)):
                if word in self.documents[i]:
                    doc_product_set_list[i].add(word)

        result = []

        if len(query_product_set) == 0:
            return [1] * len(self.documents)

        print(f'*** {query_product_set}  \n{self.documents} \n{doc_product_set_list} \n\n{zip(self.documents,doc_product_set_list)}')

        for i in range(len(self.documents)):
            if len(query_product_set) == 0 or len(doc_product_set_list[i])==0 or not query_product_set.isdisjoint(doc_product_set_list[i]):
                result.append(1)
            else:
                result.append(0)
        return result




    def get_score(self, query: str, doc_idx: int, meta_data: dict) -> float:
        """计算单个文档的精准匹配得分"""
        doc = self.corpus[doc_idx]
        doc_length = len(doc)
        model = meta_data['model'] if 'model' in meta_data else None
        print(self.corpus[doc_idx])
        print(f'#### model is {model}#####')
        # 型号匹配：归一化剥离空格/斜杠/连字符等分隔符后再做子串判断，
        # 避免 'E8S Pro' 与 'E8SPro'/'E8S-Pro' 因分隔符不一致而误判为不匹配
        if model:
            model_norm = normalize_model(model)
            doc_norm = normalize_model(self.documents[doc_idx])
            if model_norm not in doc_norm and model_norm not in self.model_blacklist:
                return self.model_cutoff

        # 处理查询词（转小写，与 _preprocess 中文档词的 word.lower() 对齐，否则 idf 命中不了）
        # cut_all=True 与文档侧 _preprocess 的切分模式保持一致，避免两侧 token 集合不对称
        query_words = jieba.lcut(merge_alnum_separators(query), cut_all=True)
        filtered_words = clean_and_split(query_words)
        query_words = [
            word.lower() for word in filtered_words
            if word not in self.stop_words and (word in self.whitelist or re.search(pattern, word))
        ]

        print(f'query words: {query_words}')

        for word in query_words:
            if word in self.conflict_map:
                for conflit in self.conflict_map[word]:
                    if conflit in self.documents[doc_idx]:
                        return self.model_cutoff

        score = 0.0
        for word in query_words:
            if word not in self.idf:
                continue
            tf = doc.count(word)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
            score += self.idf[word] * (tf * (self.k1 + 1)) / denominator
        return score

    def get_normalized_scores(self, query: str, meta_data: dict) -> List[float]:
        """获取所有文档的归一化得分（0-1）"""
        # 计算原始得分

        raw_scores = [self.get_score(query, i, meta_data) for i in range(len(self.documents))]
        #return raw_scores

        # 处理空值和相同值情况
        if not raw_scores:
            return []


        min_score = min(raw_scores)
        if min_score==self.model_cutoff:
            print('EEEEEE####')
            min_score = 0
        max_score = max(raw_scores)
        if max_score==self.model_cutoff:
            max_score =0.1

        # 归一化到0-1
        normalized_scores = []
        if max_score == min_score:
            normalized_scores = [1.0 for _ in raw_scores]
        else:
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in raw_scores]

        return normalized_scores

# -------------------------- 请求模型 --------------------------
class PreciseMatchingRequest(BaseModel):
    """精准匹配检索请求模型"""
    query: str  # 查询语句
    documents: List[str]  # 文档集合
    meta_data: Optional[List[dict]] = None
    k1: float = 0.2  # 精准匹配参数k1
    b: float = 0.75   # 精准匹配参数b

class PreciseMatchingResponse(BaseModel):
    """精准匹配检索响应模型"""
    normalized_scores: List[float]  # 每个文档的归一化得分（0-1）
    weights: Dict[str, float] = {}  # 得分统计信息

# -------------------------- API接口 --------------------------
@app.post("/precise_matching/rank", response_model=PreciseMatchingResponse)
async def precise_matching_rank(request: PreciseMatchingRequest = Body(...)):
    """
    精准匹配文本检索接口
    - 输入：查询语句、文档集合、精准匹配参数
    - 输出：每个文档的归一化得分（0-1）、得分统计信息
    """
    # 初始化精准匹配
    print(f'#############{request.meta_data}')
    content_list=[]
    scores = [1] * len(request.documents)
    if request.meta_data:
        for item in request.meta_data:
            if item['kind'] == 'document':
                filename = item['fileName']
                if filename:
                    content_list.append(item['fileName'])
                else:
                    content_list.append('')
        if request.meta_data[0]['kind'] == 'qna':
            content_list=request.documents

        assert len(content_list) == len(request.documents)

    addr = os.getenv("FILTER_SERVER",None)
    # filter 接口不稳定：调用失败或返回空/无 model 时，降级用规则从 query 提取型号
    ranked_results = {}
    try:
        ranked_results = await AsyncChatCompletionClient(base_url=f"http://{addr}").batch_requests(
                    query = request.query,
                    content_list=[]
                )
        if not isinstance(ranked_results, dict):
            ranked_results = {}
    except Exception as e:
        print(f'filter接口调用失败，降级规则提取型号: {type(e).__name__}: {e}')
        ranked_results = {}

    # 若 filter 未给出有效 model，则按规则从 query 提取
    if not ranked_results.get('model'):
        rule_model = extract_model_by_rule(request.query)
        if rule_model:
            ranked_results['model'] = rule_model
            print(f'规则提取型号: {rule_model}')

    precise_matching = ChinesePreciseMatching(
        documents=content_list,
        k1=request.k1,
        b=request.b
    )
    if request.meta_data:
        scores = precise_matching.filter(request.query)
        print(f'^^^^^^^^ {scores}')
    ### 找到互斥关系

    precise_matching = ChinesePreciseMatching(
        documents=request.documents,
        k1=request.k1,
        b=request.b
    )
    #print(ranked_results)

    # 获取归一化得分
    normalized_scores = precise_matching.get_normalized_scores(request.query, ranked_results)

    # 计算得分统计信息


    return {
        "normalized_scores": [a * b for a, b in zip(normalized_scores, scores)],
        "weights":{"alpha":0.1}
    }



# -------------------------- 启动服务 --------------------------
if __name__ == "__main__":
    import sys
    import uvicorn
    # 端口配置：优先取命令行第一个参数，否则默认8080
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
    # 启动FastAPI服务
    uvicorn.run(app, host="0.0.0.0", port=PORT)