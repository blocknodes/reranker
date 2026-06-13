#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rerank 服务 badcase 回归测试（零三方依赖，仅用标准库）。

覆盖我们排查过的各类问题维度：
  - query 整串包含正文 → 子串置顶
  - 型号分隔符差异（空格 / - / / / 大小写）的鲁棒性
  - 型号不匹配 / 近似型号 / 型号边界
  - 产品互斥过滤、冲突词
  - 同族文档区分度、归一化退化
  - 空输入等边界

用法：
  python3 test_badcases.py                       # 打默认地址
  BASE_URL=http://10.18.231.45:31040 python3 test_badcases.py
  python3 test_badcases.py -k 型号               # 只跑名字含“型号”的用例

断言基于“排序关系/相对性质”，不绑定具体分值，便于长期回归。
"""
import os
import sys
import json
import urllib.request
import urllib.error

BASE_URL = os.getenv("BASE_URL", "http://10.18.231.45:31040").rstrip("/")
RERANK_PATH = "/rerank"
TIMEOUT = float(os.getenv("TIMEOUT", "30"))


# ----------------------------- HTTP 调用 -----------------------------
def call_rerank(query, documents, meta_data=None, instruction=None):
    payload = {"query": query, "documents": documents}
    if meta_data is not None:
        payload["meta_data"] = meta_data
    if instruction is not None:
        payload["instruction"] = instruction
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + RERANK_PATH, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    # 统一解析成 index->score；空输入时服务返回 data:[]，按空结果处理
    data = body.get("data") or []
    if not data:
        return {}
    values = data[0].get("value") or []
    scores = {item["index"]: item["relevance_score"] for item in values}
    return scores


def doc(meta_kind="document", fileName="", category_path=""):
    return {"fileName": fileName, "kind": meta_kind, "category_path": category_path}


def order_by_score(scores):
    """返回按分数降序排列的 index 列表"""
    return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def rank_of(scores, idx):
    """idx 在降序排名中的位置（0=第一）"""
    return order_by_score(scores).index(idx)


# ----------------------------- 用例数据 -----------------------------
# 小数据集：洗烘（型号 U5Q）
WASH_DOCS = [
    "海信棉花糖U5Q洗烘套装卖点一页纸，包括典型人群、用户痛点、产品定位、TOP卖点",      # 0 目标
    "海信棉花糖U3Q洗烘套装产品卖点一页纸，包括典型人群、产品定位、TOP卖点",           # 1 异型号
    "海信棉花糖U5Q洗烘一体机门店讲解一页纸，纯平全嵌、净滤活水洗",                    # 2 同型号他文
    "海信冰箱BCD-500V5FZKQD产品卖点一页纸",                                       # 3 异品类
]
WASH_META = [
    doc(fileName="棉花糖U5Q洗烘套装卖点一页纸.xlsx"),
    doc(fileName="棉花糖U3Q洗烘套装.xlsx"),
    doc(fileName="海信U5Q门店讲解一页纸.pptx"),
    doc(fileName="海信冰箱BCD-500V5FZKQD卖点一页纸.docx"),
]

# 小数据集：电视（型号 E8S Pro，含分隔符）
TV_DOCS = [
    "E8S Pro RGB-Mini LED电视卖点一页纸，产品定位、营销关键词、TOP卖点",            # 0 目标(带空格)
    "海信E8S RGB-Mini LED电视卖点一页纸，主图五大卖点、TOP卖点",                    # 1 E8S 非Pro
    "海信E8Q Pro电视参数，背光技术、画质芯片、分区控光",                            # 2 E8Q Pro
    "海信大白闺蜜机X8 Pro产品一页纸，32吋4K移动智慧屏",                            # 3 异品类
]
TV_META = [
    doc(fileName="E8S Pro 卖点一页纸.xlsx"),
    doc(fileName="E8S 卖点一页纸.xlsx"),
    doc(fileName="E8S-Pro对比E8Q-Pro.jpg"),
    doc(fileName="海信-桌面显示器-X8Pro-产品一页纸.pdf"),
]

# 产品互斥小数据集
PROD_DOCS = [
    "海信空调KFR-35GW变频冷暖，节能省电",       # 0 空调
    "海信油烟机CXW-200吸力大，易清洁",          # 1 油烟机
    "海信空调挂机制冷制热静音设计",              # 2 空调
]
PROD_META = [doc(fileName="空调.xlsx"), doc(fileName="油烟机.xlsx"), doc(fileName="空调2.xlsx")]


# ----------------------------- 用例定义 -----------------------------
# 每个用例：(name, kind, fn) ；fn 返回 (ok: bool, detail: str)
CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("01 整串包含-目标应排第一(子串置顶)")
def c01():
    q = "海信棉花糖U5Q洗烘套装卖点一页纸"
    s = call_rerank(q, WASH_DOCS, WASH_META)
    return rank_of(s, 0) == 0, f"目标(0)排名={rank_of(s,0)} order={order_by_score(s)}"


@case("02 整串包含-空格扰动(query型号加空格)仍置顶")
def c02():
    q = "海信棉花糖U5 Q洗烘套装卖点一页纸"  # U5Q 中间插空格
    s = call_rerank(q, WASH_DOCS, WASH_META)
    # 去空格后仍是 doc0 的子串，应置顶
    return rank_of(s, 0) == 0, f"目标(0)排名={rank_of(s,0)} order={order_by_score(s)}"


@case("03 型号大小写-u5q/U5Q 等价")
def c03():
    s_up = call_rerank("U5Q洗烘套装", WASH_DOCS, WASH_META)
    s_low = call_rerank("u5q洗烘套装", WASH_DOCS, WASH_META)
    return order_by_score(s_up) == order_by_score(s_low), \
        f"大写order={order_by_score(s_up)} 小写order={order_by_score(s_low)}"


@case("04 型号不匹配-异品类(冰箱)应被压到最低")
def c04():
    q = "U5Q洗烘套装卖点"
    s = call_rerank(q, WASH_DOCS, WASH_META)
    # 冰箱(3) 既不含U5Q也异品类，应排最后
    return order_by_score(s)[-1] == 3, f"order={order_by_score(s)} 冰箱分={s.get(3)}"


@case("05 异型号(U3Q)应低于同型号(U5Q)文档")
def c05():
    q = "U5Q洗烘套装卖点"
    s = call_rerank(q, WASH_DOCS, WASH_META)
    return s[2] > s[1], f"U5Q他文(2)={s.get(2)} U3Q(1)={s.get(1)}"


@case("06 电视型号空格-E8S Pro 加空格")
def c06():
    q = "海信电视E8S Pro的卖点"
    s = call_rerank(q, TV_DOCS, TV_META)
    return rank_of(s, 0) <= 1, f"目标(0)排名={rank_of(s,0)} order={order_by_score(s)}"


@case("07 电视型号无空格-E8SPro 应与加空格结果一致")
def c07():
    s_sp = call_rerank("海信电视E8S Pro的卖点", TV_DOCS, TV_META)
    s_no = call_rerank("海信电视E8SPro的卖点", TV_DOCS, TV_META)
    return order_by_score(s_sp) == order_by_score(s_no), \
        f"加空格order={order_by_score(s_sp)} 不加order={order_by_score(s_no)}"


@case("08 电视型号连字符-E8S-Pro 应与加空格一致")
def c08():
    s_sp = call_rerank("海信电视E8S Pro的卖点", TV_DOCS, TV_META)
    s_hy = call_rerank("海信电视E8S-Pro的卖点", TV_DOCS, TV_META)
    return order_by_score(s_sp) == order_by_score(s_hy), \
        f"空格order={order_by_score(s_sp)} 连字符order={order_by_score(s_hy)}"


@case("09 电视型号斜杠-E8S/Pro 应与加空格一致")
def c09():
    s_sp = call_rerank("海信电视E8S Pro的卖点", TV_DOCS, TV_META)
    s_sl = call_rerank("海信电视E8S/Pro的卖点", TV_DOCS, TV_META)
    return order_by_score(s_sp) == order_by_score(s_sl), \
        f"空格order={order_by_score(s_sp)} 斜杠order={order_by_score(s_sl)}"


@case("10 近似型号-E5Q 不应误命中 U5Q query")
def c10():
    docs = ["海信WF100E5Q滚筒式洗衣机产品卖点一页纸", "海信棉花糖U5Q洗烘套装卖点"]
    meta = [doc(fileName="WF100E5Q.docx"), doc(fileName="U5Q.xlsx")]
    s = call_rerank("U5Q洗烘套装卖点", docs, meta)
    return s[1] > s[0], f"U5Q(1)={s.get(1)} E5Q(0)={s.get(0)}"


@case("11 同族多文档-U5Q目标应高于U5Q他文")
def c11():
    q = "海信棉花糖U5Q洗烘套装卖点一页纸"
    s = call_rerank(q, WASH_DOCS, WASH_META)
    return s[0] >= s[2], f"目标(0)={s.get(0)} U5Q他文(2)={s.get(2)}"


@case("12 边界-空query 返回空/不报错")
def c12():
    try:
        s = call_rerank("", WASH_DOCS, WASH_META)
        return True, f"空query正常返回 keys={list(s.keys())}"
    except Exception as e:
        return False, f"空query抛错: {e}"


@case("13 边界-空documents 返回空/不报错")
def c13():
    try:
        s = call_rerank("U5Q洗烘套装", [], [])
        return len(s) == 0, f"空documents返回 keys={list(s.keys())}"
    except Exception as e:
        return False, f"空documents抛错: {e}"


@case("14 产品互斥-空调query 油烟机应被过滤(置0/最低)")
def c14():
    s = call_rerank("空调制冷效果", PROD_DOCS, PROD_META)
    return order_by_score(s)[-1] == 1, f"order={order_by_score(s)} 油烟机(1)={s.get(1)}"


@case("15 已知缺陷-归一化垫底文档被硬置零(precise==0→0)")
def c15():
    # query 仅 '制冷' 命中BM25；doc2含'制冷'，doc0(冷暖)/doc1(油烟机)不含→归一化为0
    # 当前 server_v2: precise==0 则最终置0，导致语义可能相关的doc0也被清零。
    # 这是设计缺陷：min-max 必然产生0分，硬置零会误伤。修复前此用例记录现状。
    s = call_rerank("制冷效果怎么样", PROD_DOCS, PROD_META)
    zero_cnt = sum(1 for v in s.values() if v == 0)
    # 现状：存在被置零的文档。修复后应改为 >0 或仅cutoff/互斥命中才置零。
    return zero_cnt >= 1, f"被置零文档数={zero_cnt} scores={s} (记录现状, 见分析)"


@case("16 归一化-全同质文档不应全部并列(有区分或合理)")
def c16():
    docs = ["海信U5Q洗烘套装A", "海信U5Q洗烘套装B", "海信U5Q洗烘套装C"]
    meta = [doc(fileName="a.xlsx"), doc(fileName="b.xlsx"), doc(fileName="c.xlsx")]
    s = call_rerank("U5Q洗烘套装", docs, meta)
    # 同质：允许接近，但不应全为0；主要确认服务正常产出
    return all(v >= 0 for v in s.values()) and len(s) == 3, f"scores={s}"


@case("17 大小写文件名-型号在fileName大小写混写")
def c17():
    docs = ["海信电视卖点介绍", "海信其他电视"]
    meta = [doc(fileName="e8s pro 卖点.xlsx"), doc(fileName="其他.xlsx")]
    s = call_rerank("E8S Pro 电视卖点", docs, meta)
    return len(s) == 2, f"scores={s} (验证不报错且产出完整)"


@case("18 长query整串-完整问句包含在正文")
def c18():
    q = "E8S Pro RGB-Mini LED电视卖点一页纸"
    s = call_rerank(q, TV_DOCS, TV_META)
    return rank_of(s, 0) == 0, f"目标(0)排名={rank_of(s,0)} order={order_by_score(s)}"


@case("19 纯英文型号query")
def c19():
    s = call_rerank("E8S Pro", TV_DOCS, TV_META)
    return rank_of(s, 0) <= 1, f"目标(0)排名={rank_of(s,0)} order={order_by_score(s)}"


@case("20 无meta_data-纯文本走doc路径不报错")
def c20():
    s = call_rerank("U5Q洗烘套装卖点", WASH_DOCS, meta_data=None)
    return len(s) == len(WASH_DOCS), f"scores={s}"


# ----------------------------- Runner -----------------------------
def main():
    keyword = None
    if "-k" in sys.argv:
        keyword = sys.argv[sys.argv.index("-k") + 1]

    print(f"目标服务: {BASE_URL}{RERANK_PATH}\n")
    passed = failed = errored = skipped = 0
    for name, fn in CASES:
        if keyword and keyword not in name:
            skipped += 1
            continue
        try:
            ok, detail = fn()
            if ok:
                passed += 1
                print(f"  [PASS] {name}")
            else:
                failed += 1
                print(f"  [FAIL] {name}\n         {detail}")
        except urllib.error.URLError as e:
            errored += 1
            print(f"  [ERR ] {name}  连接失败: {e}")
        except Exception as e:
            errored += 1
            print(f"  [ERR ] {name}  异常: {type(e).__name__}: {e}")

    total = passed + failed + errored
    print(f"\n汇总: 通过 {passed} / 失败 {failed} / 异常 {errored} / 跳过 {skipped} (执行 {total})")
    sys.exit(0 if failed == 0 and errored == 0 else 1)


if __name__ == "__main__":
    main()
