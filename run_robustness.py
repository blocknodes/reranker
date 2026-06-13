#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 robustness_cases.jsonl，逐条打 rerank 服务，按断言类型校验并打印结果。

断言类型 (expect.type)：
  top        : 指定 index 必须排第一                 {index}
  top_in     : 第一名必须落在 indices 集合内           {indices}
  last       : 指定 index 必须排最后                   {index}
  gt         : score[a] > score[b]                     {a, b}
  complete   : 返回结果数量等于 n（不报错、产出完整）   {n}
  empty      : 空输入应返回空结果                       {}
  invariant  : 原query与若干变体写法的排序order完全一致  {queries:[...]}

用法：
  python3 run_robustness.py
  BASE_URL=http://10.18.231.45:31040 python3 run_robustness.py
  python3 run_robustness.py --file robustness_cases.jsonl
  python3 run_robustness.py --dim 分隔符        # 仅跑 dim 含该关键字的用例
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error

BASE_URL = os.getenv("BASE_URL", "http://10.18.231.45:31040").rstrip("/")
RERANK_PATH = "/rerank"
TIMEOUT = float(os.getenv("TIMEOUT", "30"))


def call_rerank(query, documents, meta_data=None):
    payload = {"query": query, "documents": documents}
    if meta_data is not None:
        payload["meta_data"] = meta_data
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + RERANK_PATH, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    data_field = body.get("data") or []
    if not data_field:
        return {}
    values = data_field[0].get("value") or []
    return {item["index"]: item["relevance_score"] for item in values}


def order_by_score(scores):
    return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _fmt(scores):
    """紧凑打印各index分数，按index升序，保留6位"""
    return "{" + ", ".join(f"{i}:{scores[i]:.6f}" for i in sorted(scores)) + "}"


def check(case):
    """返回 (ok, detail)"""
    q = case["query"]
    docs = case["documents"]
    meta = case.get("meta_data", None)
    exp = case["expect"]
    t = exp["type"]

    if t == "empty":
        s = call_rerank(q, docs, meta)
        return len(s) == 0, f"返回{len(s)}条 (期望空)"

    if t == "invariant":
        base = call_rerank(q, docs, meta)
        base_order = order_by_score(base)
        details = [f"base={base_order}{_fmt(base)}"]
        ok = True
        for vq in exp["queries"]:
            vs = call_rerank(vq, docs, meta)
            vo = order_by_score(vs)
            same = (vo == base_order)
            ok = ok and same
            mark = '=' if same else '≠'
            # 不一致时附带分数，便于判断是否近似并列
            extra = "" if same else _fmt(vs)
            details.append(f"{mark}[{vq}]→{vo}{extra}")
        return ok, " ".join(details)

    # 其余类型先取一次分数
    s = call_rerank(q, docs, meta)
    order = order_by_score(s)

    if t == "top":
        return (order and order[0] == exp["index"]), f"order={order} 期望首位={exp['index']}"
    if t == "top_in":
        return (order and order[0] in exp["indices"]), f"order={order} 期望首位∈{exp['indices']}"
    if t == "last":
        return (order and order[-1] == exp["index"]), f"order={order} 期望末位={exp['index']}"
    if t == "gt":
        a, b = exp["a"], exp["b"]
        return s.get(a, 0) > s.get(b, 0), f"score[{a}]={s.get(a)} > score[{b}]={s.get(b)} order={order}"
    if t == "complete":
        return len(s) == exp["n"], f"返回{len(s)}条 期望={exp['n']}"

    return False, f"未知断言类型: {t}"


def main():
    path = "robustness_cases.jsonl"
    dim_filter = None
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
    if "--dim" in sys.argv:
        dim_filter = sys.argv[sys.argv.index("--dim") + 1]

    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"目标服务: {BASE_URL}{RERANK_PATH}")
    print(f"用例文件: {path}  (共 {len(cases)} 条)\n")
    print(f"{'ID':>3} {'结果':<6} {'维度':<22} 详情")
    print("-" * 100)

    passed = failed = errored = skipped = 0
    fails = []
    for case in cases:
        cid, dim = case["id"], case["dim"]
        if dim_filter and dim_filter not in dim:
            skipped += 1
            continue
        try:
            ok, detail = check(case)
            if ok:
                passed += 1
                tag = "PASS"
            else:
                failed += 1
                tag = "FAIL"
                fails.append((cid, dim, detail))
            print(f"{cid:>3} [{tag}] {dim:<22} {detail}")
        except urllib.error.URLError as e:
            errored += 1
            print(f"{cid:>3} [ERR ] {dim:<22} 连接失败: {e}")
        except Exception as e:
            errored += 1
            print(f"{cid:>3} [ERR ] {dim:<22} {type(e).__name__}: {e}")

    print("-" * 100)
    total = passed + failed + errored
    print(f"\n汇总: 通过 {passed} / 失败 {failed} / 异常 {errored} / 跳过 {skipped}  (执行 {total})")
    if fails:
        print("\n失败用例:")
        for cid, dim, detail in fails:
            print(f"  #{cid} {dim}: {detail}")
    sys.exit(0 if failed == 0 and errored == 0 else 1)


if __name__ == "__main__":
    main()
