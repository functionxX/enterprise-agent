"""
evaluate.py — Agent 评估脚本（三维度评分）

用法（项目根目录）：
    python evaluation/evaluate.py                          # 默认 http://localhost:8000
    python evaluation/evaluate.py --api-url http://localhost:8000 --limit 5

前置条件：
    1. docker compose up（或本地启动 app + PostgreSQL + Redis）
    2. 已执行种子数据：python -m app.database.seed（backend 目录下）
    3. 已上传知识库文档（POST /api/v1/documents/upload），
       或未配置 embedding key 时自动走 mock 检索降级

【评估维度（设计决策 6，面试高频：Agent 怎么评估？）】

维度 1：Tool 调用准确率（刚性，0~1）
    只检查 Agent 是否调用了预期的 tool 名称，不匹配 SQL 字符串——
    Agent 用 JOIN 还是子查询得到相同正确结果都是合法实现。
    评估的是"工具选择"，不是"实现方式"。
    多调了预期外的 tool 不扣分（Agent 自主决策的合法行为），
    漏调预期 tool 按比例扣分。

维度 2：答案关键信息覆盖（柔性，0~1）
    预期关键词有多少出现在最终回答中。
    关键词不是"标准答案"——LLM 措辞千变万化，逐字匹配没有意义；
    关键词代表"回答必须触及的信息点"。

维度 3：幻觉检测（否定性，pass/fail）
    回答中是否出现种子数据里不存在的供应商名。
    怎么知道"哪些名字不存在"？种子数据是固定种子（SEED=42）生成的，
    评估脚本用同一个算法重建合法名单（见 _KNOWN_SUPPLIER_NAMES），
    回答中出现名单外的"XX公司/XX科技"样式的名称即判定幻觉。

【最终指标：任务完成率】
    task_completion_rate = 三个维度全部达标的用例数 / 总用例数
    这是 portfolio 里最有说服力的数字："15 个采购分析任务，Agent 完成率 93%"。

【面试追问："为什么不用 LLM-as-judge 评分？"】
    答：LLM 评委本身有波动性和成本，评估结果不可复现；
    规则化指标（tool 匹配/关键词/幻觉名单）确定性 100%，
    跑 10 次结果一样——评估体系的第一性要求是可复现。
    LLM-as-judge 适合作为补充（对答案质量打分），不适合作为唯一标准。
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

# 复用种子数据的名称池：保证"已知供应商名单"与 seed.py 永远一致
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.database.seed import (  # noqa: E402
    INDUSTRIES,
    NAME_PREFIXES,
    NAME_SUFFIXES,
    ANOMALY_SUPPLIER_DELIVERY,
    ANOMALY_SUPPLIER_GROWTH,
)
import random  # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
DEFAULT_API_URL = "http://localhost:8000"


def _known_supplier_names() -> set[str]:
    """用与 seed.py 相同的算法重建合法供应商名单（SEED=42 可复现）。"""
    rng = random.Random(42)
    names: set[str] = {ANOMALY_SUPPLIER_DELIVERY, ANOMALY_SUPPLIER_GROWTH}
    while len(names) < 50:
        names.add(rng.choice(NAME_PREFIXES) + rng.choice(NAME_SUFFIXES))
    return names


KNOWN_SUPPLIERS = _known_supplier_names()

# 幻觉检测的已知词集合 = 供应商名 + 行业名 + 行业核心词。
# 【评估实测踩坑】直接把答案跑正则会把"能源材料""包装材料"（行业名）
# 误判为幻觉供应商，还会贪婪吞掉上下文（"的管控标准或华芯半导体"）。
# 正确做法：先把所有已知词从文本中移除，剩余文本上再找未知公司名。
KNOWN_TERMS: set[str] = set(KNOWN_SUPPLIERS) | set(INDUSTRIES) | {
    # 行业核心词（"化工产品"→"化工"）：答案常以简称引用行业，也须豁免
    "化工", "材料", "包装", "办公", "自动化", "汽车零部件", "电子元器件",
    # 普通商务短语前缀：移除后"行业供应链""触发供应链风险"这类 prose
    # 不会匹配公司名正则（"供应链"作为后缀出现在动词短语里是普通名词，不是公司）
    "行业", "企业", "整体", "风险", "触发", "防范", "评估", "核查", "排查",
    "启动", "应对", "关注", "保障", "恢复", "维护", "加强",
    # 行业弱后缀词本身（单字+这些词是普通名词：能源、供应链、材料、机械……）
    # 长词先移除：这些词在"能源材料""华芯供应链"被移除后才作为残余普通词清除
    "能源", "供应链", "电子", "机械", "装备", "科技",
}

# 幻觉检测：剩余文本中出现"XX公司/XX集团/XX实业/XX半导体/XX制造"样式 → 嫌疑幻觉。
# 【为什么收紧到公司名强后缀】行业弱后缀（材料/化工/能源/机械/电子）在自由文本里
# 大量以普通名词出现（"存在供应链风险"、"此类不一致可能源<于>——能源跨词误配"），
# 正则匹配会无穷尽地产生假阳性。公司名强后缀（公司/集团/实业/半导体/制造）
# 在正常商务文本中几乎只以公司名形式出现，误报率大幅降低。
# 代价是漏报（编造的"XX材料"公司检测不到）——检测器宁可漏报不可误报，
# 误报会把正确答案判 FAIL，破坏任务完成率指标的可信度。
_COMPANY_PATTERN = re.compile(r"[一-龥]{2,8}(?:公司|集团|实业|半导体|制造|供应链|通讯|自动化)")


def score_tool_selection(expected: list[str], actual: list[str]) -> float:
    """维度 1：预期 tool 的召回率。多调不扣分，漏调按比例扣。"""
    if not expected:
        return 1.0
    hits = sum(1 for tool in set(expected) if tool in set(actual))
    return hits / len(set(expected))


def score_answer_coverage(expected_keywords: list[str], answer: str) -> float:
    """维度 2：预期关键词命中率。"""
    if not expected_keywords:
        return 1.0
    hits = sum(1 for kw in expected_keywords if kw in answer)
    return hits / len(expected_keywords)


def detect_hallucinations(answer: str) -> list[str]:
    """
    维度 3：回答中出现未知供应商名 → 幻觉嫌疑列表。

    【为什么先移除已知词再匹配（而不是直接正则整段文本）】
    直接匹配的两种误报：
    1. 行业名被当成供应商（"能源材料"是行业不是公司）；
    2. 正则贪婪吞掉上下文（"的管控标准或华芯半导体"——华芯是已知供应商，
       但整串不匹配名单）。
    先按长度降序移除所有已知词（长词优先，防"华芯半导体"被"华芯"级碎片破坏），
    剩余文本中仍出现公司名样式 → 才是真幻觉嫌疑。
    检测器宁可宽松（漏报）也不该误报——误报会把正确答案判 FAIL。
    """
    residual = answer
    for term in sorted(KNOWN_TERMS, key=len, reverse=True):
        residual = residual.replace(term, " ")
    return list(set(_COMPANY_PATTERN.findall(residual)))


async def run_case(client: httpx.AsyncClient, case: dict, api_url: str) -> dict:
    """跑单条用例，返回各维度得分与原始响应。"""
    try:
        resp = await client.post(
            f"{api_url}/api/v1/agent/chat",
            json={"query": case["query"]},
            timeout=180.0,  # Agent 单请求可能 8~12 次 LLM 调用，超时要给足
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 —— 单条失败不影响整体评估
        return {
            "id": case["id"], "query": case["query"], "difficulty": case["difficulty"],
            "error": str(exc),
            "tool_score": 0.0, "coverage_score": 0.0, "hallucinations": [],
        }

    answer = data.get("answer", "")
    tools_used = data.get("tools_used", [])

    tool_score = score_tool_selection(case["expected_tools"], tools_used)
    coverage_score = score_answer_coverage(case["expected_answer_contains"], answer)
    hallucinations = detect_hallucinations(answer)

    return {
        "id": case["id"],
        "query": case["query"],
        "difficulty": case["difficulty"],
        "tool_score": round(tool_score, 2),
        "coverage_score": round(coverage_score, 2),
        "hallucinations": hallucinations,
        "tools_used": tools_used,
        "iterations": data.get("iterations", 0),
        "token_usage": data.get("token_usage", {}),
        "answer_preview": answer[:200],
    }


def is_case_passed(result: dict, case: dict) -> bool:
    """任务完成判定：tool 全命中 + 关键信息覆盖率 ≥ 0.6 + 无幻觉。"""
    coverage_threshold = 0.6
    return (
        "error" not in result
        and result["tool_score"] >= 1.0
        and result["coverage_score"] >= coverage_threshold
        and not result["hallucinations"]
    )


async def main(api_url: str, limit: int | None) -> None:
    cases = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["cases"]
    if limit:
        cases = cases[:limit]

    print(f"评估 {len(cases)} 条用例，目标 API：{api_url}\n")
    async with httpx.AsyncClient() as client:
        results = []
        for i, case in enumerate(cases, 1):
            result = await run_case(client, case, api_url)
            results.append((case, result))
            status = "PASS" if is_case_passed(result, case) else "FAIL"
            if "error" in result:
                status = "ERROR"
            print(
                f"[{i}/{len(cases)}] {status} {case['id']} ({case['difficulty']}) "
                f"tool={result.get('tool_score', 0):.2f} coverage={result.get('coverage_score', 0):.2f} "
                f"幻觉={result.get('hallucinations', []) or '无'}"
            )

    # ---- 汇总 ----
    passed = sum(1 for case, result in results if is_case_passed(result, case))
    total = len(results)
    print("\n" + "=" * 60)
    print(f"任务完成率：{passed}/{total} = {passed / total:.1%}")

    by_difficulty: dict[str, list] = {}
    for case, result in results:
        by_difficulty.setdefault(case["difficulty"], []).append(
            is_case_passed(result, case)
        )
    for diff in sorted(by_difficulty):
        values = by_difficulty[diff]
        print(f"  {diff:<8}: {sum(values)}/{len(values)} 通过")

    total_tokens = sum(
        (result.get("token_usage") or {}).get("total", 0)
        for _, result in results
    )
    print(f"总 token 消耗：{total_tokens}")
    print("=" * 60)

    # ---- 输出详细结果 JSON（CI 可消费）----
    out_path = Path(__file__).parent / "evaluation_report.json"
    out_path.write_text(json.dumps(
        [dict(case, **result) for case, result in results],
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(f"详细报告已写入：{out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 评估脚本")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Agent API 地址")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条用例")
    args = parser.parse_args()
    asyncio.run(main(args.api_url, args.limit))
