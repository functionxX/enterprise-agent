"""
ragas_eval.py — RAGAS 自动化评估（LLM-as-judge 指标）

用法（项目根目录，需先 docker compose up + 种子数据 + 知识库文档）：
    python -m venv evaluation/.venv                         # 评估独立 venv（与运行时隔离，见 requirements.txt）
    PYTHONUTF8=1 evaluation/.venv/Scripts/python -m pip install -r evaluation/requirements.txt
    PYTHONUTF8=1 evaluation/.venv/Scripts/python evaluation/ragas_eval.py            # 全量 17 条
    PYTHONUTF8=1 evaluation/.venv/Scripts/python evaluation/ragas_eval.py --limit 3  # 只跑前 3 条
    # --api-url http://localhost:8000（默认）

【与 evaluate.py 的互补定位（面试点：Agent 怎么评估？）】
- evaluate.py 是"规则化指标"：工具调用准确率 / 关键信息覆盖 / 幻觉检测。
  确定性 100%，跑 10 次结果一样——但只能测"测得到的东西"
  （调了哪个工具、关键词有没有出现）。
- ragas_eval.py 是"LLM-as-judge 指标"：faithfulness（事实一致性）、
  answer_relevancy（回答相关性）。回答"质量"的主观维度无法用规则穷举，
  交给评分 LLM 判断。两者的关系不是替代，是互补：
  规则化指标是骨架（可复现），LLM-as-judge 是血肉（衡量真实质量）。

【为什么只选这两个指标？（诚实边界）】
RAGAS 经典四件套里：
- faithfulness：answer 是否忠实于 contexts——不需要人工参考答案，直接可用；
- answer_relevancy：answer 是否切题——同上，不需要参考答案；
- context_precision / context_recall：需要"完整人工参考答案"与逐段标注。
  本项目的 ground_truth.json 只标注了关键信息点（expected_answer_contains
  关键词），没有完整参考答案——硬上这两个指标会变成虚假标注。
  诚实做法：只跑有真实数据支撑的指标。

【上下文从哪来？（面试点：评估评的是真实调用链）】
faithfulness 检查"回答有没有忠实于依据"——依据必须是 Agent 实际用过的
上下文，而不是理想化的参考答案。API 的 include_contexts=true 返回
tool_results 原文（SQL 查询 + 返回行、检索 chunk 全文），
RAGAS 对着这份真实上下文打分。

【评分 LLM 与 embedding 配置（环境变量）】
- 评分 LLM：DeepSeek（OpenAI 兼容）。EVAL_LLM_API_KEY / EVAL_LLM_BASE_URL /
  EVAL_LLM_MODEL（默认 deepseek-chat）。评分温度必须 0——评委要确定性。
- answer_relevancy 需要 embedding 算相似度：DeepSeek 不提供 embedding，
  用硅基流动 BGE（EVAL_EMBEDDING_API_KEY / EVAL_EMBEDDING_BASE_URL /
  EVAL_EMBEDDING_MODEL，默认 BAAI/bge-large-zh-v1.5），与线上 RAG 同家，
  风格一致。
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
DEFAULT_API_URL = "http://localhost:8000"


def _env_keys() -> dict[str, str]:
    """
    读取项目根 .env 的 key（评分 LLM/embedding 默认复用 Agent 的供应商配置）。

    【为什么不直接引 backend/app/config.py？】
    1. 评估 venv 独立——不保证装了 pydantic-settings，最小化依赖；
    2. 只需两个 key 的读取，一个 10 行的简单解析比引配置中心更轻；
    3. 面试点：评估是"旁路"，不该被运行时配置中心绑架。
    简单 key=value 解析即可（注释行跳过）；.env 不存在返回空 dict。
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return {}
    keys: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip()
    return keys


def _score_llm():
    """评分 LLM：DeepSeek。评委要确定性——temperature 固定 0。

    key 来源：EVAL_LLM_API_KEY 环境变量优先，其次项目 .env 的 LLM_API_KEY
    （与 Agent 同一个 DeepSeek 账号，无需额外配置）。
    """
    from langchain_openai import ChatOpenAI
    keys = _env_keys()
    return ChatOpenAI(
        model=os.getenv("EVAL_LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("EVAL_LLM_API_KEY") or keys.get("LLM_API_KEY", ""),
        base_url=os.getenv("EVAL_LLM_BASE_URL") or keys.get("LLM_BASE_URL", "https://api.deepseek.com"),
        temperature=0.0,
        max_retries=1,
    )


def _score_embeddings():
    """answer_relevancy 需要 embedding：硅基流动 BGE（DeepSeek 不提供）。

    key 来源：EVAL_EMBEDDING_API_KEY 环境变量优先，其次项目 .env 的
    EMBEDDING_API_KEY（与线上 RAG 同一家供应商，风格一致）。

    check_embedding_ctx_length=False：langchain 默认 tiktoken 切分路径会把
    input 发成 token ID 数组——OpenAI 官方 API 接受这种格式，但硅基流动 BGE
    只接受原始文本（实测 400 code 20015）。评估文本远短于 8191 token
    上下文，跳过切分零损失。
    """
    from langchain_openai import OpenAIEmbeddings
    keys = _env_keys()
    return OpenAIEmbeddings(
        model=os.getenv("EVAL_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
        api_key=os.getenv("EVAL_EMBEDDING_API_KEY") or keys.get("EMBEDDING_API_KEY", ""),
        base_url=os.getenv("EVAL_EMBEDDING_BASE_URL")
        or keys.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
        check_embedding_ctx_length=False,
    )


async def collect_cases(api_url: str, limit: int | None) -> list[dict]:
    """跑每条 ground truth 用例，收集 {question, answer, contexts} 三元组。

    include_contexts=true 让 API 返回 Agent 实际依据的上下文原文——
    faithfulness 评的是真实调用链，不是理想化参考答案。
    """
    cases = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["cases"]
    if limit:
        cases = cases[:limit]

    rows = []
    async with httpx.AsyncClient() as client:
        for i, case in enumerate(cases, 1):
            resp = await client.post(
                f"{api_url}/api/v1/agent/chat",
                json={"query": case["query"], "include_contexts": True},
                timeout=180.0,  # Agent 单请求可能 8~12 次 LLM 调用
            )
            resp.raise_for_status()
            data = resp.json()
            rows.append({
                "case_id": case["id"],
                "difficulty": case["difficulty"],
                "question": case["query"],
                "answer": data.get("answer", "") or "（Agent 未生成回答）",
                "contexts": data.get("contexts", []),
            })
            print(f"[{i}/{len(cases)}] {case['id']} ({case['difficulty']}) "
                  f"contexts={len(rows[-1]['contexts'])}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 自动化评估（LLM-as-judge）")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Agent API 地址")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条用例")
    args = parser.parse_args()

    rows = asyncio.run(collect_cases(args.api_url, args.limit))
    if not rows:
        raise SystemExit("没有可用用例")

    # ---- 跑 RAGAS 指标（延迟导入：评估依赖与运行时依赖分离）----
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"] or ["（该用例 Agent 未检索到任何上下文）"],
        }
        for r in rows
    ])
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=_score_llm(),
        embeddings=_score_embeddings(),
    )

    df = result.to_pandas()
    per_case = []
    for idx, row in df.iterrows():
        r = rows[idx]
        per_case.append({
            "case_id": r["case_id"],
            "difficulty": r["difficulty"],
            "faithfulness": round(float(row["faithfulness"]), 4),
            "answer_relevancy": round(float(row["answer_relevancy"]), 4),
        })

    # ---- 汇总与输出 ----
    print("\n" + "=" * 60)
    print("RAGAS 评分（0~1，越高越好；评委 = DeepSeek，temperature=0）")
    for pc in per_case:
        print(f"  {pc['case_id']} ({pc['difficulty']:<6}) "
              f"faithfulness={pc['faithfulness']:.2f} "
              f"answer_relevancy={pc['answer_relevancy']:.2f}")
    avg_f = sum(pc["faithfulness"] for pc in per_case) / len(per_case)
    avg_r = sum(pc["answer_relevancy"] for pc in per_case) / len(per_case)
    print(f"平均 faithfulness={avg_f:.3f}  answer_relevancy={avg_r:.3f}")
    print("=" * 60)

    out_path = Path(__file__).parent / "ragas_report.json"
    out_path.write_text(json.dumps({
        "metrics": ["faithfulness", "answer_relevancy"],
        "scorer": {"llm": "deepseek-chat", "temperature": 0.0,
                   "embeddings": os.getenv("EVAL_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")},
        "avg": {"faithfulness": round(avg_f, 4), "answer_relevancy": round(avg_r, 4)},
        "cases": per_case,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"详细报告已写入：{out_path}")


if __name__ == "__main__":
    main()
