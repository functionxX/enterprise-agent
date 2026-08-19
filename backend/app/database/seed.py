"""
seed.py — 种子数据生成脚本（50 供应商 + 200 订单 + 30 发票）

用法（backend 目录下）：
    python -m app.database.seed

【种子数据的设计原则（面试高频追问：你的假数据"假"在哪？）】

1. 风险等级不是均匀分布，而是 65/25/10 金字塔：
   现实企业的供应商是被主动筛选过的——长期 high risk 的供应商早就被换掉，
   剩下的 high risk 通常是不可替代的战略供应商（技术垄断/关键原材料）。
   均匀分布 33/33/33 一看就是不懂业务的人造数据。

2. 订单服从帕累托分布（二八定律）：
   前 10 大供应商占约 65% 订单量，而不是 200 条订单平均撒给 50 家。

3. 故意埋了"值得 Agent 发现的异常"（与 evaluation/ground_truth.json 联动）：
   a. 恒达精密机械：近 3 个月交付率骤降（订单大量 cancelled），且 2 张发票逾期；
   b. 华芯半导体：high risk 供应商，近 2 个月订单金额反而暴增 200%；
   c. 电子元器件行业整体交付延迟率偏高。
   没有异常的数据 = 没有分析价值的白噪音。评估集里的"发现异常"类题目
   依赖这些埋点——面试官问"评估怎么测"时，答案在 seed 数据里。

4. 固定随机种子（SEED=42）：数据可复现。
   评估集的 ground truth 与埋点异常一一对应；换机器重跑结果一致。
"""

import asyncio
import random
from datetime import date, timedelta

from sqlalchemy import func, select

from app.config import get_settings
from app.database.connection import dispose_engines, init_engines, session_scope
from app.database.models import Base, Invoice, PurchaseOrder, Supplier

SEED = 42  # 固定种子 → 可复现（评估集依赖这一点）

# 供应商名称池：行业前缀 + 公司后缀，组合出 50 个看起来真实的中文企业名
NAME_PREFIXES = [
    "恒达", "华芯", "中科", "蓝天", "金鼎", "瑞丰", "东岳", "南天", "北航",
    "星辰", "凯盛", "宏图", "天成", "海纳", "云集", "龙腾", "凤凰", "山峰",
    "长江", "黄河", "紫光", "银海", "金桥", "百川", "千里", "万里", "九洲",
    "四方", "三环", "双星", "万向", "亿达", "兆丰", "光启", "明德", "正泰",
    "新希望", "老凤祥", "大白鲨", "小天鹅", "联合", "环球", "宇宙", "时空",
    "数字", "量子", "智能", "精密", "锐意", "进取",
]
NAME_SUFFIXES = [
    "科技", "电子", "机械", "材料", "化工", "能源", "装备", "精密制造",
    "供应链", "实业", "工业", "半导体", "软件", "网络", "通讯", "自动化",
]
INDUSTRIES = [
    "电子元器件", "机械设备", "原材料", "化工产品", "包装材料",
    "办公用品", "工业软件", "自动化设备", "能源材料", "汽车零部件",
]
COUNTRIES = ["中国", "德国", "日本", "韩国", "美国", "新加坡", "马来西亚", "越南"]
PRODUCTS_BY_CATEGORY = {
    "电子元器件": ["贴片电容", "功率芯片", "PCB 板", "连接器", "传感器"],
    "机械设备": ["伺服电机", "液压泵", "减速机", "数控刀具", "轴承"],
    "原材料": ["特种钢材", "铝合金锭", "铜箔", "工程塑料", "稀土材料"],
    "化工产品": ["工业润滑油", "清洗剂", "密封胶", "防锈剂", "涂料"],
    "包装材料": ["瓦楞纸箱", "防静电袋", "托盘", "缠绕膜", "木箱"],
    "办公用品": ["打印纸", "硒鼓", "办公桌椅", "文件柜", "白板"],
    "工业软件": ["ERP 授权", "MES 模块", "CAD 许可", "工控组态软件", "数据采集系统"],
    "自动化设备": ["PLC 控制器", "工业机器人", "视觉检测系统", "传送带", "AGV 小车"],
    "能源材料": ["光伏组件", "锂电池模组", "储能逆变器", "变压器", "电缆"],
    "汽车零部件": ["车用芯片", "铝合金轮毂", "减震器", "车灯模组", "内饰件"],
}
# 订单金额范围（按行业，单位元）
AMOUNT_RANGE_BY_CATEGORY = {
    "电子元器件": (5_000, 200_000),
    "机械设备": (20_000, 500_000),
    "原材料": (10_000, 800_000),
    "化工产品": (8_000, 300_000),
    "包装材料": (2_000, 80_000),
    "办公用品": (500, 30_000),
    "工业软件": (50_000, 1_000_000),
    "自动化设备": (30_000, 600_000),
    "能源材料": (100_000, 2_000_000),
    "汽车零部件": (10_000, 400_000),
}

# ---- 埋点异常（评估集依赖，改动需同步 evaluation/ground_truth.json）----
ANOMALY_SUPPLIER_DELIVERY = "恒达精密制造"   # 近 3 个月交付率骤降 + 发票逾期
ANOMALY_SUPPLIER_GROWTH = "华芯半导体"        # high risk + 订单金额暴增
ANOMALY_INDUSTRY = "电子元器件"               # 行业整体交付延迟偏高


def _generate_supplier_names(rng: random.Random) -> list[str]:
    """生成 50 个不重复的中文供应商名（含 2 个埋点供应商）。"""
    names: set[str] = set()
    # 埋点供应商优先占位——评估集按名字查它们
    names.add(ANOMALY_SUPPLIER_DELIVERY)
    names.add(ANOMALY_SUPPLIER_GROWTH)
    while len(names) < 50:
        names.add(rng.choice(NAME_PREFIXES) + rng.choice(NAME_SUFFIXES))
    return sorted(names)


def _risk_level_for(index: int, total: int) -> str:
    """按金字塔比例分配风险等级：前 65% low，中间 25% medium，后 10% high。

    【为什么 high 集中在少数行业？】
    现实里 high risk 供应商通常是不可替代的战略供应商——芯片、特种材料等。
    所以排序后把 high 分配给特定行业（见下），而不是随机撒。
    """
    ratio = index / total
    if ratio < 0.65:
        return "low"
    if ratio < 0.90:
        return "medium"
    return "high"


def _rating_for_risk(risk: str, rng: random.Random) -> float:
    """评分与风险等级正相关：low→4.0~5.0，medium→2.8~3.9，high→1.5~2.7。"""
    if risk == "low":
        return round(rng.uniform(4.0, 5.0), 1)
    if risk == "medium":
        return round(rng.uniform(2.8, 3.9), 1)
    return round(rng.uniform(1.5, 2.7), 1)


async def seed_database() -> dict:
    """主入口：清空并重建全部种子数据。"""
    settings = get_settings()
    init_engines(settings)

    rng = random.Random(SEED)
    today = date.today()
    start = today - timedelta(days=180)  # 过去 6 个月

    try:
        async with session_scope() as session:
            # 先建表（幂等）：seed 作为独立脚本运行时表可能不存在
            # （应用内建表在 main.py lifespan，但 docker exec 跑 seed 时不走应用）。
            # create_all 需要跑在 Connection 上（不是 Session）——session.connection()
            # 拿到底层连接后 run_sync 执行同步 DDL。
            conn = await session.connection()
            await conn.run_sync(Base.metadata.create_all)
            # 幂等：先清空（demo 可反复重跑，不产生重复数据）
            await session.execute(Invoice.__table__.delete())
            await session.execute(PurchaseOrder.__table__.delete())
            await session.execute(Supplier.__table__.delete())

            # ---------- 1. 供应商（50 家） ----------
            names = _generate_supplier_names(rng)
            # 先随机分配行业，再显式分配风险等级（金字塔 66/24/10 ≈ 65/25/10）
            STRATEGIC_INDUSTRIES = {"电子元器件", "原材料", "能源材料", "化工产品"}
            name_to_industry: dict[str, tuple[str, str]] = {}
            for name in names:
                if name == ANOMALY_SUPPLIER_GROWTH:
                    name_to_industry[name] = ("电子元器件", "美国")
                elif name == ANOMALY_SUPPLIER_DELIVERY:
                    name_to_industry[name] = ("机械设备", "中国")
                else:
                    name_to_industry[name] = (rng.choice(INDUSTRIES), rng.choice(COUNTRIES))

            # high risk（5 家）：华芯半导体（埋点，必须 high）+ 战略行业排序最靠后的 4 家。
            # 现实语义：被保留的 high risk 是"不可替代的战略供应商"——集中在卡脖子行业
            strategic_names = sorted(
                n for n in names
                if n != ANOMALY_SUPPLIER_GROWTH and name_to_industry[n][0] in STRATEGIC_INDUSTRIES
            )
            high_names = {ANOMALY_SUPPLIER_GROWTH, *strategic_names[-4:]}
            # medium（12 家）：恒达精密制造（交付异常，需关注）+ 其余 11 家
            remaining = [n for n in names if n not in high_names]
            remaining.sort()
            medium_names = {ANOMALY_SUPPLIER_DELIVERY, *remaining[-11:]}
            low_names = [n for n in remaining if n not in medium_names]

            risk_by_name: dict[str, str] = {}
            for n in high_names:
                risk_by_name[n] = "high"
            for n in medium_names:
                risk_by_name[n] = "medium"
            for n in low_names:
                risk_by_name[n] = "low"

            suppliers = []
            for name in sorted(names):
                industry, country = name_to_industry[name]
                risk = risk_by_name[name]
                suppliers.append(Supplier(
                    name=name,
                    industry=industry,
                    country=country,
                    rating=_rating_for_risk(risk, rng),
                    risk_level=risk,
                ))
            session.add_all(suppliers)
            await session.flush()  # 拿到 supplier.id

            # ---------- 2. 采购订单（200 条，帕累托分布 + 埋点异常） ----------
            orders: list[PurchaseOrder] = []
            # 帕累托：前 10 大供应商占 65%（130 条），其余 40 家分 70 条。
            # 注意：两个埋点供应商必须进"前 10 大"——没有足够订单量，
            # "交付率骤降/订单激增"的异常模式在统计上无从显现
            anomaly_names = {ANOMALY_SUPPLIER_DELIVERY, ANOMALY_SUPPLIER_GROWTH}
            top_suppliers = [s for s in suppliers if s.name in anomaly_names]
            top_suppliers += [s for s in suppliers if s.name not in anomaly_names][:8]
            others = [s for s in suppliers if s.name not in anomaly_names][8:]

            def _make_order(sup: Supplier, order_date: date) -> PurchaseOrder:
                """生成一条订单。"""
                cat = sup.industry or "办公用品"
                lo, hi = AMOUNT_RANGE_BY_CATEGORY.get(cat, (1_000, 50_000))
                # 交付异常供应商：近 3 个月大量 cancelled（埋点 a）
                if sup.name == ANOMALY_SUPPLIER_DELIVERY and order_date >= today - timedelta(days=90):
                    status = "cancelled" if rng.random() < 0.6 else "completed"
                # 行业异常：电子元器件整体 cancelled 率偏高（埋点 c）
                elif cat == ANOMALY_INDUSTRY and order_date >= today - timedelta(days=90):
                    status = "cancelled" if rng.random() < 0.35 else "completed"
                else:
                    status = rng.choices(
                        ["completed", "pending", "cancelled"], weights=[85, 8, 7]
                    )[0]
                return PurchaseOrder(
                    supplier_id=sup.id,
                    product_name=rng.choice(PRODUCTS_BY_CATEGORY[cat]),
                    category=cat,
                    amount=round(rng.uniform(lo, hi), 2),
                    quantity=rng.randint(1, 500),
                    status=status,
                    order_date=order_date,
                )

            for sup in top_suppliers:
                for _ in range(13):  # 10 × 13 = 130 条
                    orders.append(_make_order(sup, start + timedelta(days=rng.randint(0, 179))))
            for sup in others:
                n = rng.choices([1, 2, 3], weights=[60, 30, 10])[0]
                for _ in range(n):
                    orders.append(_make_order(sup, start + timedelta(days=rng.randint(0, 179))))

            # 埋点 b：华芯半导体近 2 个月订单金额暴增（high risk + 增长）
            growth_extra = 5  # 额外 5 笔大额订单
            growth_sup = next(s for s in suppliers if s.name == ANOMALY_SUPPLIER_GROWTH)
            for _ in range(growth_extra):
                orders.append(PurchaseOrder(
                    supplier_id=growth_sup.id,
                    product_name=rng.choice(PRODUCTS_BY_CATEGORY["电子元器件"]),
                    category="电子元器件",
                    amount=round(rng.uniform(400_000, 900_000), 2),
                    quantity=rng.randint(100, 1000),
                    status="completed",
                    order_date=today - timedelta(days=rng.randint(0, 60)),
                ))
            session.add_all(orders)
            await session.flush()

            # ---------- 3. 发票（30 条，埋 4 条 overdue） ----------
            # 确定性埋点：恒达精密制造必有 2 张逾期发票（与交付异常联动，
            # 构成"交付率骤降 + 付款逾期"的完整风险画像），另 2 张随机分布
            delivery_sup = next(s for s in suppliers if s.name == ANOMALY_SUPPLIER_DELIVERY)
            delivery_completed = [o for o in orders if o.supplier_id == delivery_sup.id and o.status == "completed"]
            # 取恒达最早完成的 2 单作为逾期（历史订单欠款更久）
            delivery_completed.sort(key=lambda o: o.order_date)
            overdue_orders = delivery_completed[:2]

            other_completed = [o for o in orders if o.status == "completed" and o.supplier_id != delivery_sup.id]
            rng.shuffle(other_completed)
            overdue_orders += other_completed[:2]  # 另 2 条随机逾期

            invoice_orders: list[PurchaseOrder] = delivery_completed + other_completed
            invoices: list[Invoice] = []
            for order in invoice_orders[:30]:
                if order in overdue_orders:
                    status = "overdue"
                else:
                    status = rng.choices(["paid", "unpaid"], weights=[80, 20])[0]
                invoices.append(Invoice(
                    order_id=order.id,
                    invoice_amount=order.amount,
                    payment_status=status,
                    due_date=order.order_date + timedelta(days=30),
                ))
            session.add_all(invoices)

        # 统计并返回摘要
        async with session_scope() as session:
            supplier_count = (await session.execute(select(func.count()).select_from(Supplier))).scalar()
            order_count = (await session.execute(select(func.count()).select_from(PurchaseOrder))).scalar()
            invoice_count = (await session.execute(select(func.count()).select_from(Invoice))).scalar()
            overdue = (await session.execute(
                select(func.count()).select_from(Invoice).where(Invoice.payment_status == "overdue")
            )).scalar()
            high_risk = (await session.execute(
                select(func.count()).select_from(Supplier).where(Supplier.risk_level == "high")
            )).scalar()

        summary = {
            "suppliers": supplier_count,
            "purchase_orders": order_count,
            "invoices": invoice_count,
            "overdue_invoices": overdue,
            "high_risk_suppliers": high_risk,
        }
        print(f"[seed] 完成：{summary}")
        return summary
    finally:
        await dispose_engines()


if __name__ == "__main__":
    asyncio.run(seed_database())
