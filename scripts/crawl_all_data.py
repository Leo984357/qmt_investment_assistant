"""
爬取因子库所需的所有数据
"""
import akshare as ak
import pandas as pd
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data/raw/fetched_data')
DATA_DIR.mkdir(exist_ok=True, parents=True)

def save_data(df, filename):
    """保存数据"""
    if df is not None and len(df) > 0:
        path = DATA_DIR / filename
        df.to_parquet(path, index=False)
        print(f"  ✓ 保存: {path} ({len(df)} 行)")
        return True
    return False


# ==================== 1. 资金流数据 ====================
print("=" * 60)
print("1. 爬取资金流数据")
print("=" * 60)

# 1.1 个股资金流向
try:
    print("\n1.1 个股资金流向...")
    df = ak.stock_individual_fund_flow(stock='000001', market='sz')
    save_data(df, 'money_flow_sample.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 1.2 行业资金流向
try:
    print("\n1.2 行业资金流向...")
    df = ak.stock_sector_fund_flow_rank(indicator='5日', sector_type='行业资金流向')
    save_data(df, 'sector_fund_flow.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 1.3 概念资金流向
try:
    print("\n1.3 概念资金流向...")
    df = ak.stock_sector_fund_flow_rank(indicator='5日', sector_type='概念资金流向')
    save_data(df, 'concept_fund_flow.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 1.4 主力资金流向
try:
    print("\n1.4 主力资金流向...")
    df = ak.stock_main_fund_flow(symbol='全部')
    save_data(df, 'main_fund_flow.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 1.5 北向资金流向
try:
    print("\n1.5 北向资金流向...")
    df = ak.stock_hsgt_fund_flow_summary_em()
    save_data(df, 'hsgt_fund_flow.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 1.6 沪股通历史
try:
    print("\n1.6 沪股通历史...")
    df = ak.stock_hsgt_hsgt_list_em(symbol='沪股通')
    save_data(df, 'hgt_history.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 1.7 深股通历史
try:
    print("\n1.7 深股通历史...")
    df = ak.stock_hsgt_hsgt_list_em(symbol='深股通')
    save_data(df, 'sgt_history.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")


# ==================== 2. 融资融券数据 ====================
print("\n" + "=" * 60)
print("2. 爬取融资融券数据")
print("=" * 60)

# 2.1 融资融券汇总
try:
    print("\n2.1 融资融券汇总...")
    df = ak.stock_margin_detail_szse(date='20260410')
    save_data(df, 'margin_summary.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 2.2 融资融券历史
try:
    print("\n2.2 融资融券历史...")
    df = ak.stock_margin_detail_sse(start_date='20240101', end_date='20260410')
    save_data(df, 'margin_history_sse.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")


# ==================== 3. 龙虎榜数据 ====================
print("\n" + "=" * 60)
print("3. 爬取龙虎榜数据")
print("=" * 60)

# 3.1 龙虎榜个股详情
try:
    print("\n3.1 龙虎榜个股详情...")
    df = ak.stock_lhb_stock_detail_em(start_date='20240101', end_date='20260410')
    save_data(df, 'lhb_stock_detail.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 3.2 龙虎榜营业部
try:
    print("\n3.2 龙虎榜营业部排行...")
    df = ak.stock_lhb_yybph_em()
    save_data(df, 'lhb_yyb.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 3.3 龙虎榜统计
try:
    print("\n3.3 龙虎榜统计...")
    df = ak.stock_lhb_stock_statistic_em(start_date='20240101', end_date='20260410')
    save_data(df, 'lhb_statistic.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

time.sleep(2)


# ==================== 4. 分析师预期数据 ====================
print("\n" + "=" * 60)
print("4. 爬取分析师预期数据")
print("=" * 60)

# 4.1 盈利预测
try:
    print("\n4.1 盈利预测...")
    df = ak.stock_profit_forecast(indicator='全部', symbol='00856')
    save_data(df, 'profit_forecast_sample.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 4.2 评级汇总
try:
    print("\n4.2 评级汇总...")
    df = ak.stock_rating_combined(symbol='SH600519')
    save_data(df, 'rating_combined_sample.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 4.3 目标价
try:
    print("\n4.3 目标价...")
    df = ak.stock_target_price_estate_em(symbol='SH600519')
    save_data(df, 'target_price_sample.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 4.4 盈利预测详情
try:
    print("\n4.4 盈利预测详情...")
    df = ak.stock_profit_forecast(indicator='个股评级', symbol='00856')
    save_data(df, 'profit_forecast_stock.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 4.5 投资评级
try:
    print("\n4.5 投资评级...")
    df = ak.stock_invest_rating_report_em(symbol='评级研报')
    save_data(df, 'invest_rating.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

time.sleep(2)


# ==================== 5. 情绪因子数据 ====================
print("\n" + "=" * 60)
print("5. 爬取情绪因子数据")
print("=" * 60)

# 5.1 恐慌指数
try:
    print("\n5.1 恐慌指数...")
    df = ak.stock_vix_index(symbol='VIX', period='3个月')
    save_data(df, 'vix_index.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 5.2 融资融券比
try:
    print("\n5.2 融资融券比...")
    df = ak.macro_china_market_margin_sh()
    save_data(df, 'margin_ratio_sh.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 5.3 货币供应量
try:
    print("\n5.3 货币供应量...")
    df = ak.macro_china_money_supply()
    save_data(df, 'money_supply.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 5.4 涨跌停数据
try:
    print("\n5.4 涨跌停统计...")
    df = ak.stock_up_down_em(start_date='20240101', end_date='20260410')
    save_data(df, 'up_down_stats.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 5.5 市场交易统计
try:
    print("\n5.5 市场交易统计...")
    df = ak.stock_market_trade_summary_em(date='20260410')
    save_data(df, 'market_trade_summary.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 5.6 情绪指数
try:
    print("\n5.6 市场情绪...")
    df = ak.stock_sentiment_em(index='上证指数')
    save_data(df, 'market_sentiment.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

time.sleep(2)


# ==================== 6. 宏观数据 ====================
print("\n" + "=" * 60)
print("6. 爬取宏观数据")
print("=" * 60)

# 6.1 GDP
try:
    print("\n6.1 GDP...")
    df = ak.macro_china_gdp()
    save_data(df, 'gdp.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 6.2 CPI
try:
    print("\n6.2 CPI...")
    df = ak.macro_china_cpi()
    save_data(df, 'cpi.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 6.3 PPI
try:
    print("\n6.3 PPI...")
    df = ak.macro_china_ppi()
    save_data(df, 'ppi.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 6.4 LPR
try:
    print("\n6.4 LPR...")
    df = ak.macro_china_lpr()
    save_data(df, 'lpr.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 6.5 存款准备金率
try:
    print("\n6.5 存款准备金率...")
    df = ak.macro_china_reserve_requirement_ratio()
    save_data(df, 'rrr.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 6.6 国债收益率
try:
    print("\n6.6 国债收益率...")
    df = ak.bond_china_yield()
    save_data(df, 'bond_yield.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")


# ==================== 7. 其他数据 ====================
print("\n" + "=" * 60)
print("7. 爬取其他数据")
print("=" * 60)

# 7.1 大宗交易
try:
    print("\n7.1 大宗交易...")
    df = ak.stock_fund_flow_big_deal(start_date='20240101', end_date='20260410')
    save_data(df, 'big_deal.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 7.2 股东户数
try:
    print("\n7.2 股东户数...")
    df = ak.stock_shareholder_number_change(symbol='全部', end_date='20260410')
    save_data(df, 'shareholder_number.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 7.3 机构调研
try:
    print("\n7.3 机构调研...")
    df = ak.stock_analyst_rank_em()
    save_data(df, 'analyst_rank.parquet')
except Exception as e:
    print(f"  ✗ 失败: {e}")


print("\n" + "=" * 60)
print("数据爬取完成!")
print("=" * 60)

# 列出所有获取的数据
print("\n获取的数据文件:")
for f in sorted(DATA_DIR.glob('*.parquet')):
    df = pd.read_parquet(f)
    print(f"  {f.name}: {len(df)} 行, {len(df.columns)} 列")
