"""
计算分析师预期因子、资金流因子、Barra风格因子

数据来源:
- 盈利预测: stock_profit_forecast_all.parquet
- 融资融券: margin_summary.parquet
- 北向资金: hsgt_hist.parquet
- 财务数据: baostock_dupont.parquet, financial_report_sina.parquet
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data/raw')
CACHE_DIR = Path('data/factor_cache')
FETCHED_DIR = DATA_DIR / 'fetched_data'

print("=" * 80)
print("计算分析师预期因子、资金流因子、Barra风格因子")
print("=" * 80)

# Load price data
bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
bars['trade_date'] = pd.to_datetime(bars['trade_date'])
bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
bars = bars[bars['trade_date'] >= '2020-01-01'].copy()

# Calculate returns
bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)

print(f"价格数据: {len(bars):,} 行, {bars['symbol'].nunique()} 只股票")

# =============================================================================
# 1. 分析师预期因子
# =============================================================================
print("\n[1/4] 计算分析师预期因子...")

try:
    forecast = pd.read_parquet(FETCHED_DIR / 'stock_profit_forecast_all.parquet')
    print(f"  盈利预测数据: {len(forecast)} 行")
    
    # 提取分析师评级数据
    analyst_df = forecast[['代码', '名称', '研报数', 
                           '2024预测每股收益', '2025预测每股收益', 
                           '2026预测每股收益', '2027预测每股收益',
                           '机构投资评级(近六个月)-买入', 
                           '机构投资评级(近六个月)-增持',
                           '机构投资评级(近六个月)-中性']].copy()
    
    analyst_df.columns = ['symbol', 'name', 'report_count', 
                          'eps_f_2024', 'eps_f_2025', 'eps_f_2026', 'eps_f_2027',
                          'rating_buy', 'rating_add', 'rating_neutral']
    
    # 标准化代码
    analyst_df['symbol'] = analyst_df['symbol'].astype(str).str.zfill(6)
    analyst_df['symbol'] = analyst_df['symbol'].apply(
        lambda x: x + '.SZ' if x.startswith(('0', '3')) else x + '.SH' if x.startswith(('6',)) else x
    )
    
    # 计算分析师预期因子
    analyst_df['eps_forecast_1y'] = analyst_df['eps_f_2025']  # 1年期预测
    analyst_df['eps_forecast_2y'] = analyst_df['eps_f_2026']  # 2年期预测
    analyst_df['eps_forecast_growth_1y'] = (analyst_df['eps_f_2025'] / analyst_df['eps_f_2024'] - 1) * 100
    analyst_df['eps_forecast_growth_2y'] = (analyst_df['eps_f_2026'] / analyst_df['eps_f_2024'] - 1) * 100
    analyst_df['rating_score'] = (analyst_df['rating_buy'] * 5 + 
                                   analyst_df['rating_add'] * 4 + 
                                   analyst_df['rating_neutral'] * 3) / (analyst_df['report_count'] + 1)
    analyst_df['report_count_norm'] = np.log1p(analyst_df['report_count'])
    
    # 与当前价格计算预期收益率 (简化代理)
    bars_current = bars.groupby('symbol').last().reset_index()
    bars_current = bars_current[['symbol', 'close']]
    analyst_df = analyst_df.merge(bars_current, on='symbol', how='left')
    
    # 假设1年后股价 = eps_forecast_1y * 行业平均PE
    analyst_df['expected_return_1y'] = (analyst_df['eps_f_2025'] / analyst_df['close'] - 1) * 100
    
    print(f"  计算了 {len(analyst_df)} 只股票的分析师预期因子")
    
except Exception as e:
    print(f"  分析师数据加载失败: {e}")
    analyst_df = pd.DataFrame()

# =============================================================================
# 2. 资金流因子
# =============================================================================
print("\n[2/4] 计算资金流因子...")

try:
    # 北向资金
    hsgt = pd.read_parquet(FETCHED_DIR / 'hsgt_hist.parquet')
    hsgt['日期'] = pd.to_datetime(hsgt['日期'])
    hsgt = hsgt.rename(columns={
        '日期': 'trade_date',
        '当日成交净买额': 'hsgt_net_buy',
        '买入成交额': 'hsgt_buy',
        '卖出成交额': 'hsgt_sell'
    })
    hsgt['hsgt_net_buy_pct'] = hsgt['hsgt_net_buy'] / (hsgt['hsgt_buy'] + hsgt['hsgt_sell']) * 100
    print(f"  北向资金: {len(hsgt)} 行")
    
    # 融资融券
    margin = pd.read_parquet(FETCHED_DIR / 'margin_summary.parquet')
    margin['证券代码'] = margin['证券代码'].astype(str).str.zfill(6)
    margin['symbol'] = margin['证券代码'].apply(
        lambda x: x + '.SZ' if x.startswith(('0', '3')) else x + '.SH' if x.startswith(('6',)) else x
    )
    
    # 计算融资因子
    margin['margin_balance'] = margin['融资余额']
    margin['margin_buy'] = margin['融资买入额']
    margin['short_balance'] = margin['融券余额']
    margin['short_volume'] = margin['融券余量']
    margin['margin_ratio'] = margin['融资余额'] / margin['融资融券余额']
    
    # 5日/20日移动平均 (按市值加权简化)
    margin['margin_change_5d'] = margin.groupby('symbol')['margin_balance'].pct_change(5)
    margin['margin_change_20d'] = margin.groupby('symbol')['margin_balance'].pct_change(20)
    
    print(f"  融资融券: {len(margin)} 行, {margin['symbol'].nunique()} 只股票")
    
    # 个股资金流 (如果有)
    stock_mf = pd.read_parquet(FETCHED_DIR / 'stock_money_flow.parquet')
    if len(stock_mf) > 0:
        print(f"  个股资金流: {len(stock_mf)} 行")
    
except Exception as e:
    print(f"  资金流数据加载失败: {e}")
    hsgt = pd.DataFrame()
    margin = pd.DataFrame()

# =============================================================================
# 3. Barra风格因子 (从财务数据计算)
# =============================================================================
print("\n[3/4] 计算Barra风格因子...")

try:
    # 加载财务数据
    fin_report = pd.read_parquet(FETCHED_DIR / 'financial_report_sina.parquet')
    print(f"  财务报告: {len(fin_report)} 行, 列: {list(fin_report.columns)[:10]}...")
    
    # Dupont分析
    dupont = pd.read_parquet(FETCHED_DIR / 'baostock_dupont.parquet')
    if len(dupont) > 0:
        print(f"  Dupont数据: {len(dupont)} 行")
    
except Exception as e:
    print(f"  财务数据加载失败: {e}")
    fin_report = pd.DataFrame()
    dupont = pd.DataFrame()

# =============================================================================
# 4. 计算技术代理因子
# =============================================================================
print("\n[4/4] 计算技术代理因子...")

# 计算市场代理因子
bars['mkt_cap_proxy'] = bars['amount'] / bars['volume'].replace(0, np.nan) * bars['volume']
bars['ln_mkt_cap_proxy'] = np.log(bars['mkt_cap_proxy'].replace(0, np.nan))
bars['size_proxy'] = -bars['ln_mkt_cap_proxy']  # size因子 (小市值 = 大值)

# 计算Beta代理
bars['vol_60d'] = bars.groupby('symbol')['returns'].transform(
    lambda x: x.rolling(60, min_periods=20).std()
)
mkt_vol = bars.groupby('trade_date')['vol_60d'].transform('mean')
bars['beta_proxy'] = bars['vol_60d'] / mkt_vol.replace(0, np.nan)

# 计算流动性代理
bars['adv20'] = bars.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
bars['liquidity_proxy'] = -np.log((bars['adv20'] * bars['close']).replace(0, np.nan))

# 计算动量因子
bars['mom_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20)
bars['mom_60d'] = bars.groupby('symbol')['adj_close'].pct_change(60)
bars['mom_120d'] = bars.groupby('symbol')['adj_close'].pct_change(120)
bars['mom_250d'] = bars.groupby('symbol')['adj_close'].pct_change(250)

# 计算波动率因子
bars['vol_20d'] = bars.groupby('symbol')['returns'].transform(lambda x: x.rolling(20, min_periods=5).std() * np.sqrt(252))
bars['vol_120d'] = bars.groupby('symbol')['returns'].transform(lambda x: x.rolling(120, min_periods=30).std() * np.sqrt(252))

# =============================================================================
# 合并所有因子到bars
# =============================================================================
print("\n合并因子到价格数据...")

# Barra代理因子
bars['size_nonlinear'] = 3 * (-bars['ln_mkt_cap_proxy'].groupby(bars['trade_date']).rank(pct=True)) - \
                         2 * ((-bars['ln_mkt_cap_proxy'].groupby(bars['trade_date']).rank(pct=True)) ** 2)
bars['book_to_price_proxy'] = 1 / bars['close'].replace(0, np.nan)
bars['volatility_proxy'] = bars['vol_60d']

# =============================================================================
# 计算IC
# =============================================================================
print("\n" + "=" * 80)
print("计算IC...")
print("=" * 80)

def calculate_ic(panel, factor_col, label_col='fwd_return_20d'):
    if factor_col not in panel.columns or label_col not in panel.columns:
        return np.nan
    valid = panel[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return np.nan
    return valid[factor_col].corr(valid[label_col], method='spearman')

# 因子列表
factors_to_test = {
    # 分析师因子
    'eps_forecast_1y': 'analyst',
    'eps_forecast_2y': 'analyst',
    'eps_forecast_growth_1y': 'analyst',
    'eps_forecast_growth_2y': 'analyst',
    'rating_score': 'analyst',
    'report_count_norm': 'analyst',
    'expected_return_1y': 'analyst',
    
    # 资金流因子
    'hsgt_net_buy': 'money_flow',  # 需要日期匹配
    'hsgt_net_buy_pct': 'money_flow',
    
    # Barra代理因子
    'size': 'barra',
    'size_nonlinear': 'barra',
    'beta': 'barra',
    'beta_proxy': 'barra',
    'liquidity': 'barra',
    'liquidity_proxy': 'barra',
    'volatility': 'barra',
    'volatility_proxy': 'barra',
    'book_to_price_proxy': 'barra',
    
    # 动量因子
    'mom_20d': 'technical',
    'mom_60d': 'technical',
    'mom_120d': 'technical',
    'mom_250d': 'technical',
    
    # 波动率因子
    'vol_20d': 'technical',
    'vol_60d': 'technical',
    'vol_120d': 'technical',
}

results = []
for name, source in factors_to_test.items():
    if name in bars.columns:
        ic = calculate_ic(bars, name)
        if not np.isnan(ic):
            results.append({
                'factor': name,
                'source': source,
                'rank_ic': ic
            })
            print(f"  {name:<25s} IC={ic:+.4f} ({source})")

# 保存结果
results_df = pd.DataFrame(results)
if len(results_df) > 0:
    # 合并到现有IC
    existing_ic = pd.read_csv(CACHE_DIR / 'all_factors_ic_complete.csv')
    all_ic = pd.concat([existing_ic, results_df], ignore_index=True)
    all_ic = all_ic.drop_duplicates(subset=['factor'], keep='last')
    all_ic = all_ic.sort_values('rank_ic', ascending=False)
    all_ic.to_csv(CACHE_DIR / 'all_factors_ic_complete.csv', index=False)
    print(f"\n新增 {len(results)} 个因子IC, 总计 {len(all_ic)} 个")

print("\n完成!")
