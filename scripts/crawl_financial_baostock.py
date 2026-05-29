"""
财务数据爬取脚本 v3 - 高效版

只查询最新可用数据，避免大量循环
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import baostock as bs
    BAOSTOCK_OK = True
except ImportError:
    BAOSTOCK_OK = False

RAW_DATA_DIR = Path('data/raw/fetched_data')
SILVER_DATA_DIR = Path('data/silver')
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_stock_list() -> list[str]:
    """获取A股股票列表"""
    if not BAOSTOCK_OK:
        return []
    
    bs.login()
    rs = bs.query_stock_basic(code_name='')
    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    stocks = df[
        df['code'].str.match(r'^sh\.6|sz\.000|sz\.002|sz\.300') & 
        (df['status'] == '1')
    ]['code'].tolist()
    return stocks


def crawl_stock_latest(stock_code: str):
    """只爬取单只股票最新可用财务数据"""
    if not BAOSTOCK_OK:
        return {}
    
    exchange = stock_code[:2]
    code = stock_code[3:]
    bs_code = f"{exchange}.{code}"
    
    result = {'symbol': stock_code}
    
    quarters = [('2024', '4'), ('2024', '3'), ('2024', '2'), ('2024', '1'),
                ('2023', '4'), ('2023', '3'), ('2023', '2'), ('2023', '1')]
    
    for year, quarter in quarters:
        try:
            # 利润表
            rs = bs.query_profit_data(bs_code, year=year, quarter=quarter)
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                if row and len(row) > 3 and row[3] and row[3] != '':
                    result['roe'] = float(row[3])
                    if row[4]: result['npm'] = float(row[4])
                    if row[5]: result['gpm'] = float(row[5])
                    if row[7]: result['eps'] = float(row[7])
                    break
            
            # 杜邦
            rs = bs.query_dupont_data(bs_code, year=year, quarter=quarter)
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                if row and len(row) > 3 and row[3] and row[3] != '':
                    result['dupont_roe'] = float(row[3])
                    if row[4]: result['dupont_leverage'] = float(row[4])
                    if row[5]: result['dupont_asset_turn'] = float(row[5])
                    break
            
            # 资产负债
            rs = bs.query_balance_data(bs_code, year=year, quarter=quarter)
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                if row and len(row) > 3 and row[3] and row[3] != '':
                    if row[3]: result['current_ratio'] = float(row[3])
                    if row[4]: result['quick_ratio'] = float(row[4])
                    if row[5]: result['cash_ratio'] = float(row[5])
                    if row[8]: result['debt_ratio'] = float(row[8])
                    break
            
            # 现金流
            rs = bs.query_cash_flow_data(bs_code, year=year, quarter=quarter)
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                if row and len(row) > 3 and row[3] and row[3] != '':
                    if row[3]: result['cf_ca_to_asset'] = float(row[3])
                    if row[8]: result['cf_cfo_to_or'] = float(row[8])
                    break
            
            # 运营能力
            rs = bs.query_operation_data(bs_code, year=year, quarter=quarter)
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                if row and len(row) > 3 and row[3] and row[3] != '':
                    if row[3]: result['nr_turn'] = float(row[3])
                    if row[5]: result['inv_turn'] = float(row[5])
                    if row[8]: result['asset_turn'] = float(row[8])
                    break
            
            # 成长能力
            rs = bs.query_growth_data(bs_code, year=year, quarter=quarter)
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                if row and len(row) > 3 and row[3] and row[3] != '':
                    if row[3]: result['equity_growth'] = float(row[3])
                    if row[4]: result['asset_growth'] = float(row[4])
                    if row[5]: result['ni_growth'] = float(row[5])
                    break
            
            # 如果有数据就停止
            if len(result) > 1:
                break
                
        except Exception as e:
            continue
        
        time.sleep(0.05)
    
    return result


def crawl_batch(stocks: list[str], max_stocks: int = 500):
    """批量爬取"""
    if not BAOSTOCK_OK:
        return pd.DataFrame()
    
    stocks = stocks[:max_stocks]
    all_data = []
    
    bs.login()
    
    for i, stock in enumerate(stocks):
        data = crawl_stock_latest(stock)
        if data and len(data) > 1:
            all_data.append(data)
        
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(stocks)}, 有效: {len(all_data)}")
        
        time.sleep(0.08)
    
    bs.logout()
    return pd.DataFrame(all_data)


def create_factor_cache(financial_df: pd.DataFrame) -> pd.DataFrame:
    """转换为因子缓存"""
    if financial_df.empty:
        return pd.DataFrame()
    
    factor_mappings = {
        'roe': ('roe', 1),
        'dupont_roe': ('roe', 1),
        'dupont_asset_turn': ('asset_turnover', 1),
        'asset_turn': ('asset_turnover', 1),
        'nr_turn': ('ar_turnover', 1),
        'inv_turn': ('inv_turnover', 1),
        'debt_ratio': ('debt_ratio', -1),
        'current_ratio': ('current_ratio', 1),
        'quick_ratio': ('quick_ratio', 1),
        'cash_ratio': ('cash_ratio', 1),
        'npm': ('net_margin', 1),
        'gpm': ('gross_margin', 1),
        'eps': ('eps', 1),
        'cf_ca_to_asset': ('ocf_to_asset', 1),
        'cf_cfo_to_or': ('ocf_to_revenue', 1),
        'equity_growth': ('equity_growth', 1),
        'asset_growth': ('asset_growth', -1),
        'ni_growth': ('profit_growth', 1),
        'dupont_leverage': ('leverage', 1),
    }
    
    records = []
    trade_date = pd.Timestamp.now()
    
    for _, row in financial_df.iterrows():
        symbol = row.get('symbol')
        if not symbol:
            continue
        
        for col, (factor_name, direction) in factor_mappings.items():
            if col in row.index and pd.notna(row[col]) and row[col] != '':
                try:
                    value = float(row[col])
                    if direction < 0:
                        value = -value
                    records.append({
                        'symbol': symbol,
                        'trade_date': trade_date,
                        'factor_name': factor_name,
                        'value': value,
                    })
                except (ValueError, TypeError):
                    continue
    
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Baostock 财务数据爬取 v3 (高效版)")
    print("=" * 60)
    
    if not BAOSTOCK_OK:
        print("baostock 不可用")
        return
    
    print("\n[1/3] 获取股票列表...")
    stocks = get_stock_list()
    print(f"  找到 {len(stocks)} 只股票")
    
    print("\n[2/3] 爬取财务数据...")
    df = crawl_batch(stocks, max_stocks=500)
    
    if not df.empty:
        print(f"\n  成功: {len(df)} 只股票")
        
        # 保存
        raw_path = RAW_DATA_DIR / 'baostock_financial_v3.parquet'
        df.to_parquet(raw_path, index=False)
        print(f"  保存: {raw_path}")
        
        # 因子缓存
        cache = create_factor_cache(df)
        if not cache.empty:
            cache_path = RAW_DATA_DIR / 'financial_factors.parquet'
            cache.to_parquet(cache_path, index=False)
            print(f"  因子缓存: {len(cache)} 条")
            print(cache['factor_name'].value_counts())
        
        # silver层
        silver_path = SILVER_DATA_DIR / 'financial_indicator.parquet'
        df.to_parquet(silver_path, index=False)
        print(f"  silver层: {silver_path}")
    else:
        print("  无数据")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
