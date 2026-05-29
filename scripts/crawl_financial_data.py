"""
爬取财务数据: 利润表、资产负债表、现金流量表

数据来源:
- akshare: 主要财务数据
- baostock: 历史财务数据
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# 尝试导入数据源
try:
    import akshare as ak
    AKSHARE_OK = True
except:
    AKSHARE_OK = False
    print("akshare 未安装")

try:
    import baostock as bs
    BAOSTOCK_OK = True
except:
    BAOSTOCK_OK = False
    print("baostock 未安装")

DATA_DIR = Path('data/raw/fetched_data')

print("=" * 80)
print("爬取财务数据")
print("=" * 80)

# 获取股票列表
hs300_stocks = []
try:
    # 尝试读取现有股票列表
    bars = pd.read_parquet('data/raw/daily_bar.parquet')
    hs300_stocks = bars['symbol'].unique().tolist()[:50]  # 取前50只作为样本
    print(f"使用价格数据中的 {len(hs300_stocks)} 只股票")
except Exception as e:
    print(f"读取股票列表失败: {e}")

if not hs300_stocks:
    hs300_stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600016.SH', '600019.SH',
                    '600028.SH', '600030.SH', '600050.SH', '600104.SH', '600519.SH']

print(f"\n样本股票: {len(hs300_stocks)} 只")

# =============================================================================
# 1. 爬取akshare财务数据
# =============================================================================
if AKSHARE_OK:
    print("\n[1/4] 爬取akshare财务数据...")
    
    try:
        # 股票列表
        stock_info = ak.stock_info_a_code_name()
        print(f"  A股股票总数: {len(stock_info)}")
        
        # 主板股票代码
        main_stocks = stock_info[stock_info['code'].str.startswith(('6', '000', '002', '300'))]['code'].tolist()[:200]
        print(f"  样本股票: {len(main_stocks)} 只")
        
        # 批量获取财务数据
        all_financial = []
        batch_size = 50
        
        for i in range(0, len(main_stocks), batch_size):
            batch = main_stocks[i:i+batch_size]
            for code in batch:
                try:
                    symbol = code + ('.SH' if code.startswith('6') else '.SZ')
                    
                    # 财务指标
                    df_indicator = ak.stock_financial_indicator_sina(symbol, start_date='2020-01-01')
                    if len(df_indicator) > 0:
                        df_indicator['code'] = code
                        all_financial.append(df_indicator)
                    
                    time.sleep(0.3)
                except Exception as e:
                    continue
                    
            print(f"  进度: {min(i+batch_size, len(main_stocks))}/{len(main_stocks)}")
            time.sleep(1)
        
        if all_financial:
            financial_df = pd.concat(all_financial, ignore_index=True)
            financial_df.to_parquet(DATA_DIR / 'akshare_financial_indicators.parquet', index=False)
            print(f"  保存 {len(financial_df)} 行财务指标数据")
            
    except Exception as e:
        print(f"  akshare爬取失败: {e}")

# =============================================================================
# 2. 爬取baostock财务数据
# =============================================================================
if BAOSTOCK_OK:
    print("\n[2/4] 爬取baostock财务数据...")
    
    try:
        bs.login()
        
        # 财务数据
        all_baostock_fin = []
        
        for symbol in hs300_stocks[:20]:  # 限制数量
            code = symbol.split('.')[0]
            exchange = 'sh' if symbol.endswith('.SH') else 'sz'
            bs_code = f"{exchange}.{code}"
            
            try:
                # 利润表
                rs_profit = bs.query_profit_statement_data(bs_code, start_date='2020-01-01')
                profit_data = []
                while rs_profit.error_code == '0' and rs_profit.next():
                    profit_data.append(rs_profit.get_row_data())
                if profit_data:
                    df_profit = pd.DataFrame(profit_data, columns=rs_profit.fields)
                    df_profit['symbol'] = symbol
                    all_baostock_fin.append(df_profit)
                
                time.sleep(0.2)
            except Exception as e:
                continue
        
        if all_baostock_fin:
            fin_df = pd.concat(all_baostock_fin, ignore_index=True)
            fin_df.to_parquet(DATA_DIR / 'baostock_profit_statement.parquet', index=False)
            print(f"  保存 {len(fin_df)} 行利润表数据")
        
        bs.logout()
        
    except Exception as e:
        print(f"  baostock爬取失败: {e}")

# =============================================================================
# 3. 爬取资产负债数据
# =============================================================================
if BAOSTOCK_OK:
    print("\n[3/4] 爬取资产负债表...")
    
    try:
        bs.login()
        
        all_balance = []
        for symbol in hs300_stocks[:20]:
            code = symbol.split('.')[0]
            exchange = 'sh' if symbol.endswith('.SH') else 'sz'
            bs_code = f"{exchange}.{code}"
            
            try:
                rs_balance = bs.query_balance_sheet_data(bs_code, start_date='2020-01-01')
                balance_data = []
                while rs_balance.error_code == '0' and rs_balance.next():
                    balance_data.append(rs_balance.get_row_data())
                if balance_data:
                    df_balance = pd.DataFrame(balance_data, columns=rs_balance.fields)
                    df_balance['symbol'] = symbol
                    all_balance.append(df_balance)
                
                time.sleep(0.2)
            except Exception as e:
                continue
        
        if all_balance:
            balance_df = pd.concat(all_balance, ignore_index=True)
            balance_df.to_parquet(DATA_DIR / 'baostock_balance_sheet.parquet', index=False)
            print(f"  保存 {len(balance_df)} 行资产负债表数据")
        
        bs.logout()
        
    except Exception as e:
        print(f"  资产负债表爬取失败: {e}")

# =============================================================================
# 4. 爬取现金流量数据
# =============================================================================
if BAOSTOCK_OK:
    print("\n[4/4] 爬取现金流量表...")
    
    try:
        bs.login()
        
        all_cashflow = []
        for symbol in hs300_stocks[:20]:
            code = symbol.split('.')[0]
            exchange = 'sh' if symbol.endswith('.SH') else 'sz'
            bs_code = f"{exchange}.{code}"
            
            try:
                rs_cashflow = bs.query_cash_flow_data(bs_code, start_date='2020-01-01')
                cashflow_data = []
                while rs_cashflow.error_code == '0' and rs_cashflow.next():
                    cashflow_data.append(rs_cashflow.get_row_data())
                if cashflow_data:
                    df_cashflow = pd.DataFrame(cashflow_data, columns=rs_cashflow.fields)
                    df_cashflow['symbol'] = symbol
                    all_cashflow.append(df_cashflow)
                
                time.sleep(0.2)
            except Exception as e:
                continue
        
        if all_cashflow:
            cashflow_df = pd.concat(all_cashflow, ignore_index=True)
            cashflow_df.to_parquet(DATA_DIR / 'baostock_cashflow.parquet', index=False)
            print(f"  保存 {len(cashflow_df)} 行现金流量表数据")
        
        bs.logout()
        
    except Exception as e:
        print(f"  现金流量表爬取失败: {e}")

print("\n完成!")
