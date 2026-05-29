#!/usr/bin/env python3
"""
综合数据爬取脚本 - 爬取所有因子计算需要的数据

使用方法:
    python scripts/crawl_comprehensive_data.py

数据来源:
    - akshare: 财务数据、资金流、分析师数据、宏观数据
    - baostock: 历史财务数据
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# 数据目录
DATA_DIR = Path('data/raw/fetched_data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path('data/silver')
SILVER_DIR.mkdir(parents=True, exist_ok=True)

# 检查可用库
try:
    import akshare as ak
    AKSHARE_OK = True
    print("✓ akshare 已安装")
except ImportError:
    AKSHARE_OK = False
    print("✗ akshare 未安装，将跳过相关数据")

try:
    import baostock as bs
    BAOSTOCK_OK = True
    print("✓ baostock 已安装")
except ImportError:
    BAOSTOCK_OK = False
    print("✗ baostock 未安装，将跳过历史财务数据")

print("=" * 70)
print("综合数据爬取")
print("=" * 70)


def get_stock_list() -> list[str]:
    """获取A股股票列表"""
    if AKSHARE_OK:
        try:
            stock_info = ak.stock_info_a_code_name()
            main_stocks = stock_info[stock_info['code'].str.startswith(('6', '000', '002', '300'))]['code'].tolist()
            print(f"  A股股票总数: {len(main_stocks)}")
            return main_stocks
        except Exception as e:
            print(f"  获取股票列表失败: {e}")
    return []


def crawl_financial_indicators(stock_list: list[str], limit: int = 200) -> pd.DataFrame:
    """爬取财务指标数据"""
    print("\n[1] 爬取财务指标数据...")
    all_data = []
    
    if not AKSHARE_OK:
        print("  akshare 未安装，跳过")
        return pd.DataFrame()
    
    stocks = stock_list[:limit]
    for i, code in enumerate(stocks):
        try:
            symbol = code + ('.SH' if code.startswith('6') else '.SZ')
            
            # 方法1: 新浪财务指标
            try:
                df = ak.stock_financial_indicator_sina(symbol, start_date='2018-01-01')
                if df is not None and len(df) > 0:
                    df['symbol'] = symbol
                    df['code'] = code
                    all_data.append(df)
            except:
                pass
            
            time.sleep(0.3)
            
        except Exception as e:
            continue
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.to_parquet(DATA_DIR / 'financial_indicators.parquet', index=False)
        print(f"  保存 {len(result)} 行财务指标")
        return result
    return pd.DataFrame()


def crawl_baostock_financial(stock_list: list[str], limit: int = 100) -> dict:
    """爬取baostock财务数据"""
    print("\n[2] 爬取baostock历史财务数据...")
    
    if not BAOSTOCK_OK:
        print("  baostock 未安装，跳过")
        return {}
    
    bs.login()
    
    results = {
        'profit': [],
        'balance': [],
        'cashflow': [],
    }
    
    stocks = stock_list[:limit]
    for i, symbol in enumerate(stocks):
        try:
            code = symbol.split('.')[0]
            exchange = 'sh' if symbol.endswith('.SH') else 'sz'
            bs_code = f"{exchange}.{code}"
            
            # 利润表
            rs = bs.query_profit_statement_data(bs_code, start_date='2018-01-01')
            data = []
            while rs.error_code == '0' and rs.next():
                data.append(rs.get_row_data())
            if data:
                df = pd.DataFrame(data, columns=rs.fields)
                df['symbol'] = symbol
                results['profit'].append(df)
            
            # 资产负债表
            rs = bs.query_balance_sheet_data(bs_code, start_date='2018-01-01')
            data = []
            while rs.error_code == '0' and rs.next():
                data.append(rs.get_row_data())
            if data:
                df = pd.DataFrame(data, columns=rs.fields)
                df['symbol'] = symbol
                results['balance'].append(df)
            
            # 现金流量表
            rs = bs.query_cash_flow_data(bs_code, start_date='2018-01-01')
            data = []
            while rs.error_code == '0' and rs.next():
                data.append(rs.get_row_data())
            if data:
                df = pd.DataFrame(data, columns=rs.fields)
                df['symbol'] = symbol
                results['cashflow'].append(df)
            
            time.sleep(0.2)
            
        except Exception as e:
            continue
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
    
    bs.logout()
    
    # 保存
    for name, dfs in results.items():
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df.to_parquet(DATA_DIR / f'baostock_{name}.parquet', index=False)
            print(f"  保存 {len(df)} 行{name}数据")
    
    return results


def crawl_money_flow(stock_list: list[str], limit: int = 50) -> pd.DataFrame:
    """爬取资金流数据"""
    print("\n[3] 爬取资金流数据...")
    all_data = []
    
    if not AKSHARE_OK:
        print("  akshare 未安装，跳过")
        return pd.DataFrame()
    
    stocks = stock_list[:limit]
    for i, code in enumerate(stocks):
        try:
            # 个股资金流
            df = ak.stock_individual_fund_flow(symbol=code)
            if df is not None and len(df) > 0:
                df['code'] = code
                all_data.append(df)
            
            time.sleep(0.5)
            
        except Exception as e:
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.to_parquet(DATA_DIR / 'money_flow_individual.parquet', index=False)
        print(f"  保存 {len(result)} 行资金流数据")
        return result
    return pd.DataFrame()


def crawl_sector_money_flow() -> pd.DataFrame:
    """爬取板块资金流"""
    print("\n[4] 爬取板块资金流...")
    
    if not AKSHARE_OK:
        print("  akshare 未安装，跳过")
        return pd.DataFrame()
    
    all_data = []
    
    for period in ['今日', '3日排行', '5日排行', '10日排行']:
        try:
            df = ak.stock_sector_fund_flow_rank(indicator=period)
            if df is not None and len(df) > 0:
                df['period'] = period
                all_data.append(df)
            time.sleep(0.5)
        except:
            pass
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.to_parquet(DATA_DIR / 'money_flow_sector.parquet', index=False)
        print(f"  保存 {len(result)} 行板块资金流")
        return result
    return pd.DataFrame()


def crawl_analyst_data() -> dict:
    """爬取分析师数据"""
    print("\n[5] 爬取分析师数据...")
    
    if not AKSHARE_OK:
        print("  akshare 未安装，跳过")
        return {}
    
    results = {}
    
    # 盈利预测
    try:
        df = ak.stock_profit_forecast(indicator="每股收益")
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'analyst_forecast.parquet', index=False)
            results['forecast'] = df
            print(f"  盈利预测: {len(df)} 行")
    except Exception as e:
        print(f"  盈利预测失败: {e}")
    
    # 评级汇总
    try:
        df = ak.stock_rating_summary(symbol="沪深300")
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'analyst_rating.parquet', index=False)
            results['rating'] = df
            print(f"  评级数据: {len(df)} 行")
    except Exception as e:
        print(f"  评级数据失败: {e}")
    
    # 研报数据
    try:
        df = ak.stock_research_report(symbol="沪深300")
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'analyst_report.parquet', index=False)
            results['report'] = df
            print(f"  研报数据: {len(df)} 行")
    except Exception as e:
        print(f"  研报数据失败: {e}")
    
    return results


def crawl_macro_data() -> dict:
    """爬取宏观数据"""
    print("\n[6] 爬取宏观数据...")
    
    if not AKSHARE_OK:
        print("  akshare 未安装，跳过")
        return {}
    
    results = {}
    
    # 货币供应量
    try:
        df = ak.macro_china_money_supply()
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'macro_money_supply.parquet', index=False)
            results['money_supply'] = df
            print(f"  货币供应: {len(df)} 行")
    except Exception as e:
        print(f"  货币供应失败: {e}")
    
    # CPI数据
    try:
        df = ak.macro_china_cpi()
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'macro_cpi.parquet', index=False)
            results['cpi'] = df
            print(f"  CPI: {len(df)} 行")
    except Exception as e:
        print(f"  CPI失败: {e}")
    
    # 国债收益率
    try:
        df = ak.bond_zh_hsYield()
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'macro_bond_yield.parquet', index=False)
            results['bond_yield'] = df
            print(f"  国债收益率: {len(df)} 行")
    except Exception as e:
        print(f"  国债收益率失败: {e}")
    
    # 利率数据
    try:
        df = ak.macro_china_interest_rate()
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'macro_interest_rate.parquet', index=False)
            results['interest_rate'] = df
            print(f"  利率: {len(df)} 行")
    except Exception as e:
        print(f"  利率失败: {e}")
    
    return results


def crawl_sentiment_data() -> dict:
    """爬取情绪数据"""
    print("\n[7] 爬取情绪数据...")
    
    if not AKSHARE_OK:
        print("  akshare 未安装，跳过")
        return {}
    
    results = {}
    
    # 概念板块
    try:
        df = ak.stock_board_concept_name()
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'concept_board.parquet', index=False)
            results['concept'] = df
            print(f"  概念板块: {len(df)} 行")
    except Exception as e:
        print(f"  概念板块失败: {e}")
    
    # 行业板块
    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'industry_board.parquet', index=False)
            results['industry'] = df
            print(f"  行业板块: {len(df)} 行")
    except Exception as e:
        print(f"  行业板块失败: {e}")
    
    # 涨停股池
    try:
        today = datetime.now().strftime('%Y%m%d')
        df = ak.stock_zt_pool_strong(date=today)
        if df is not None and len(df) > 0:
            df.to_parquet(DATA_DIR / 'zt_pool.parquet', index=False)
            results['zt'] = df
            print(f"  涨停池: {len(df)} 行")
    except Exception as e:
        print(f"  涨停池失败: {e}")
    
    return results


def main():
    """主函数"""
    print("\n开始数据爬取...")
    
    # 获取股票列表
    stock_list = get_stock_list()
    
    if not stock_list:
        print("无法获取股票列表，退出")
        return
    
    # 1. 财务指标
    crawl_financial_indicators(stock_list, limit=300)
    
    # 2. Baostock财务数据
    crawl_baostock_financial(stock_list, limit=200)
    
    # 3. 个股资金流
    crawl_money_flow(stock_list, limit=100)
    
    # 4. 板块资金流
    crawl_sector_money_flow()
    
    # 5. 分析师数据
    crawl_analyst_data()
    
    # 6. 宏观数据
    crawl_macro_data()
    
    # 7. 情绪数据
    crawl_sentiment_data()
    
    print("\n" + "=" * 70)
    print("数据爬取完成!")
    print("=" * 70)
    
    # 列出所有爬取的数据文件
    print("\n已爬取的数据文件:")
    for f in sorted(DATA_DIR.glob('*.parquet')):
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size:.2f} MB")


if __name__ == '__main__':
    main()
