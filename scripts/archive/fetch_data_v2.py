"""
数据获取脚本 v2 - 使用正确的akshare API
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")

def fetch_financial_data():
    """获取财务数据"""
    print("\n" + "="*50)
    print("1. 获取财务数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 获取个股资金流
        print("  获取个股资金流...")
        try:
            df = ak.stock_individual_fund_flow_rank(indicator="近5日")
            df.to_parquet(DATA_DIR / "moneyflow_individual_5d.parquet", index=False)
            print(f"  个股资金流5日: {len(df)}条")
        except Exception as e:
            print(f"  个股资金流失败: {e}")
        
        try:
            df = ak.stock_individual_fund_flow_rank(indicator="近10日")
            df.to_parquet(DATA_DIR / "moneyflow_individual_10d.parquet", index=False)
            print(f"  个股资金流10日: {len(df)}条")
        except Exception as e:
            print(f"  个股资金流10日失败: {e}")
            
        # 获取行业资金流
        print("  获取行业资金流...")
        try:
            df = ak.stock_fund_flow_industry()
            df.to_parquet(DATA_DIR / "moneyflow_industry.parquet", index=False)
            print(f"  行业资金流: {len(df)}条")
        except Exception as e:
            print(f"  行业资金流失败: {e}")
        
        # 获取概念资金流
        print("  获取概念资金流...")
        try:
            df = ak.stock_fund_flow_concept()
            df.to_parquet(DATA_DIR / "moneyflow_concept.parquet", index=False)
            print(f"  概念资金流: {len(df)}条")
        except Exception as e:
            print(f"  概念资金流失败: {e}")
        
        # 获取主要资金流
        print("  获取主要资金流...")
        try:
            df = ak.stock_main_fund_flow(symbol="北向资金")
            df.to_parquet(DATA_DIR / "moneyflow_main_north.parquet", index=False)
            print(f"  北向资金: {len(df)}条")
        except Exception as e:
            print(f"  北向资金失败: {e}")
            
        try:
            df = ak.stock_main_fund_flow(symbol="南向资金")
            df.to_parquet(DATA_DIR / "moneyflow_main_south.parquet", index=False)
            print(f"  南向资金: {len(df)}条")
        except Exception as e:
            print(f"  南向资金失败: {e}")
        
        # 财务指标
        print("  获取财务指标...")
        try:
            # 获取财务分析指标
            df = ak.stock_financial_analysis_indicator(symbol="000001", start_year="2017")
            df.to_parquet(DATA_DIR / "financial_indicator_sample.parquet", index=False)
            print(f"  财务指标样本: {len(df)}条")
        except Exception as e:
            print(f"  财务指标失败: {e}")
            
    except Exception as e:
        print(f"  财务数据获取失败: {e}")


def fetch_macro_data():
    """获取宏观数据"""
    print("\n" + "="*50)
    print("2. 获取宏观数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 货币供应量
        print("  获取货币供应量...")
        try:
            df = ak.macro_china_money_supply()
            df.to_parquet(DATA_DIR / "macro_money_supply.parquet", index=False)
            print(f"  货币供应量: {len(df)}条")
        except Exception as e:
            print(f"  货币供应量失败: {e}")
        
        # CPI
        print("  获取CPI数据...")
        try:
            df = ak.macro_china_cpi()
            df.to_parquet(DATA_DIR / "macro_cpi.parquet", index=False)
            print(f"  CPI: {len(df)}条")
        except Exception as e:
            print(f"  CPI失败: {e}")
        
        # 国债收益率
        print("  获取国债收益率...")
        try:
            df = ak.bond_china_benchmark()
            df.to_parquet(DATA_DIR / "bond_benchmark.parquet", index=False)
            print(f"  国债基准: {len(df)}条")
        except Exception as e:
            print(f"  国债收益率失败: {e}")
        
        # 存款准备金率
        print("  获取存款准备金率...")
        try:
            df = ak.bond_china_rrr()
            df.to_parquet(DATA_DIR / "bond_rrr.parquet", index=False)
            print(f"  存款准备金率: {len(df)}条")
        except Exception as e:
            print(f"  存款准备金率失败: {e}")
        
        # 利率
        print("  获取利率数据...")
        try:
            df = ak.macro_bank_china_interest_rate()
            df.to_parquet(DATA_DIR / "macro_interest_rate.parquet", index=False)
            print(f"  利率: {len(df)}条")
        except Exception as e:
            print(f"  利率失败: {e}")
        
        # SHIBOR
        print("  获取SHIBOR...")
        try:
            df = ak.macro_shibor()
            df.to_parquet(DATA_DIR / "macro_shibor.parquet", index=False)
            print(f"  SHIBOR: {len(df)}条")
        except Exception as e:
            print(f"  SHIBOR失败: {e}")
            
    except Exception as e:
        print(f"  宏观数据获取失败: {e}")


def fetch_market_data():
    """获取市场数据"""
    print("\n" + "="*50)
    print("3. 获取市场数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 北向资金流向
        print("  获取沪深港通资金流...")
        try:
            df = ak.stock_hsgt_fund_flow_summary_em()
            df.to_parquet(DATA_DIR / "hsgt_fund_flow.parquet", index=False)
            print(f"  沪深港通: {len(df)}条")
        except Exception as e:
            print(f"  沪深港通失败: {e}")
        
        # 大盘资金流
        print("  获取大盘资金流...")
        try:
            df = ak.stock_individual_fund_flow_rank(indicator="今日")
            df.to_parquet(DATA_DIR / "moneyflow_today.parquet", index=False)
            print(f"  今日资金流: {len(df)}条")
        except Exception as e:
            print(f"  大盘资金流失败: {e}")
        
        # 获取指数成分股权重
        print("  获取沪深300成分股...")
        try:
            df = ak.index_hs300_cons_em()
            df.to_parquet(DATA_DIR / "hs300_constituent.parquet", index=False)
            print(f"  沪深300成分: {len(df)}条")
        except Exception as e:
            print(f"  沪深300失败: {e}")
            
    except Exception as e:
        print(f"  市场数据获取失败: {e}")


def main():
    print("="*60)
    print("数据获取脚本 v2")
    print("="*60)
    
    fetch_financial_data()
    fetch_macro_data()
    fetch_market_data()
    
    print("\n" + "="*60)
    print("数据获取完成!")
    print("="*60)
    
    # 列出所有数据
    print("\n已获取数据文件:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")


if __name__ == "__main__":
    main()
