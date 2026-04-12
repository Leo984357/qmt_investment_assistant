"""
完整数据获取脚本 - 获取因子库所需全部数据

数据需求:
1. 财务数据 (akshare)
2. 资金流数据 (akshare东方财富)
3. 分析师预期 (akshare)
4. 宏观数据 (akshare)
5. 行业分类数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 数据目录
DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 股票列表 (从universe获取)
def get_stock_list():
    """获取股票列表"""
    universe = pd.read_parquet(DATA_DIR / "universe_membership.parquet")
    stocks = universe['symbol'].unique()
    print(f"股票列表: {len(stocks)}只")
    return stocks.tolist()


def fetch_financial_data():
    """获取财务数据"""
    print("\n" + "="*50)
    print("1. 获取财务数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 获取所有股票的财务数据
        stock_list = get_stock_list()
        
        all_financial = []
        
        for i, stock in enumerate(stock_list[:100]):  # 先测试100只
            try:
                if stock.endswith('.SH'):
                    code = stock.replace('.SH', '')
                    market = 'sh'
                else:
                    code = stock.replace('.SZ', '')
                    market = 'sz'
                
                # 获取财务指标
                df = ak.stock_financial_abstract_ths(symbol=code, start_year='2017')
                if df is not None and len(df) > 0:
                    df['symbol'] = stock
                    all_financial.append(df)
                
                if (i+1) % 20 == 0:
                    print(f"  进度: {i+1}/{len(stock_list[:100])}")
                    
            except Exception as e:
                continue
        
        if all_financial:
            financial_df = pd.concat(all_financial, ignore_index=True)
            financial_df.to_parquet(DATA_DIR / "financial_data.parquet", index=False)
            print(f"  财务数据已保存: {len(financial_df)}条")
        else:
            print("  未能获取财务数据")
            
    except Exception as e:
        print(f"  财务数据获取失败: {e}")


def fetch_money_flow_data():
    """获取资金流数据"""
    print("\n" + "="*50)
    print("2. 获取资金流数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 东方财富资金流
        print("  获取东方财富资金流...")
        
        # 大盘资金流
        try:
            df = ak.moneyflow_hsgt_em()
            df.to_parquet(DATA_DIR / "moneyflow_hsgt.parquet", index=False)
            print(f"  沪深港通资金流: {len(df)}条")
        except Exception as e:
            print(f"  沪深港通资金流失败: {e}")
        
        # 个股资金流
        print("  获取个股资金流(概念板块)...")
        
        # 获取概念板块资金流
        try:
            df = ak.moneyflow_individual_em(symbol="主力净流入")
            df.to_parquet(DATA_DIR / "moneyflow_concept.parquet", index=False)
            print(f"  概念板块资金流: {len(df)}条")
        except Exception as e:
            print(f"  概念板块资金流失败: {e}")
        
        # 行业资金流
        try:
            df = ak.moneyflow_industry_em()
            df.to_parquet(DATA_DIR / "moneyflow_industry.parquet", index=False)
            print(f"  行业资金流: {len(df)}条")
        except Exception as e:
            print(f"  行业资金流失败: {e}")
            
    except Exception as e:
        print(f"  资金流数据获取失败: {e}")


def fetch_analyst_data():
    """获取分析师预期数据"""
    print("\n" + "="*50)
    print("3. 获取分析师预期数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 研报数据
        print("  获取券商研报数据...")
        try:
            df = ak.stock_research_report_em()
            df.to_parquet(DATA_DIR / "research_reports.parquet", index=False)
            print(f"  研报数据: {len(df)}条")
        except Exception as e:
            print(f"  研报数据失败: {e}")
        
        # 盈利预测
        print("  获取盈利预测数据...")
        try:
            df = ak.stock_profit_forecast_by_report_em(symbol="比亚迪")
            df.to_parquet(DATA_DIR / "profit_forecast.parquet", index=False)
            print(f"  盈利预测: {len(df)}条")
        except Exception as e:
            print(f"  盈利预测失败: {e}")
            
        # 评级数据
        print("  获取评级数据...")
        try:
            df = ak.stock_rating_detail()
            df.to_parquet(DATA_DIR / "stock_rating.parquet", index=False)
            print(f"  评级数据: {len(df)}条")
        except Exception as e:
            print(f"  评级数据失败: {e}")
            
    except Exception as e:
        print(f"  分析师数据获取失败: {e}")


def fetch_macro_data():
    """获取宏观数据"""
    print("\n" + "="*50)
    print("4. 获取宏观数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 国债收益率
        print("  获取国债收益率曲线...")
        try:
            df = ak.bond_china_yield_curve()
            df.to_parquet(DATA_DIR / "bond_yield_curve.parquet", index=False)
            print(f"  国债收益率: {len(df)}条")
        except Exception as e:
            print(f"  国债收益率失败: {e}")
        
        # CPI/PPI数据
        print("  获取CPI/PPI数据...")
        try:
            df = ak.macro_china_cpi()
            df.to_parquet(DATA_DIR / "macro_cpi.parquet", index=False)
            print(f"  CPI: {len(df)}条")
        except Exception as e:
            print(f"  CPI失败: {e}")
        
        # 社融数据
        print("  获取社融数据...")
        try:
            df = ak.macro_china_shrzgm()
            df.to_parquet(DATA_DIR / "macro_shrzgm.parquet", index=False)
            print(f"  社融: {len(df)}条")
        except Exception as e:
            print(f"  社融失败: {e}")
        
        # 汇率数据
        print("  获取汇率数据...")
        try:
            df = ak.currency_usd_cny_hist()
            df.to_parquet(DATA_DIR / "fx_usd_cny.parquet", index=False)
            print(f"  USD/CNY: {len(df)}条")
        except Exception as e:
            print(f"  汇率失败: {e}")
        
        # 商品期货
        print("  获取商品期货数据...")
        try:
            df = ak.futures_cn_index()
            df.to_parquet(DATA_DIR / "futures_index.parquet", index=False)
            print(f"  商品指数: {len(df)}条")
        except Exception as e:
            print(f"  商品期货失败: {e}")
            
    except Exception as e:
        print(f"  宏观数据获取失败: {e}")


def fetch_industry_classification():
    """获取行业分类数据"""
    print("\n" + "="*50)
    print("5. 获取行业分类数据")
    print("="*50)
    
    try:
        import akshare as ak
        
        # 申万行业分类
        print("  获取申万行业分类...")
        try:
            df = ak.sw_index_second_cons_sina(symbol="801010")
            df.to_parquet(DATA_DIR / "industry_sw_second.parquet", index=False)
            print(f"  申万二级行业: {len(df)}条")
        except Exception as e:
            print(f"  申万行业失败: {e}")
        
        # 东财行业分类
        print("  获取东财行业分类...")
        try:
            df = ak.stock_board_industry_name_em()
            df.to_parquet(DATA_DIR / "industry_em.parquet", index=False)
            print(f"  东财行业: {len(df)}条")
        except Exception as e:
            print(f"  东财行业失败: {e}")
        
        # 概念板块
        print("  获取概念板块...")
        try:
            df = ak.stock_board_concept_name_em()
            df.to_parquet(DATA_DIR / "concept_board.parquet", index=False)
            print(f"  概念板块: {len(df)}条")
        except Exception as e:
            print(f"  概念板块失败: {e}")
            
    except Exception as e:
        print(f"  行业分类获取失败: {e}")


def main():
    print("="*60)
    print("完整数据获取脚本")
    print("="*60)
    print(f"开始时间: {datetime.now()}")
    print(f"数据目录: {DATA_DIR}")
    
    # 1. 财务数据
    fetch_financial_data()
    
    # 2. 资金流数据
    fetch_money_flow_data()
    
    # 3. 分析师数据
    fetch_analyst_data()
    
    # 4. 宏观数据
    fetch_macro_data()
    
    # 5. 行业分类
    fetch_industry_classification()
    
    print("\n" + "="*60)
    print("数据获取完成!")
    print("="*60)
    print(f"结束时间: {datetime.now()}")
    
    # 列出所有获取的数据
    print("\n已获取数据文件:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size:.2f} MB")


if __name__ == "__main__":
    main()
