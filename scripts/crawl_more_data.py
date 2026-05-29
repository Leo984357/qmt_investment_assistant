"""
爬取更多数据以覆盖剩余493个因子
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data/raw/fetched_data')

print("="*70)
print("爬取更多数据")
print("="*70)

# =============================================================================
# 1. 爬取扩展财务数据
# =============================================================================
print("\n[1] 爬取扩展财务数据...")

try:
    import akshare as ak
    
    # 股票列表
    stock_info = ak.stock_info_a_code_name()
    main_stocks = stock_info[stock_info['code'].str.startswith(('6', '000', '002', '300'))]['code'].tolist()[:100]
    
    all_financial = []
    
    for i, code in enumerate(main_stocks[:50]):  # 限制数量
        symbol = code + ('.SH' if code.startswith('6') else '.SZ')
        try:
            # 财务指标
            df = ak.stock_financial_analysis_indicator(symbol)
            if len(df) > 0:
                df['code'] = code
                df['symbol'] = symbol
                all_financial.append(df)
            time.sleep(0.2)
        except Exception as e:
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/50")
    
    if all_financial:
        fin_df = pd.concat(all_financial, ignore_index=True)
        fin_df.to_parquet(DATA_DIR / 'akshare_extended_financial.parquet', index=False)
        print(f"  保存 {len(fin_df)} 行扩展财务数据")
        
except Exception as e:
    print(f"  失败: {e}")

# =============================================================================
# 2. 爬取分析师预测数据
# =============================================================================
print("\n[2] 爬取分析师预测数据...")

try:
    import akshare as ak
    
    # 获取盈利预测
    try:
        forecast_df = ak.stock_profit_forecast(indicator="每股收益")
        if len(forecast_df) > 0:
            forecast_df.to_parquet(DATA_DIR / 'akshare_profit_forecast.parquet', index=False)
            print(f"  盈利预测: {len(forecast_df)} 行")
    except:
        pass
    
    # 评级数据
    try:
        rating_df = ak.stock_rating_summary(symbol="沪深300")
        if len(rating_df) > 0:
            rating_df.to_parquet(DATA_DIR / 'akshare_rating.parquet', index=False)
            print(f"  评级数据: {len(rating_df)} 行")
    except:
        pass
    
    # 研报数据
    try:
        report_df = ak.stock_research_report(symbol="沪深300")
        if len(report_df) > 0:
            report_df.to_parquet(DATA_DIR / 'akshare_research_report.parquet', index=False)
            print(f"  研报数据: {len(report_df)} 行")
    except:
        pass
        
except Exception as e:
    print(f"  失败: {e}")

# =============================================================================
# 3. 爬取资金流详细数据
# =============================================================================
print("\n[3] 爬取资金流详细数据...")

try:
    import akshare as ak
    
    # 大单资金流
    try:
        for code in ['000001', '600000', '600519'][:3]:
            mf = ak.stock_individual_fund_flow(symbol=code)
            if len(mf) > 0:
                mf.to_parquet(DATA_DIR / f'akshare_moneyflow_{code}.parquet', index=False)
                print(f"  {code}资金流: {len(mf)} 行")
            time.sleep(0.5)
    except:
        pass
    
    # 板块资金流
    try:
        sector_mf = ak.stock_sector_fund_flow_rank(indicator="今日")
        if len(sector_mf) > 0:
            sector_mf.to_parquet(DATA_DIR / 'akshare_sector_moneyflow.parquet', index=False)
            print(f"  板块资金流: {len(sector_mf)} 行")
    except:
        pass
        
except Exception as e:
    print(f"  失败: {e}")

# =============================================================================
# 4. 爬取情绪数据
# =============================================================================
print("\n[4] 爬取情绪数据...")

try:
    import akshare as ak
    
    # 涨跌停数据
    try:
        zt_df = ak.stock_zt_pool_strong(date=pd.Timestamp.now().strftime('%Y%m%d'))
        if len(zt_df) > 0:
            zt_df.to_parquet(DATA_DIR / 'akshare_zt_pool_today.parquet', index=False)
            print(f"  涨停池: {len(zt_df)} 行")
    except:
        pass
    
    # 强势股
    try:
        strong_df = ak.stock_zt_pool_strong(date=pd.Timestamp.now().strftime('%Y%m%d'))
        if len(strong_df) > 0:
            strong_df.to_parquet(DATA_DIR / 'akshare_strong_stocks.parquet', index=False)
            print(f"  强势股: {len(strong_df)} 行")
    except:
        pass
        
    # 概念板块
    try:
        concept = ak.stock_board_concept_name()
        concept.to_parquet(DATA_DIR / 'akshare_concept_board.parquet', index=False)
        print(f"  概念板块: {len(concept)} 行")
    except:
        pass
        
except Exception as e:
    print(f"  失败: {e}")

# =============================================================================
# 5. 爬取宏观数据
# =============================================================================
print("\n[5] 爬取宏观数据...")

try:
    import akshare as ak
    
    # 国债收益率
    try:
        bond = ak.bond_zh_hs yield(start_date="20180101", end_date=pd.Timestamp.now().strftime('%Y%m%d'))
        if len(bond) > 0:
            bond.to_parquet(DATA_DIR / 'akshare_bond_yield.parquet', index=False)
            print(f"  国债收益率: {len(bond)} 行")
    except Exception as e:
        print(f"  国债收益率失败: {e}")
    
    # 社融数据
    try:
        m2 = ak.macro_china_money_supply()
        if len(m2) > 0:
            m2.to_parquet(DATA_DIR / 'akshare_money_supply.parquet', index=False)
            print(f"  货币供应: {len(m2)} 行")
    except:
        pass
        
except Exception as e:
    print(f"  失败: {e}")

print("\n" + "="*70)
print("数据爬取完成!")
print("="*70)
