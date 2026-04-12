"""
分析师预期数据获取脚本

数据源：akshare
"""

import akshare as ak
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

print("="*70)
print("分析师预期数据获取")
print("="*70)

OUTPUT_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/raw/analyst_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 1. 券商评级 ====================
print("\n[1/4] 获取券商评级数据...")

try:
    # 获取分析师评级详情
    df = ak.stock_analyst_detail_em()
    print(f"  成功: {len(df)}条")
    
    # 保存
    df.to_parquet(OUTPUT_DIR / "analyst_ratings.parquet")
    print(f"  已保存: analyst_ratings.parquet")
    
    # 显示样本
    print(f"\n  样本数据:")
    print(df.head(3)[['股票代码', '股票名称', '最新评级名称', '阶段涨跌幅']].to_string(index=False))
    
except Exception as e:
    print(f"  失败: {e}")

# ==================== 2. 盈利预测 ====================
print("\n[2/4] 获取盈利预测数据...")

try:
    # 尝试获取盈利预测
    symbols = ['000001', '000002', '600000', '600036']  # 试点几只
    
    all_forecasts = []
    for sym in symbols:
        try:
            df = ak.stock_profit_forecast_em(symbol=sym)
            if df is not None and len(df) > 0:
                df['symbol'] = sym
                all_forecasts.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"  {sym}: {e}")
    
    if all_forecasts:
        forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        print(f"  成功: {len(forecasts_df)}条")
        forecasts_df.to_parquet(OUTPUT_DIR / "profit_forecasts.parquet")
        print(f"  已保存: profit_forecasts.parquet")
    else:
        print("  无数据")
        
except Exception as e:
    print(f"  失败: {e}")

# ==================== 3. 分析师排名 ====================
print("\n[3/4] 获取分析师排名...")

try:
    df = ak.stock_analyst_rank_em(symbol='2024')
    print(f"  成功: {len(df)}条")
    df.to_parquet(OUTPUT_DIR / "analyst_rankings.parquet")
    print(f"  已保存: analyst_rankings.parquet")
except Exception as e:
    print(f"  失败: {e}")

# ==================== 4. 一致预期 ====================
print("\n[4/4] 获取一致预期...")

try:
    # 同花顺盈利预测
    df = ak.stock_profit_forecast_ths(symbol='000001')
    print(f"  成功: {len(df)}条")
    df.to_parquet(OUTPUT_DIR / "consensus_forecast.parquet")
    print(f"  已保存: consensus_forecast.parquet")
except Exception as e:
    print(f"  失败: {e}")

# ==================== 总结 ====================
print("\n" + "="*70)
print("数据获取完成")
print("="*70)

# 列出获取的数据
import os
files = list(OUTPUT_DIR.glob("*.parquet"))
print(f"\n已获取 {len(files)} 个文件:")
for f in files:
    size = os.path.getsize(f) / 1024
    print(f"  - {f.name}: {size:.1f}KB")

print("\n注意: 分析师预期数据覆盖度有限，需扩展股票池")
