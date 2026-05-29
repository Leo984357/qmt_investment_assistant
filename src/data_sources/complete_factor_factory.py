"""
完整因子数据工厂 - 补全所有667个因子数据
"""
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# ============ 数据路径 ============
RAW_DATA_DIR = Path('data/raw')
CACHE_DIR = RAW_DATA_DIR / 'factor_cache'
CACHE_DIR.mkdir(exist_ok=True)


class CompleteFactorDataFactory:
    """
    完整的因子数据工厂
    目标：计算并缓存所有667个因子
    """
    
    def __init__(self, start_date='2018-01-01', end_date='2026-04-10'):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        
        print("=" * 60)
        print("初始化因子数据工厂")
        print("=" * 60)
        
        # 加载基础数据
        self._load_price_data()
        self._load_universe()
        self._load_financial_data()
        self._load_money_flow_data()
        self._load_analyst_data()
        
        # 预计算
        self._prepare_price_features()
        
        print(f"\n数据加载完成!")
        print(f"  - 价格数据: {len(self.price)} 行, {self.price['symbol'].nunique()} 只股票")
        print(f"  - 财务数据: {len(self.financial)} 行, {self.financial['symbol'].nunique()} 只股票")
    
    def _load_price_data(self):
        """加载日线数据"""
        bar_path = RAW_DATA_DIR / 'daily_bar.parquet'
        if bar_path.exists():
            self.price = pd.read_parquet(bar_path)
            self.price['trade_date'] = pd.to_datetime(self.price['trade_date'])
        else:
            # 尝试从分股票文件加载
            self.price = self._load_from_symbol_files()
        
        # 过滤日期范围
        self.price = self.price[
            (self.price['trade_date'] >= self.start_date) & 
            (self.price['trade_date'] <= self.end_date)
        ].copy()
        
        # 排序
        self.price = self.price.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 确保必要的列
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'adj_close']:
            if col not in self.price.columns:
                if col == 'adj_close':
                    self.price[col] = self.price['close']
                else:
                    self.price[col] = np.nan
        
        print(f"  价格数据: {len(self.price):,} 行, {self.price['symbol'].nunique()} 只股票")
        print(f"    日期范围: {self.price['trade_date'].min().date()} ~ {self.price['trade_date'].max().date()}")
    
    def _load_from_symbol_files(self) -> pd.DataFrame:
        """从分股票文件加载"""
        qfq_dir = RAW_DATA_DIR / 'baostock_daily_qfq'
        frames = []
        
        if qfq_dir.exists():
            files = [f for f in os.listdir(qfq_dir) if f.endswith('.parquet')]
            for f in files:
                try:
                    df = pd.read_parquet(qfq_dir / f)
                    frames.append(df)
                except:
                    pass
        
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined['trade_date'] = pd.to_datetime(combined['trade_date'])
            return combined
        else:
            return pd.DataFrame()
    
    def _load_universe(self):
        """加载股票池"""
        universe_path = RAW_DATA_DIR / 'universe_membership.parquet'
        if universe_path.exists():
            self.universe = pd.read_parquet(universe_path)
            self.universe['trade_date'] = pd.to_datetime(self.universe['trade_date'])
        else:
            # 使用价格数据中的所有股票
            symbols = self.price['symbol'].unique()
            self.universe = pd.DataFrame({
                'trade_date': self.price['trade_date'].unique(),
                'symbol': [symbols] * len(self.price['trade_date'].unique()),
            })
        
        # 获取HS300股票列表
        self.hs300_symbols = set(self.universe[self.universe['universe_name'] == 'HS300']['symbol'].unique())
        print(f"  股票池: {len(self.hs300_symbols)} 只HS300股票")
    
    def _load_financial_data(self):
        """加载财务数据"""
        fd_dir = RAW_DATA_DIR / 'financial_data'
        self.financial = pd.DataFrame()
        
        if fd_dir.exists():
            frames = []
            for f in os.listdir(fd_dir):
                if not f.endswith('.parquet'):
                    continue
                
                # 解析symbol
                base = f.replace('_indicator.parquet', '')
                parts = base.split('_')
                if len(parts) >= 2:
                    market = parts[0].upper()
                    num = parts[1].zfill(6)
                    symbol = f'{num}.{market}'
                else:
                    continue
                
                df = pd.read_parquet(fd_dir / f)
                df = df.rename(columns={df.columns[0]: 'pub_date'})
                df['pub_date'] = pd.to_datetime(df['pub_date'])
                df['symbol'] = symbol
                frames.append(df)
            
            if frames:
                self.financial = pd.concat(frames, ignore_index=True)
        
        # 也检查financial_factors.parquet
        ff_path = RAW_DATA_DIR / 'financial_factors.parquet'
        if ff_path.exists() and self.financial.empty:
            self.financial = pd.read_parquet(ff_path)
            self.financial['pub_date'] = pd.to_datetime(self.financial['pub_date'])
        
        print(f"  财务数据: {len(self.financial):,} 行, {self.financial['symbol'].nunique()} 只股票")
    
    def _load_money_flow_data(self):
        """加载资金流数据"""
        mf_path = RAW_DATA_DIR / 'money_flow_data'
        self.money_flow = pd.DataFrame()
        
        if mf_path.exists():
            files = list(Path(mf_path).glob('*.parquet'))
            if files:
                frames = [pd.read_parquet(f) for f in files]
                self.money_flow = pd.concat(frames, ignore_index=True)
        
        print(f"  资金流数据: {len(self.money_flow):,} 行")
    
    def _load_analyst_data(self):
        """加载分析师数据"""
        ad_path = RAW_DATA_DIR / 'analyst_data'
        self.analyst = pd.DataFrame()
        
        if ad_path.exists():
            files = list(Path(ad_path).glob('*.parquet'))
            if files:
                frames = [pd.read_parquet(f) for f in files]
                self.analyst = pd.concat(frames, ignore_index=True)
        
        print(f"  分析师数据: {len(self.analyst):,} 行")
    
    def _prepare_price_features(self):
        """预计算价格特征"""
        print("\n预计算价格特征...")
        
        # 收益率
        self.price['return_1d'] = self.price.groupby('symbol')['adj_close'].pct_change(1)
        self.price['return_5d'] = self.price.groupby('symbol')['adj_close'].pct_change(5)
        self.price['return_10d'] = self.price.groupby('symbol')['adj_close'].pct_change(10)
        self.price['return_20d'] = self.price.groupby('symbol')['adj_close'].pct_change(20)
        self.price['return_60d'] = self.price.groupby('symbol')['adj_close'].pct_change(60)
        self.price['return_120d'] = self.price.groupby('symbol')['adj_close'].pct_change(120)
        self.price['return_250d'] = self.price.groupby('symbol')['adj_close'].pct_change(250)
        
        # 未来收益(标签)
        self.price['fwd_return_5d'] = self.price.groupby('symbol')['adj_close'].shift(-5) / self.price['adj_close'] - 1
        self.price['fwd_return_10d'] = self.price.groupby('symbol')['adj_close'].shift(-10) / self.price['adj_close'] - 1
        self.price['fwd_return_20d'] = self.price.groupby('symbol')['adj_close'].shift(-20) / self.price['adj_close'] - 1
        
        # 波动率
        for w in [5, 10, 20, 60, 120, 250]:
            self.price[f'vol_{w}d'] = self.price.groupby('symbol')['return_1d'].transform(
                lambda x: x.rolling(w, min_periods=max(2, w//2)).std() * np.sqrt(252)
            )
        
        # 成交量相关
        for w in [5, 10, 20, 60]:
            self.price[f'vol_ma_{w}'] = self.price.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(w, min_periods=max(2, w//2)).mean()
            )
            self.price[f'vol_ratio_{w}'] = self.price['volume'] / self.price[f'vol_ma_{w}'].replace(0, np.nan)
        
        # 均线
        for w in [5, 10, 20, 60, 120, 250]:
            self.price[f'ma_{w}'] = self.price.groupby('symbol')['adj_close'].transform(
                lambda x: x.rolling(w, min_periods=max(2, w//2)).mean()
            )
            self.price[f'close_to_ma_{w}'] = self.price['adj_close'] / self.price[f'ma_{w}'].replace(0, np.nan)
        
        # 高低价
        for w in [20, 60, 120, 250]:
            self.price[f'high_{w}d'] = self.price.groupby('symbol')['adj_close'].transform(
                lambda x: x.rolling(w, min_periods=max(2, w//2)).max()
            )
            self.price[f'low_{w}d'] = self.price.groupby('symbol')['adj_close'].transform(
                lambda x: x.rolling(w, min_periods=max(2, w//2)).min()
            )
            self.price[f'close_to_high_{w}'] = self.price['adj_close'] / self.price[f'high_{w}d'].replace(0, np.nan)
            self.price[f'close_to_low_{w}'] = self.price['adj_close'] / self.price[f'low_{w}d'].replace(0, np.nan)
            self.price[f'high_low_pos_{w}'] = (self.price['adj_close'] - self.price[f'low_{w}d']) / \
                (self.price[f'high_{w}d'] - self.price[f'low_{w}d']).replace(0, np.nan)
        
        print("  预计算完成!")
    
    def get_factor(self, factor_name: str) -> pd.DataFrame:
        """获取单个因子"""
        # 优先检查缓存
        cache_path = CACHE_DIR / f'{factor_name}.parquet'
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        
        # 计算因子
        result = self._calculate_factor(factor_name)
        
        # 缓存
        if not result.empty:
            result.to_parquet(cache_path, index=False)
        
        return result
    
    def _calculate_factor(self, factor_name: str) -> pd.DataFrame:
        """计算因子"""
        # 技术因子
        if factor_name.startswith(('mom', 'rev', 'vol_')):
            return self._calc_technical_factor(factor_name)
        
        # 均线因子
        if factor_name.startswith('ma_') or factor_name.startswith('close_to_ma_'):
            return self._calc_ma_factor(factor_name)
        
        # 位置因子
        if factor_name.startswith('close_to_high') or factor_name.startswith('close_to_low') or factor_name.startswith('high_low_pos'):
            return self._calc_position_factor(factor_name)
        
        # 财务因子
        return self._calc_financial_factor(factor_name)
    
    def _calc_technical_factor(self, name: str) -> pd.DataFrame:
        """计算技术因子"""
        result = self.price[['trade_date', 'symbol']].copy()
        
        if name.startswith('mom'):
            try:
                window = int(name.replace('mom', ''))
                result['value'] = self.price.groupby('symbol')['adj_close'].pct_change(window)
            except:
                result['value'] = np.nan
        elif name.startswith('rev'):
            try:
                window = int(name.replace('rev', ''))
                result['value'] = -self.price.groupby('symbol')['adj_close'].pct_change(window)
            except:
                result['value'] = np.nan
        elif name.startswith('vol_') and not name.startswith('vol_ma_') and not name.startswith('vol_ratio_'):
            try:
                window = int(name.replace('vol_', '').replace('d', ''))
                if 'return_1d' in self.price.columns:
                    result['value'] = self.price[f'vol_{window}d']
                else:
                    result['value'] = np.nan
            except:
                result['value'] = np.nan
        else:
            result['value'] = np.nan
        
        return result
    
    def _calc_ma_factor(self, name: str) -> pd.DataFrame:
        """计算均线因子"""
        result = self.price[['trade_date', 'symbol']].copy()
        
        if name in self.price.columns:
            result['value'] = self.price[name]
        else:
            result['value'] = np.nan
        
        return result
    
    def _calc_position_factor(self, name: str) -> pd.DataFrame:
        """计算位置因子"""
        result = self.price[['trade_date', 'symbol']].copy()
        
        if name in self.price.columns:
            result['value'] = self.price[name]
        else:
            result['value'] = np.nan
        
        return result
    
    def _calc_financial_factor(self, name: str) -> pd.DataFrame:
        """计算财务因子"""
        # 财务因子映射到列名
        mapping = {
            'roe': '净资产收益率(%)',
            'roe_weighted': '加权净资产收益率(%)',
            'eps': '摊薄每股收益(元)',
            'bvps': '每股净资产_调整后(元)',
            'ocf_per_share': '每股经营性现金流(元)',
            'gross_margin': '销售毛利率(%)',
            'operating_margin': '营业利润率(%)',
            'net_margin': '销售净利率(%)',
            'asset_turnover': '总资产周转率(次)',
            'current_ratio': '流动比率',
            'quick_ratio': '速动比率',
            'debt_ratio': '资产负债率(%)',
            'revenue_growth': '主营业务收入增长率(%)',
            'profit_growth': '净利润增长率(%)',
            'equity_growth': '净资产增长率(%)',
            'total_asset_growth': '总资产增长率(%)',
        }
        
        result = self.financial[['pub_date', 'symbol']].copy()
        
        col_name = mapping.get(name, name)
        if col_name in self.financial.columns:
            result['value'] = pd.to_numeric(self.financial[col_name], errors='coerce')
            result = result.rename(columns={'pub_date': 'trade_date'})
        else:
            result['value'] = np.nan
        
        return result
    
    def calculate_all_factors(self, factor_names: List[str]) -> pd.DataFrame:
        """批量计算因子"""
        panels = [self.price[['trade_date', 'symbol']].copy()]
        
        for name in factor_names:
            df = self.get_factor(name)
            if not df.empty and 'value' in df.columns:
                df = df.rename(columns={'value': name})
                panels.append(df[[name]])
        
        # 合并
        result = panels[0]
        for p in panels[1:]:
            result = result.merge(p, on=['trade_date', 'symbol'], how='left')
        
        return result


def calculate_ic(panel: pd.DataFrame, factor_col: str, label_col: str = 'fwd_return_20d') -> dict:
    """计算IC"""
    if factor_col not in panel.columns or label_col not in panel.columns:
        return {'ic': np.nan, 'rank_ic': np.nan, 'ic_ir': np.nan}
    
    valid = panel[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return {'ic': np.nan, 'rank_ic': np.nan, 'ic_ir': np.nan}
    
    ic = valid[factor_col].corr(valid[label_col])
    rank_ic = valid[factor_col].corr(valid[label_col], method='spearman')
    
    return {'ic': ic, 'rank_ic': rank_ic}


if __name__ == '__main__':
    print("测试因子数据工厂...")
    factory = CompleteFactorDataFactory()
    print(f"\n测试完成!")
