"""
策略研究编排器 - 标准化研究流程

整合:
1. 因子计算
2. 健康检查
3. 增量分析
4. 冗余压缩
5. 组合构建
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
from datetime import datetime


@dataclass
class ResearchConfig:
    """研究配置"""
    data_path: str
    output_path: str
    min_ic: float = 0.01
    min_ir: float = 0.05
    max_turnover: float = 0.5
    cost_bps: float = 15.0
    test_period: str = "2017-2026"


@dataclass
class ResearchResult:
    """研究结果"""
    run_id: str
    timestamp: str
    config: dict
    
    # 因子筛选结果
    raw_pool_size: int = 0
    health_check_passed: int = 0
    conditional_factors: list = None
    rejected_factors: list = None
    
    # 增量结果
    incremental_factors: list = None
    r_squared: float = 0.0
    
    # 冗余分析
    redundant_pairs: list = None
    compressed_pool: list = None
    
    # 综合评分
    composite_score: float = 0.0
    
    def to_dict(self):
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            **self.config,
            'raw_pool_size': self.raw_pool_size,
            'health_check_passed': self.health_check_passed,
            'conditional_factors': self.conditional_factors,
            'incremental_factors': self.incremental_factors,
            'r_squared': self.r_squared,
            'compressed_pool': self.compressed_pool,
            'composite_score': self.composite_score,
        }


class ResearchOrchestrator:
    """研究编排器"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.run_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timestamp = datetime.now().isoformat()
        self.results = None
    
    def run_full_pipeline(self) -> ResearchResult:
        """运行完整研究流程"""
        print("=" * 80)
        print(f"策略研究编排器 - {self.run_id}")
        print("=" * 80)
        
        # Step 1: 加载数据
        print("\n【Step 1】加载数据...")
        bars, labels = self._load_data()
        
        # Step 2: 计算因子
        print("\n【Step 2】计算候选因子...")
        factor_panel = self._calculate_factors(bars)
        
        # Step 3: 健康检查
        print("\n【Step 3】单因子健康检查...")
        health_results = self._run_health_check(factor_panel, labels)
        
        # Step 4: 增量分析
        print("\n【Step 4】因子增量分析...")
        incremental_results = self._run_incremental_analysis(factor_panel, labels)
        
        # Step 5: 冗余压缩
        print("\n【Step 5】冗余分析...")
        compression_results = self._run_redundancy_analysis(factor_panel)
        
        # Step 6: 生成最终池
        print("\n【Step 6】生成最终研究池...")
        final_pool = self._generate_final_pool(
            health_results, 
            incremental_results, 
            compression_results
        )
        
        # Step 7: 保存结果
        print("\n【Step 7】保存结果...")
        self._save_results(
            bars, labels, factor_panel, 
            health_results, incremental_results, 
            compression_results, final_pool
        )
        
        # 构建结果对象
        self.results = ResearchResult(
            run_id=self.run_id,
            timestamp=self.timestamp,
            config={
                'min_ic': self.config.min_ic,
                'min_ir': self.config.min_ir,
                'max_turnover': self.config.max_turnover,
                'cost_bps': self.config.cost_bps,
            },
            raw_pool_size=len(factor_panel.columns) - 2,
            health_check_passed=len([r for r in health_results if r['status'] == 'pass']),
            conditional_factors=[r['factor'] for r in health_results if r['status'] == 'conditional'],
            rejected_factors=[r['factor'] for r in health_results if r['status'] == 'fail'],
            incremental_factors=incremental_results['selected'],
            r_squared=incremental_results['r_squared'],
            redundant_pairs=compression_results['pairs'],
            compressed_pool=final_pool,
            composite_score=self._calculate_composite_score(
                health_results, incremental_results, compression_results
            ),
        )
        
        print("\n" + "=" * 80)
        print("研究完成!")
        print("=" * 80)
        
        return self.results
    
    def _load_data(self):
        """加载数据"""
        bars = pd.read_parquet(self.config.data_path)
        labels_path = Path(self.config.data_path).parent / "labels" / "label_panel.parquet"
        if labels_path.exists():
            labels = pd.read_parquet(labels_path)
        else:
            labels = self._generate_labels(bars)
        return bars, labels
    
    def _generate_labels(self, bars: pd.DataFrame) -> pd.DataFrame:
        """生成标签"""
        bars = bars.sort_values(['symbol', 'trade_date'])
        bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].shift(-20)
        bars['fwd_return_20d'] = bars.groupby('symbol')['fwd_return_20d'].pct_change()
        return bars[['trade_date', 'symbol', 'fwd_return_20d']].dropna()
    
    def _calculate_factors(self, bars: pd.DataFrame) -> pd.DataFrame:
        """计算候选因子"""
        from src.features.factor_calculator import FactorCalculator
        
        calc = FactorCalculator(bars)
        
        # 核心候选因子
        factor_configs = [
            ('mom250', 'mom', {'window': 250}),
            ('mom120', 'mom', {'window': 120}),
            ('mom60', 'mom', {'window': 60}),
            ('mom20', 'mom', {'window': 20}),
            ('close_to_high250', 'close_to_high', {'window': 250}),
            ('close_to_high120', 'close_to_high', {'window': 120}),
            ('close_to_high60', 'close_to_high', {'window': 60}),
            ('Size', 'size', {}),
            ('vol120', 'vol', {'window': 120}),
            ('vol60', 'vol', {'window': 60}),
            ('vol20', 'vol', {'window': 20}),
            ('rsi6', 'rsi', {'window': 6}),
            ('rsi14', 'rsi', {'window': 14}),
            ('rev20', 'reversal', {'window': 20}),
            ('rev5', 'reversal', {'window': 5}),
            ('high_low_pos120', 'high_low_pos', {'window': 120}),
            ('high_low_pos60', 'high_low_pos', {'window': 60}),
            ('ma_diff_5_20', 'ma_diff', {'short': 5, 'long': 20}),
            ('amount_growth20', 'amount_growth', {'window': 20}),
            ('turnover_rate20', 'turnover_rate', {'window': 20}),
            ('vol_ratio_5_20', 'vol_ratio', {'short': 5, 'long': 20}),
            ('price_to_ma60', 'price_to_ma', {'window': 60}),
            ('candle_body_ratio', 'candle_body_ratio', {}),
        ]
        
        panel = bars[['trade_date', 'symbol']].copy()
        
        for name, method, kwargs in factor_configs:
            try:
                df = calc.calculate(method, **kwargs)
                df = df.rename(columns={'value': name})
                panel = panel.merge(df, on=['trade_date', 'symbol'], how='left')
            except Exception as e:
                print(f"   {name}: FAILED")
        
        return panel
    
    def _run_health_check(
        self, 
        panel: pd.DataFrame, 
        labels: pd.DataFrame
    ) -> list[dict]:
        """运行健康检查"""
        from src.features.factor_health_check import batch_check
        
        panel = panel.merge(labels, on=['trade_date', 'symbol'], how='left')
        
        factor_names = [c for c in panel.columns 
                       if c not in ['trade_date', 'symbol', 'fwd_return_20d']]
        
        reports = batch_check(panel, factor_names)
        
        results = []
        for r in reports:
            results.append({
                'factor': r.name,
                'status': r.health_status.value,
                'ic_mean': r.ic_mean,
                'ic_ir': r.ic_ir,
                'monotonicity': r.monotonicity_score,
                'net_return_30bp': r.net_return_30bp,
                'avg_turnover': r.avg_turnover,
                'reasons': r.reasons_to_reject,
            })
        
        return results
    
    def _run_incremental_analysis(
        self, 
        panel: pd.DataFrame, 
        labels: pd.DataFrame
    ) -> dict:
        """增量分析"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        panel = panel.merge(labels, on=['trade_date', 'symbol'], how='left')
        
        factor_names = [c for c in panel.columns 
                       if c not in ['trade_date', 'symbol', 'fwd_return_20d']]
        
        # 清理数据
        for col in factor_names + ['fwd_return_20d']:
            if col in panel.columns:
                panel[col] = panel[col].replace([np.inf, -np.inf], np.nan)
        
        valid = panel.dropna(subset=['fwd_return_20d'])
        
        X = valid[factor_names].fillna(0).values
        y = valid['fwd_return_20d'].values
        
        # 计算IC
        ic_results = {}
        for i, name in enumerate(factor_names):
            ic = pd.Series(X[:, i]).corr(pd.Series(y), method='spearman')
            ic_results[name] = ic
        
        # 按IC排序
        sorted_factors = sorted(ic_results.items(), key=lambda x: x[1], reverse=True)
        positive_factors = [f[0] for f in sorted_factors if f[1] > 0]
        
        # 逐步回归
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        r_squared = lr.score(X_scaled, y)
        
        return {
            'ic_results': ic_results,
            'sorted_by_ic': sorted_factors,
            'positive_factors': positive_factors,
            'r_squared': r_squared,
            'selected': positive_factors[:5],  # 选择前5个正向因子
        }
    
    def _run_redundancy_analysis(self, panel: pd.DataFrame) -> dict:
        """冗余分析"""
        factor_names = [c for c in panel.columns 
                       if c not in ['trade_date', 'symbol']]
        
        # 清理数据
        for col in factor_names:
            if col in panel.columns:
                panel[col] = panel[col].replace([np.inf, -np.inf], np.nan)
        
        corr = panel[factor_names].corr()
        
        high_corr_pairs = []
        for i in range(len(factor_names)):
            for j in range(i+1, len(factor_names)):
                if abs(corr.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'factor1': factor_names[i],
                        'factor2': factor_names[j],
                        'correlation': corr.iloc[i, j],
                    })
        
        return {
            'correlation_matrix': corr,
            'pairs': high_corr_pairs,
        }
    
    def _generate_final_pool(
        self,
        health_results: list,
        incremental_results: dict,
        compression_results: dict
    ) -> list:
        """生成最终研究池"""
        # 从健康检查获取条件通过的
        conditional = [r['factor'] for r in health_results if r['status'] == 'conditional']
        
        # 添加增量分析中有效的
        selected = incremental_results['selected']
        
        # 合并去重
        final_pool = list(set(conditional + selected))
        
        # 排除高冗余的
        high_corr = set()
        for pair in compression_results['pairs']:
            if pair['correlation'] > 0.9:
                # 保留IC更高的
                ic1 = incremental_results['ic_results'].get(pair['factor1'], 0)
                ic2 = incremental_results['ic_results'].get(pair['factor2'], 0)
                if ic1 < ic2:
                    high_corr.add(pair['factor1'])
                else:
                    high_corr.add(pair['factor2'])
        
        final_pool = [f for f in final_pool if f not in high_corr]
        
        return final_pool
    
    def _calculate_composite_score(
        self,
        health_results: list,
        incremental_results: dict,
        compression_results: dict
    ) -> float:
        """计算综合评分"""
        # 基于有效因子数量、IC质量、冗余度
        passed = len([r for r in health_results if r['status'] in ['pass', 'conditional']])
        avg_ic = np.mean([r['ic_mean'] for r in health_results if r['ic_mean'] > 0])
        redundancy_penalty = len(compression_results['pairs']) * 0.1
        
        score = passed * 10 + avg_ic * 100 - redundancy_penalty
        return score
    
    def _save_results(
        self,
        bars: pd.DataFrame,
        labels: pd.DataFrame,
        factor_panel: pd.DataFrame,
        health_results: list,
        incremental_results: dict,
        compression_results: dict,
        final_pool: list
    ):
        """保存结果"""
        output_dir = Path(self.config.output_path) / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始数据
        factor_panel.to_parquet(output_dir / "factor_panel.parquet", index=False)
        
        # 保存健康检查结果
        pd.DataFrame(health_results).to_csv(
            output_dir / "health_check.csv", index=False
        )
        
        # 保存增量分析结果
        pd.DataFrame(incremental_results['sorted_by_ic'], columns=['factor', 'ic']).to_csv(
            output_dir / "incremental_analysis.csv", index=False
        )
        
        # 保存最终池
        pd.DataFrame({'factor': final_pool}).to_csv(
            output_dir / "final_pool.csv", index=False
        )
        
        # 保存元数据
        meta = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'final_pool_size': len(final_pool),
            'final_pool': final_pool,
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"   结果已保存到: {output_dir}")


def run_research(config: Optional[ResearchConfig] = None):
    """便捷运行函数"""
    if config is None:
        config = ResearchConfig(
            data_path='data/bronze/daily_bar.parquet',
            output_path='artifacts/research',
        )
    
    orchestrator = ResearchOrchestrator(config)
    return orchestrator.run_full_pipeline()


if __name__ == "__main__":
    run_research()
