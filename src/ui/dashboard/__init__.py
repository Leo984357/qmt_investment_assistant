"""Streamlit research workbench for local experiment artifacts."""

from __future__ import annotations

import streamlit as st

from .charts import *
from .helpers import *
from .pages import *


def main() -> None:
    st.set_page_config(page_title='QMT 研究工作台', layout='wide', initial_sidebar_state='expanded')
    _inject_theme()

    run_dir_values = _list_run_dirs()
    if not run_dir_values:
        st.warning('还没有实验产物。先运行 `python -m src.cli experiment --config configs/experiments/hs300_lightgbm.yaml`。')
        return

    ledger = _build_ledger(tuple(run_dir_values))
    if ledger.empty:
        st.warning('实验目录存在，但还没有可读的 run summary。')
        return

    with st.sidebar:
        st.markdown("### QMT 研究工作台")
        st.caption('本地优先的组合研究平台')
        mode_options = list(WORKSPACE_MODES.keys())
        preferred_mode = st.session_state.get('workspace_mode', 'daily')
        mode_index = mode_options.index(preferred_mode) if preferred_mode in mode_options else 0
        mode_key = st.radio('使用模式', mode_options, index=mode_index, format_func=lambda key: WORKSPACE_MODES[key]['label'])
        st.session_state['workspace_mode'] = mode_key
        st.caption(WORKSPACE_MODES[mode_key]['description'])
        page_candidates = WORKSPACE_MODES[mode_key]['pages']
        preferred_page = st.session_state.get('page_key', page_candidates[0])
        if preferred_page not in page_candidates:
            preferred_page = page_candidates[0]
        page_key = st.radio('工作区', page_candidates, index=page_candidates.index(preferred_page), format_func=lambda key: PAGE_OPTIONS[key])
        st.session_state['page_key'] = page_key
        experiment_options = ['全部'] + sorted(ledger['experiment_name'].dropna().unique().tolist())
        experiment_filter = st.selectbox('实验名称', experiment_options)
        filtered_ledger = ledger if experiment_filter == '全部' else ledger.loc[ledger['experiment_name'] == experiment_filter].copy()
        run_options = filtered_ledger['run_name'].tolist()
        preferred_run_name = st.session_state.get('preferred_run_name')
        default_run_index = run_options.index(preferred_run_name) if preferred_run_name in run_options else 0
        run_name = st.selectbox('运行记录', run_options, index=default_run_index)
        st.session_state['preferred_run_name'] = run_name
        st.markdown('---')
        with st.expander('怎么使用', expanded=False):
            st.markdown(
                '\n'.join(
                    WORKSPACE_MODES[mode_key]['checklist']
                    + [
                        '第一次使用，先执行 `python -m src.cli bootstrap-data --config configs/experiments/hs300_lightgbm.yaml` 初始化真实数据。',
                        '之后重复执行同一命令时，会优先走本地 raw 缓存并增量补日线尾部。',
                        '运行实验：`python -m src.cli experiment --config configs/experiments/hs300_lightgbm.yaml`。',
                        '打开界面：`python -m src.cli dashboard`。',
                    ]
                )
            )
        if st.button('清空页面缓存', use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown('---')
        st.caption('运行一次新实验')
        st.code('python -m src.cli experiment --config configs/experiments/hs300_lightgbm.yaml')
        st.caption('打开研究界面')
        st.code('python -m src.cli dashboard')

    selected_row = ledger.loc[ledger['run_name'] == run_name].iloc[0]
    bundle = _load_run_bundle(selected_row['run_dir'])

    _render_hero(bundle, run_name)

    if page_key == 'overview':
        _render_overview(bundle)
    elif page_key == 'ledger':
        _render_ledger(filtered_ledger, run_name)
    elif page_key == 'data':
        _render_data_status(bundle)
    elif page_key == 'backtest':
        _render_backtest(bundle)
    elif page_key == 'signal':
        _render_signal_desk(bundle)
    elif page_key == 'factorlab':
        _render_factor_lab(bundle)
    elif page_key == 'studio':
        _render_studio(bundle)
    elif page_key == 'artifacts':
        _render_artifacts(bundle)


if __name__ == '__main__':
    main()
