#!/usr/bin/env python3
"""标准入口：python run_experiments.py；离线运行已归档数据。"""
import os
# 控制小矩阵多右端项求解的线程开销，同时固定复现环境。
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import io
import json
import logging
import platform
from pathlib import Path
import sys
import time
import unittest
import numpy as np
import scipy
import matplotlib
from experiments import calibration, entropy_experiment, collinearity
from real_data import optional_real
from plots import make_plots

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def validate_config(cfg):
    if cfg['ntrain'] != 200 or cfg['horizon'] not in (10, 20):
        raise ValueError('课程实验要求ntrain=200, horizon=10或20')
    if cfg['M'] < 300 or cfg['B'] < 999 or cfg['collinearity_repeats'] < 200:
        raise ValueError('校准至少M=300、B=999，共线性重复至少200')
    if cfg['blocks'] != [1, 5, 10] or cfg['ar_rhos'] != [0.0, 0.6]:
        raise ValueError('必须比较block=1/5/10与AR rho=0/0.6')
    if len(cfg['beta']) != 3 or not 0 < cfg['alpha'] < 1:
        raise ValueError('beta长度或显著性阈值无效')


def main():
    parser = argparse.ArgumentParser(description='三个数学主题的可复现实验')
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(cfg)
    if not (ROOT/'figures').is_dir():
        (ROOT/'figures').mkdir()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(ROOT/'run.log', mode='w', encoding='utf-8'),
                                  logging.StreamHandler()])
    started = time.perf_counter()
    previous_path = ROOT/'results.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else None
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT/'tests.log').write_text(stream.getvalue(), encoding='utf-8')
    if not tests.wasSuccessful():
        raise RuntimeError('单元测试失败，查看tests.log')
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(cfg['seed']).spawn(4)]
    logging.info('单元测试通过，开始合成校准')
    result = {'config': cfg, 'data_type': 'synthetic; optional verified archive separately gated',
              'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                              'matplotlib': matplotlib.__version__, 'platform': platform.platform()},
              'method': {'topics': ['最小二乘回归', '离散分箱Shannon熵', '蒙特卡罗/残差块bootstrap'],
                         'statistic': 'abs(sum(test_y - test_X @ beta_hat))',
                         'bootstrap': '训练OLS残差中心化；非循环moving blocks；联合生成n+h伪误差；每个伪训练样本重新拟合beta。',
                         'conditioning': '合成校准各重复共享同一固定X，概率对随机AR误差与模拟抽样取平均。',
                         'scenarios': '训练完全不变；只改测试段：均值+0.8、标准差x2(方差x4)、单个正极端值+8。',
                         'limits': ['有限训练样本及固定短块使bootstrap近似而非精确检验；+1修正不自动保证有效性。',
                                    'block=5/10截断长滞后，拼块接缝破坏依赖；ntrain为块长倍数，训练/测试边界恰是接缝。',
                                    '未做AR模型拟合或条件于最后残差的预测；不保证给定训练历史的条件显著性。',
                                    'T为累计残差的绝对值，针对累计均值偏移，对方差变化或正负抵消的异常并非专门检验。',
                                    '不同方法在每次重复共用基础误差，误报率差必须按配对拒绝指示量分析；不能把两组边际Wilson区间当作独立比例差区间。',
                                    '不同场景共用同一误差与bootstrap零分布，检出率估计相关；Wilson为各率的边际区间，不校正多重比较。',
                                    '报告的功效未经size-adjustment；不同方法误报率不相等，检出率不能直接解释为同等实际size下的功效优劣。',
                                    '真实训练200与测试20的经验分箱熵受样本量偏差影响，仅为描述，不作为熵显著性证据。',
                                    '每个p值MC标准误是独立抽取块条件下的插值近似，不包含残差池估计误差。',
                                    '共线性测试分布与训练一致，低预测误差不保证分布外预测稳定。']}}
    result['bootstrap'] = calibration(cfg, streams[0])
    result['entropy'] = entropy_experiment(cfg, streams[1])
    result['collinearity'] = collinearity(cfg, streams[2])
    result['real_data'] = optional_real(cfg, ROOT, streams[3])
    result['figures'] = make_plots(result, ROOT)
    result['elapsed_seconds'] = time.perf_counter()-started
    write_json(ROOT/'results.json', result)
    finite = all(np.isfinite(r['rejection_rate']) for r in result['bootstrap']['records'])
    col = result['collinearity']
    numeric_keys = ('config', 'bootstrap', 'entropy', 'collinearity', 'real_data')
    fingerprint = hashlib.sha256(json.dumps({k: result[k] for k in numeric_keys},
                                            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    comparable = (previous is not None and previous.get('config') == cfg
                  and previous.get('environment') == result['environment']
                  and previous.get('real_data', {}).get('input_sha256') == result['real_data'].get('input_sha256')
                  and previous.get('real_data', {}).get('status') == result['real_data']['status'])
    repeat_match = all(previous.get(k) == result[k] for k in numeric_keys) if comparable else None
    core_match = None
    if previous is not None and previous.get('config') == cfg:
        core_match = (all(previous.get('bootstrap', {}).get(k) == result['bootstrap'][k]
                          for k in ('fixed_design', 'records', 'illustration', 'mc_se_at_alpha', 'resolution'))
                      and previous.get('entropy') == result['entropy']
                      and previous.get('collinearity') == result['collinearity'])
    validation = {'unittest': {'success': tests.wasSuccessful(), 'run': tests.testsRun,
                               'failures': len(tests.failures), 'errors': len(tests.errors), 'log': 'tests.log'},
                  'checks': {'all_rates_finite': finite,
                             'calibration_runs': sum(r['M'] for r in result['bootstrap']['records'])//4,
                             'orthogonality_max': max(r['max_abs_Xt_residual'] for r in col),
                             'sse_decomposition_max': max(r['max_sse_decomposition_error'] for r in col),
                             'all_figure_files_nonempty': all((ROOT/p).stat().st_size > 1000 for p in result['figures']),
                             'no_network_required': True},
                  'core_synthetic_match_previous_same_config': core_match,
                  'analytic_variance_reference': result['bootstrap']['analytic_variance_reference'],
                  'data_availability': result['real_data'].get('data_availability'),
                  'numerical_fingerprint_sha256': fingerprint,
                  'repeat_numerical_match_previous_same_environment_config_input': repeat_match,
                  'real_data_status': result['real_data']['status'],
                  'reproducibility': '固定种子、4条独立随机流、固定配置。数值结果在同环境可重现；日志时间、耗时、SVG元数据与哈希不保证逐字节一致。'}
    if not finite or not validation['checks']['all_figure_files_nonempty']:
        raise RuntimeError('产物检查失败')
    write_json(ROOT/'validation.json', validation)
    logging.info('完成，耗时 %.2f秒，真实数据状态=%s', result['elapsed_seconds'], result['real_data']['status'])
    # 日志关闭后哈希；清单不包含自身、虚拟环境和字节码。
    logging.shutdown()
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '.venv' not in p.parts
                   and '.git' not in p.parts and '__pycache__' not in p.parts
                   and p.name not in ('manifest.sha256', '.DS_Store'))
    manifest = ''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files)
    (ROOT/'manifest.sha256').write_text(manifest, encoding='utf-8')


if __name__ == '__main__':
    main()
