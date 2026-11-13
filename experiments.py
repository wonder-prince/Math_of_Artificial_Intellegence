"""合成实验，仅使用明确标识的人工数据。"""
import logging
import numpy as np
from math_core import (fit_ols, design, ar1, bootstrap_null, mc_pvalue,
                       wilson, shannon, binned_entropy, ar1_covariance,
                       ar1_sum_variance, prediction_sum_weights)


def calibration(cfg, rng):
    n, h, b, m = (cfg[k] for k in ('ntrain', 'horizon', 'B', 'M'))
    beta = np.array(cfg['beta'])
    # 全部重复共享同一设计；推断条件于这组训练/测试X。
    x = design(rng, n+h, cfg['design_rho'])
    xt, xf = x[:n], x[n:]
    records, illustration = [], None
    for rho in cfg['ar_rhos']:
        store = {block: {s: [] for s in ('null', 'mean_shift', 'variance_increase', 'outlier')}
                 for block in cfg['blocks']}
        for repeat in range(m):
            e = ar1(rng, n+h, rho)
            yt = xt @ beta + e[:n]
            bhat, resid, _ = fit_ols(xt, yt)
            scenarios = {'null': e[n:].copy(), 'mean_shift': e[n:]+cfg['mean_shift'],
                         'variance_increase': e[n:]*cfg['variance_scale'], 'outlier': e[n:].copy()}
            scenarios['outlier'][h//2] += cfg['outlier_size']
            statistics = {s: float(abs((xf @ beta + noise - xf @ bhat).sum()))
                          for s, noise in scenarios.items()}
            for block in cfg['blocks']:
                null = bootstrap_null(rng, xt, xf, yt, block, b)
                for s, observed in statistics.items():
                    item = mc_pvalue(observed, null)
                    item['statistic'] = observed
                    store[block][s].append(item)
                if rho == 0.6 and repeat == 0 and block == 10:
                    illustration = {'rho': rho, 'block': block, 'null_statistics': null.tolist(),
                                    'observed': statistics, 'training_residuals': resid.tolist(),
                                    'test_noise': {k: v.tolist() for k, v in scenarios.items()}}
            if (repeat+1) % 50 == 0:
                logging.info('校准 rho=%s，已完成 %d/%d', rho, repeat+1, m)
        for block, scenarios in store.items():
            for scenario, values in scenarios.items():
                p = np.array([v['p'] for v in values])
                count = int((p <= cfg['alpha']).sum())
                records.append({'rho': rho, 'block': block, 'scenario': scenario,
                                'rejections': count, 'M': m, 'rejection_rate': count/m,
                                'wilson95': wilson(count, m),
                                'mean_mc_se': float(np.mean([v['mc_se_plugin'] for v in values])),
                                'runs': values})
    paired = []
    for rho in cfg['ar_rhos']:
        indicators = {block: np.array([v['p'] <= cfg['alpha'] for v in next(
            r['runs'] for r in records if r['rho'] == rho and r['block'] == block and r['scenario'] == 'null')])
                      for block in cfg['blocks']}
        for i, first in enumerate(cfg['blocks']):
            for second in cfg['blocks'][i+1:]:
                a, b_reject = indicators[first], indicators[second]
                differences = a.astype(float)-b_reject.astype(float)
                paired.append({'rho': rho, 'first_block': first, 'second_block': second,
                               'difference_first_minus_second': float(differences.mean()),
                               'paired_standard_error': float(differences.std(ddof=1)/np.sqrt(m)),
                               'first_only_rejects': int(np.sum(a & ~b_reject)),
                               'second_only_rejects': int(np.sum(~a & b_reject)),
                               'M': m})
    weights = prediction_sum_weights(xt, xf)
    analytic = [{'rho': rho, 'future_noise_sum_variance': ar1_sum_variance(h, rho),
                 'prediction_error_sum_variance': float(weights @ ar1_covariance(n+h, rho) @ weights),
                 'iid_reference_h_plus_c_squared': float(h+weights[:n] @ weights[:n])}
                for rho in cfg['ar_rhos']]
    return {'fixed_design': x.tolist(), 'records': records, 'illustration': illustration,
            'paired_null_rate_differences': paired, 'analytic_variance_reference': analytic,
            'power_size_adjusted': False,
            'mc_se_at_alpha': float(np.sqrt(b*cfg['alpha']*(1-cfg['alpha']))/(b+1)),
            'resolution': 1/(b+1), 'conditional_on_design': True}


def entropy_experiment(cfg, rng):
    edges = [-11, -5, 0, 5, 11]
    exact = []
    for label, support, probabilities in [('A', [-1, 1], [.5, .5]), ('B', [-10, 10], [.9, .1])]:
        values, p = np.array(support), np.array(probabilities)
        mean = float(np.sum(values*p))
        sample = rng.choice(values, size=10000, p=p)
        exact.append({'label': label, 'support': support, 'probabilities': probabilities,
                      'entropy_bits': shannon(p), 'variance': float(np.sum(p*(values-mean)**2)),
                      'sample_size': len(sample), 'sample_entropy_bits': binned_entropy(sample, edges),
                      'sample_std_ddof1': float(sample.std(ddof=1))})
    # 同一条独立合成路径，窗口只用尾部；每个bin数两段共用固定边界。
    baseline = rng.normal(size=2000)
    changed = rng.normal(loc=0.8, scale=2, size=2000)
    sensitivity = []
    for window in cfg['entropy_windows']:
        for bins in cfg['entropy_bins']:
            common = np.linspace(-12, 12, bins+1)
            a, b = baseline[-window:], changed[-window:]
            ha, hb = binned_entropy(a, common), binned_entropy(b, common)
            sensitivity.append({'window': window, 'bins': bins, 'edges': common.tolist(),
                                'baseline_entropy': ha, 'changed_entropy': hb, 'delta_entropy': hb-ha,
                                'baseline_std': float(a.std(ddof=1)), 'changed_std': float(b.std(ddof=1))})
    return {'counterexamples': exact, 'counterexample_common_edges': edges, 'sensitivity': sensitivity,
            'sensitivity_synthetic_baseline': baseline.tolist(),
            'sensitivity_synthetic_changed': changed.tolist(),
            'interpretation': '离散分箱Shannon熵依赖概率、箱边界和窗口；与标准差无普遍单调关系。'}


def collinearity(cfg, rng):
    results = []
    truth = np.array(cfg['beta'])
    for rho in cfg['collinearity_rhos']:
        errors, mse, cond, orth, decomposition, estimates = [], [], [], [], [], []
        for _ in range(cfg['collinearity_repeats']):
            x = design(rng, cfg['ntrain'], rho)
            y = x @ truth + rng.normal(size=len(x))
            bhat, resid, condition = fit_ols(x, y)
            test = design(rng, 1000, rho)
            ytest = test @ truth + rng.normal(size=len(test))
            errors.append((bhat-truth)**2)
            estimates.append(bhat.tolist())
            mse.append(float(np.mean((ytest-test @ bhat)**2)))
            cond.append(condition)
            orth.append(float(np.max(np.abs(x.T @ resid))))
            fitted = x @ bhat
            decomposition.append(float(abs(np.sum((y-y.mean())**2) - np.sum((fitted-y.mean())**2) - resid@resid)))
        results.append({'rho': rho, 'repeats': cfg['collinearity_repeats'],
                        'coefficient_rmse': np.sqrt(np.mean(errors, axis=0)).tolist(),
                        'coefficient_rmse_joint': float(np.sqrt(np.mean(errors))),
                        'condition_number_median': float(np.median(cond)),
                        'condition_number_max': float(np.max(cond)),
                        'out_of_sample_mse_mean': float(np.mean(mse)),
                        'out_of_sample_mse_se': float(np.std(mse, ddof=1)/np.sqrt(len(mse))),
                        'max_abs_Xt_residual': float(max(orth)),
                        'max_sse_decomposition_error': float(max(decomposition)),
                        'coefficient_estimates': estimates, 'sample_mse': mse, 'condition_numbers': cond})
    return results
