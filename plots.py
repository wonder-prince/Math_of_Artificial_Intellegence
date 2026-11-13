"""浅色背景、独立可导出的实验图。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_plots(result, root):
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'figure.facecolor': 'white',
                         'axes.facecolor': '#fafbfd', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    outputs = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'svg'):
            target = root/'figures'/f'{name}.{extension}'
            fig.savefig(target, bbox_inches='tight')
            outputs.append(str(target.relative_to(root)))
        plt.close(fig)

    records = result['bootstrap']['records']
    scenarios = ['null', 'mean_shift', 'variance_increase', 'outlier']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, rho in zip(axes, result['config']['ar_rhos']):
        for j, block in enumerate(result['config']['blocks']):
            selected = [next(r for r in records if r['rho'] == rho and r['block'] == block and r['scenario'] == s)
                        for s in scenarios]
            y = np.array([r['rejection_rate'] for r in selected])
            lo = np.array([r['wilson95'][0] for r in selected])
            hi = np.array([r['wilson95'][1] for r in selected])
            ax.errorbar(np.arange(4)+(j-1)*.17, y, yerr=[y-lo, hi-y], fmt='o', capsize=3,
                        label=f'block={block}')
        ax.axhline(.05, ls='--', color='gray', lw=1, label='nominal 0.05')
        ax.set_xticks(range(4), ['Null', 'Mean shift', 'Variance x4', 'One outlier'], rotation=18)
        ax.set_title(f"AR(1) rho={rho}: M={result['config']['M']}, B={result['config']['B']}")
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Rejection rate with 95% Wilson interval')
    axes[1].legend(fontsize=9)
    save(fig, '01_bootstrap_calibration')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rho in zip(axes, result['config']['ar_rhos']):
        for block in result['config']['blocks']:
            r = next(r for r in records if r['rho'] == rho and r['block'] == block and r['scenario'] == 'null')
            p = np.sort([v['p'] for v in r['runs']])
            ax.step(p, np.arange(1, len(p)+1)/len(p), where='post', label=f'block={block}')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(xlabel='Null p-value', ylabel='Empirical CDF', title=f'Null p-value calibration: rho={rho}')
        ax.legend()
    save(fig, '02_null_pvalues')

    c = result['entropy']['counterexamples']
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ['A: +/-1, 0.5/0.5', 'B: +/-10, 0.9/0.1']
    axes[0].bar(labels, [r['entropy_bits'] for r in c], color=['#4285b4', '#da8c52'])
    axes[1].bar(labels, [r['variance'] for r in c], color=['#4285b4', '#da8c52'])
    axes[0].set(ylabel='Shannon entropy (bits)', title='Exact probability counterexample')
    axes[1].set(ylabel='Variance', title='Higher variance, lower discrete entropy')
    for ax in axes:
        ax.tick_params(axis='x', labelsize=9)
    save(fig, '03_entropy_counterexample')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for window in result['config']['entropy_windows']:
        rows = [r for r in result['entropy']['sensitivity'] if r['window'] == window]
        ax.plot([r['bins'] for r in rows], [r['delta_entropy'] for r in rows], 'o-', label=f'window={window}')
    ax.axhline(0, ls='--', color='gray')
    ax.set(xlabel='Number of common bins over [-12, 12]', ylabel='Changed minus baseline entropy (bits)',
           title='Synthetic entropy: bin and window sensitivity')
    ax.legend()
    save(fig, '04_entropy_sensitivity')

    col = result['collinearity']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ['coefficient_rmse_joint', 'condition_number_median', 'out_of_sample_mse_mean'],
                             ['Coefficient RMSE', 'Median condition number', 'Out-of-sample MSE']):
        ax.plot(range(len(col)), [r[key] for r in col], 'o-', color='#4285b4')
        ax.set_xticks(range(len(col)), [str(r['rho']) for r in col])
        ax.set(xlabel='Predictor correlation rho', title=title)
        ax.grid(alpha=.2)
        if key == 'out_of_sample_mse_mean':
            ax.set_ylim(0, max(1.15, max(r[key] for r in col)*1.1))
            ax.axhline(1, ls='--', color='gray', lw=1, label='Noise variance = 1')
            ax.errorbar(range(len(col)), [r[key] for r in col],
                        yerr=[1.96*r['out_of_sample_mse_se'] for r in col],
                        fmt='none', capsize=4, color='#4285b4')
            ax.legend(fontsize=8, loc='lower right')
    fig.suptitle(f"{result['config']['collinearity_repeats']} repetitions; test design matches training population", y=1.04)
    save(fig, '05_collinearity')

    demo = result['bootstrap']['illustration']
    if demo:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        axes[0].hist(demo['null_statistics'], bins=35, color='#75a5bd', alpha=.85)
        for key, color in [('null', '#42679a'), ('mean_shift', '#d17339'), ('variance_increase', '#549871')]:
            axes[0].axvline(demo['observed'][key], label=key, color=color)
        axes[0].set(xlabel='T = abs(sum of test residuals)', ylabel='Bootstrap count', title='One illustrative draw, rho=0.6')
        axes[0].legend(fontsize=9)
        axes[1].plot(demo['training_residuals'], lw=1, color='#42679a')
        axes[1].set(xlabel='Training index', ylabel='OLS residual', title='Centered AR-dependent residual pool')
        save(fig, '06_bootstrap_distribution')
    return outputs
