import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from math_core import (fit_ols, design, moving_indices, bootstrap_null, mc_pvalue,
                       shannon, binned_entropy, chronological_split, wilson,
                       DegenerateDataError, NonFiniteStatisticError,
                       ar1_covariance, ar1_sum_variance, prediction_sum_weights)
from real_data import load_verified, optional_real, data_availability


class MathTests(unittest.TestCase):
    def test_known_truth_ols(self):
        rng = np.random.default_rng(10)
        x = design(rng, 200, .999)
        truth = np.array([.3, 1.2, -.7])
        beta, resid, _ = fit_ols(x, x @ truth)
        np.testing.assert_allclose(beta, truth, atol=1e-11)
        np.testing.assert_allclose(resid, 0, atol=1e-11)

    def test_orthogonality_and_sse(self):
        rng = np.random.default_rng(11)
        x = design(rng, 200, .9)
        y = rng.normal(size=200)
        beta, residual, _ = fit_ols(x, y)
        np.testing.assert_allclose(x.T @ residual, 0, atol=1e-11)
        self.assertAlmostEqual(float(np.sum((y-y.mean())**2)),
                               float(np.sum((x @ beta-y.mean())**2)+residual @ residual), places=10)

    def test_entropy_boundaries(self):
        self.assertEqual(shannon([1, 0]), 0)
        self.assertEqual(shannon([.25]*4), 2)
        self.assertEqual(binned_entropy([-1, 1], [-1, 0, 1]), 1)
        with self.assertRaises(ValueError):
            binned_entropy([-2, 1], [-1, 0, 1])
        with self.assertRaises(ValueError):
            shannon([.3, .4])

    def test_entropy_counterexample(self):
        self.assertEqual(shannon([.5, .5]), 1)
        self.assertAlmostEqual(shannon([.9, .1]), .4689955935892812)
        p, v = np.array([.9, .1]), np.array([-10, 10])
        self.assertAlmostEqual(float(np.sum(p*(v-np.sum(p*v))**2)), 36)
        self.assertLess(binned_entropy([-10]*9+[10], [-11, -5, 0, 5, 11]),
                        binned_entropy([-1, 1]*5, [-11, -5, 0, 5, 11]))

    def test_pvalue_boundaries_and_ties(self):
        self.assertEqual(mc_pvalue(20, np.arange(9))['p'], .1)
        self.assertEqual(mc_pvalue(0, np.arange(9))['p'], 1)
        self.assertEqual(mc_pvalue(1, np.array([1, 1, 2]))['p'], 1)
        with self.assertRaises(ValueError):
            mc_pvalue(1, [])

    def test_block_continuity_without_wrap(self):
        for block in (1, 5, 10):
            idx = moving_indices(np.random.default_rng(12), 23, 40, block, 50)
            self.assertTrue(np.all((idx >= 0) & (idx < 23)))
            if block > 1:
                np.testing.assert_array_equal(np.diff(idx.reshape(50, -1, block), axis=-1), 1)
        with self.assertRaises(ValueError):
            moving_indices(np.random.default_rng(12), 5, 20, 10, 3)

    def test_refit_bootstrap_matches_manual(self):
        rng = np.random.default_rng(13)
        x = design(rng, 30, .5)
        xfuture = design(rng, 10, .5)
        y = x @ np.array([1., 2., 3.]) + rng.normal(size=30)
        actual = bootstrap_null(np.random.default_rng(14), x, xfuture, y, 5, 12)
        beta, resid, _ = fit_ols(x, y)
        idx = moving_indices(np.random.default_rng(14), 30, 40, 5, 12)
        errors = (resid-resid.mean())[idx]
        expected = []
        for draw in errors:
            b, _, _ = fit_ols(x, x @ beta + draw[:30])
            expected.append(abs(np.sum(xfuture @ beta + draw[30:] - xfuture @ b)))
        np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_no_future_y_in_bootstrap(self):
        rng = np.random.default_rng(44)
        x = design(rng, 220, .5)
        y = x @ np.array([1., 2., 3.]) + rng.normal(size=220)
        altered = y.copy()
        altered[200:] += 1000
        beta1, _, _ = fit_ols(x[:200], y[:200])
        beta2, _, _ = fit_ols(x[:200], altered[:200])
        np.testing.assert_array_equal(beta1, beta2)
        null1 = bootstrap_null(np.random.default_rng(45), x[:200], x[200:], y[:200], 10, 99)
        null2 = bootstrap_null(np.random.default_rng(45), x[:200], x[200:], altered[:200], 10, 99)
        np.testing.assert_array_equal(null1, null2)
        t1 = abs((y[200:]-x[200:] @ beta1).sum())
        t2 = abs((altered[200:]-x[200:] @ beta2).sum())
        self.assertGreater(t2, t1+10000)

    def test_chronological_no_leakage(self):
        dates = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        tr, te = chronological_split(dates, 2, 2)
        self.assertLess(max(dates[tr]), min(dates[te]))
        for bad in (dates[::-1], dates[:2]+dates[:2], ['2024-01-02', '2024-01-12']):
            with self.assertRaises(ValueError):
                chronological_split(bad, len(bad)-1, 1)

    def test_wilson_boundaries(self):
        self.assertGreater(wilson(0, 300)[1], 0)
        self.assertLess(wilson(300, 300)[0], 1)
        self.assertTrue(wilson(15, 300)[0] < .05 < wilson(15, 300)[1])

    def test_real_data_evidence_gate(self):
        path = Path(__file__).resolve().parents[1]/'data'/'verified_prices.json'
        base = {'status': 'unverified', 'source': '单元测试虚构来源',
                'rows': [{'date': '2024-01-02', 'NVDA': 1, 'SPY': 1},
                         {'date': '2024-01-03', 'NVDA': 2, 'SPY': 2}]}
        # 仅在文件I/O边界模拟归档字节，状态/日期/哈希校验均执行真实实现。
        with patch.object(Path, 'read_text', side_effect=lambda **kw: json.dumps(base)):
            with self.assertRaises(ValueError):
                load_verified(path)
            base['status'] = 'verified_download'
            with self.assertRaises(ValueError):
                load_verified(path)
            raw = b'unit-test-artifact-only'
            base['download_evidence'] = {'url': 'https://example.invalid/unit-test',
                                         'retrieved_at': '2024-01-03T12:00:00Z',
                                         'artifact': 'raw.txt', 'sha256': hashlib.sha256(raw).hexdigest()}
            with patch.object(Path, 'read_bytes', return_value=raw):
                _, dates, _ = load_verified(path)
                self.assertEqual(len(dates), 2)
            with patch.object(Path, 'read_bytes', return_value=b'changed'):
                with self.assertRaises(ValueError):
                    load_verified(path)

    def test_real_rolling_interface_and_future_isolation(self):
        # 只模拟上游读取；下面执行真实滚动计算，不创建/清理临时文件。
        root = Path(__file__).resolve().parents[1]
        rng = np.random.default_rng(74)
        prices = 100*np.exp(np.cumsum(rng.normal(0, .01, size=(41, 2)), axis=0))
        dates = [str(d) for d in np.arange(np.datetime64('2024-01-01'), np.datetime64('2024-02-11'))]
        doc = {'source': '仅单元测试合成夹具', 'download_evidence': {}}
        cfg = {'ntrain': 20, 'horizon': 5, 'blocks': [1, 5, 10], 'B': 99}
        with patch.object(Path, 'exists', return_value=True), patch.object(Path, 'read_bytes', return_value=b'fixture'), \
                patch('real_data.data_availability', return_value={'status': 'unit_test_only'}):
            with patch('real_data.load_verified', return_value=(doc, dates, prices)):
                first = optional_real(cfg, root, np.random.default_rng(75))
            self.assertEqual(first['status'], 'computed')
            self.assertEqual(len(first['windows']), 4)
            for w in first['windows']:
                self.assertLess(w['train_end'], w['test_start'])
                self.assertEqual(len(w['test_residuals']), 5)
            changed = prices.copy()
            changed[21:26, 0] *= 1.2
            with patch('real_data.load_verified', return_value=(doc, dates, changed)):
                second = optional_real(cfg, root, np.random.default_rng(75))
            self.assertEqual(first['windows'][0]['beta'], second['windows'][0]['beta'])
            self.assertEqual(first['windows'][0]['train_entropy'], second['windows'][0]['train_entropy'])
            self.assertNotEqual(first['windows'][0]['test_residuals'], second['windows'][0]['test_residuals'])

    def test_ar1_covariance_stationary_innovations(self):
        for rho in (0.0, 0.6, -0.4):
            n = 20
            loading = np.zeros((n, n))
            loading[:, 0] = rho**np.arange(n)
            for t in range(1, n):
                loading[t, 1:t+1] = np.sqrt(1-rho*rho)*rho**np.arange(t-1, -1, -1)
            cov = ar1_covariance(n, rho)
            np.testing.assert_allclose(cov, loading @ loading.T, atol=1e-14)
            np.testing.assert_array_equal(np.diag(cov), np.ones(n))
            self.assertAlmostEqual(cov[0, 3], rho**3)
        np.testing.assert_array_equal(ar1_covariance(20, 0), np.eye(20))

    def test_ar1_sum_variance_exact_formula(self):
        h, rho = 20, .6
        # 几何级数闭式独立核对滞后求和及协方差矩阵二次型。
        closed = h*(1+rho)/(1-rho)-2*rho*(1-rho**h)/(1-rho)**2
        actual = ar1_sum_variance(h, rho)
        self.assertAlmostEqual(actual, closed, places=12)
        self.assertAlmostEqual(actual, float(np.ones(h) @ ar1_covariance(h, rho) @ np.ones(h)), places=12)
        self.assertEqual(f'{actual:.6f}', '72.500274')
        self.assertEqual(ar1_sum_variance(h, 0), h)

    def test_fixed_design_prediction_sum_weights_and_covariance(self):
        rng = np.random.default_rng(83)
        n, h = 200, 20
        x = design(rng, n+h, .5)
        w = prediction_sum_weights(x[:n], x[n:])
        c = -w[:n]
        np.testing.assert_allclose(x[:n].T @ c, x[n:].T @ np.ones(h), atol=1e-12)
        np.testing.assert_array_equal(w[n:], np.ones(h))
        self.assertAlmostEqual(w[:n].sum(), -h, places=11)
        beta = np.array([.3, 1.2, -.7])
        errors = rng.normal(size=n+h)
        bhat, _, _ = fit_ols(x[:n], x[:n] @ beta+errors[:n])
        signed_sum = np.sum(x[n:] @ beta+errors[n:]-x[n:] @ bhat)
        self.assertAlmostEqual(signed_sum, float(w @ errors), places=11)
        self.assertAlmostEqual(float(w @ ar1_covariance(n+h, 0) @ w), h+float(c @ c), places=12)
        cov = ar1_covariance(n+h, .6)
        cross = float(c @ cov[:n, n:] @ np.ones(h))
        expanded = ar1_sum_variance(h, .6)+float(c @ cov[:n, :n] @ c)-2*cross
        self.assertAlmostEqual(float(w @ cov @ w), expanded, places=11)
        self.assertGreater(abs(cross), 1e-6)
        self.assertGreater(abs(float(w @ cov @ w)-(expanded+4*cross)), 1e-6)

    def test_bootstrap_nonfinite_statistics_are_rejected(self):
        for observed, null in ((np.inf, [0, 1]), (1, [np.nan, 1]), (1, [0, np.inf])):
            with self.assertRaises(NonFiniteStatisticError):
                mc_pvalue(observed, np.array(null))
        rng = np.random.default_rng(84)
        x = design(rng, 30, .5)
        y = rng.normal(size=30)
        xtest = x[:5].copy()
        xtest[0, 1] = np.inf
        with self.assertRaises(NonFiniteStatisticError):
            bootstrap_null(rng, x, xtest, y, 5, 99)

    def test_bootstrap_degenerate_pool_and_distribution(self):
        rng = np.random.default_rng(85)
        x = design(rng, 30, .5)
        with self.assertRaises(DegenerateDataError):
            bootstrap_null(rng, x, x[:5], x @ np.array([.3, 1.2, -.7]), 5, 99)
        with self.assertRaises(DegenerateDataError):
            mc_pvalue(1, np.ones(99))
        y = rng.normal(size=30)
        # 块长等于整个残差池时只有一个起点，真实抽样也退化为同一路径。
        with self.assertRaises(DegenerateDataError):
            bootstrap_null(rng, x, x[:5], y, 30, 99)
        # 非退化训练池，但所有伪路径完全相同，也必须拒绝点质量零分布。
        with patch('math_core.moving_indices', return_value=np.zeros((99, 35), dtype=int)):
            with self.assertRaises(DegenerateDataError):
                bootstrap_null(rng, x, x[:5], y, 5, 99)

    def rolling_memory_fixture(self, log_returns, bootstrap_error=None):
        root = Path(__file__).resolve().parents[1]
        prices = 100*np.exp(np.vstack((np.zeros(2), np.cumsum(log_returns, axis=0))))
        dates = [str(np.datetime64('2024-01-01')+i) for i in range(len(prices))]
        raw = b'synthetic-memory-fixture-only'
        doc = {'status': 'verified_download', 'source': '仅内存测试夹具',
               'download_evidence': {'url': 'https://example.invalid/unit-test',
                                     'retrieved_at': '2024-02-01T12:00:00Z', 'artifact': 'fixture.bin',
                                     'sha256': hashlib.sha256(raw).hexdigest()},
               'rows': [{'date': d, 'NVDA': float(p[0]), 'SPY': float(p[1])} for d, p in zip(dates, prices)]}
        cfg = {'ntrain': 20, 'horizon': 5, 'blocks': [1, 5, 10], 'B': 99}
        with patch.object(Path, 'exists', return_value=True), \
                patch.object(Path, 'read_text', return_value=json.dumps(doc)), \
                patch.object(Path, 'read_bytes', return_value=raw), \
                patch('real_data.data_availability', return_value={'status': 'unit_test_only'}):
            if bootstrap_error is not None:
                with patch('real_data.bootstrap_null', side_effect=bootstrap_error):
                    return optional_real(cfg, root, np.random.default_rng(86))
            return optional_real(cfg, root, np.random.default_rng(86))

    def test_real_rank_deficient_window_is_recorded_and_continues(self):
        returns = np.random.default_rng(87).normal(0, .01, size=(40, 2))
        returns[:20, 1] = 0
        result = self.rolling_memory_fixture(returns)
        first = result['windows'][0]
        self.assertEqual(first['status'], 'skipped_degenerate')
        self.assertIn('秩不足', first['reason'])
        self.assertEqual(result['windows'][1]['status'], 'computed')
        self.assertEqual(len(result['windows']), 4)
        self.assertEqual(result['window_status_counts']['skipped_degenerate'], 1)

    def test_real_zero_residual_window_entropy_zero_and_continues(self):
        returns = np.random.default_rng(88).normal(0, .01, size=(40, 2))
        returns[:20, 0] = 0
        result = self.rolling_memory_fixture(returns)
        first = result['windows'][0]
        self.assertEqual(first['status'], 'skipped_degenerate')
        self.assertIn('全零', first['reason'])
        self.assertEqual(first['train_entropy'], 0)
        self.assertIsNone(first['test_entropy'])
        self.assertIsNone(first['bootstrap'])
        self.assertFalse(first['entropy_comparable_for_significance'])
        self.assertEqual(result['windows'][1]['status'], 'computed')
        self.assertEqual(len(result['windows']), 4)

    def test_real_controlled_numerical_errors_and_unexpected_errors(self):
        returns = np.random.default_rng(89).normal(0, .01, size=(30, 2))
        result = self.rolling_memory_fixture(returns, NonFiniteStatisticError('测试非有限统计量'))
        self.assertEqual(result['status'], 'no_valid_windows')
        self.assertTrue(all(w['status'] == 'skipped_numerical' for w in result['windows']))
        result = self.rolling_memory_fixture(returns, np.linalg.LinAlgError('测试SVD失败'))
        self.assertTrue(all(w['status'] == 'skipped_numerical' for w in result['windows']))
        result = self.rolling_memory_fixture(returns, DegenerateDataError('测试退化零分布'))
        self.assertTrue(all(w['status'] == 'skipped_degenerate' for w in result['windows']))
        with self.assertRaisesRegex(RuntimeError, '程序缺陷'):
            self.rolling_memory_fixture(returns, RuntimeError('程序缺陷'))
        with self.assertRaisesRegex(ValueError, '配置错误'):
            self.rolling_memory_fixture(returns, ValueError('配置错误'))

    def test_missing_prices_retains_download_failure_without_network(self):
        root = Path(__file__).resolve().parents[1]
        record = {'status': 'blocked_permission', 'provider': 'Yahoo Finance',
                  'http_status': 403, 'error': 'HTTP 403 Forbidden；停止下载。', 'rows': 0}
        with patch.object(Path, 'exists', side_effect=lambda: True):
            with patch.object(Path, 'read_text', return_value=json.dumps(record)):
                availability = data_availability(root)
        self.assertEqual(availability['download_validation']['http_status'], 403)
        with patch('real_data.data_availability', return_value=availability), \
                patch.object(Path, 'exists', return_value=False), \
                patch('real_data.load_verified', side_effect=AssertionError('不应读取不存在的价格')):
            result = optional_real({}, root, np.random.default_rng(90))
        self.assertEqual(result['status'], 'not_provided')
        self.assertEqual(result['data_availability']['status'], 'blocked_permission')
        self.assertIn('403', result['data_availability']['reason'])
        self.assertFalse(result['data_availability']['network_attempted_by_experiments'])


if __name__ == '__main__':
    unittest.main()
