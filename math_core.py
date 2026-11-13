"""三个数学主题的数值核心；随机数生成器均由调用者传入。"""
from datetime import date
import numpy as np
from scipy.stats import norm


class DegenerateDataError(ValueError):
    """秩不足、常量残差池或点质量零分布，不提供常规检验p值。"""


class NonFiniteStatisticError(FloatingPointError):
    """输入或计算统计量包含非有限值。"""


def effectively_constant(values, reference_scale=1.0):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise NonFiniteStatisticError('常量检查要求非空有限数组')
    tolerance = 64*np.finfo(float).eps*max(1.0, float(reference_scale), float(np.max(np.abs(values))))
    return bool(np.ptp(values) <= tolerance)


def fit_ols(x, y):
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise NonFiniteStatisticError('OLS输入包含非有限值')
    beta, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise DegenerateDataError('设计矩阵秩不足，无法识别全部系数')
    residual = y - x @ beta
    if not np.isfinite(beta).all() or not np.isfinite(residual).all():
        raise NonFiniteStatisticError('OLS计算产生非有限值')
    return beta, residual, float(singular[0] / singular[-1])


def ar1_covariance(n, rho):
    if n < 1 or not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError('AR(1)要求n为正，且|rho|<1')
    indices = np.arange(n)
    return rho**np.abs(indices[:, None]-indices[None, :])


def ar1_sum_variance(horizon, rho):
    if horizon < 1 or not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError('AR(1)要求正窗口与|rho|<1')
    lags = np.arange(1, horizon)
    return float(horizon+2*np.sum((horizon-lags)*rho**lags))


def prediction_sum_weights(xtrain, xtest):
    # Xtrain.T c = Xtest.T 1 的最小范数解，等价于Xtrain(Xtrain'Xtrain)^-1Xtest'1。
    # 直接对转置设计做lstsq，避免显式逆与正规方程的条件数平方。
    c, _, rank, _ = np.linalg.lstsq(xtrain.T, xtest.T @ np.ones(len(xtest)), rcond=None)
    if rank != xtrain.shape[1]:
        raise DegenerateDataError('预测误差权重要求满列秩设计')
    return np.r_[-c, np.ones(len(xtest))]


def design(rng, n, rho):
    z = rng.normal(size=(n, 2))
    return np.column_stack((np.ones(n), z[:, 0], rho*z[:, 0] + np.sqrt(1-rho*rho)*z[:, 1]))


def ar1(rng, n, rho):
    # 平稳初始化，边际方差为1；创新方差为1-rho²。
    e = np.empty(n)
    e[0] = rng.normal()
    for t in range(1, n):
        e[t] = rho*e[t-1] + np.sqrt(1-rho*rho)*rng.normal()
    return e


def moving_indices(rng, n, length, block, draws):
    if not 1 <= block <= n or length < 1 or draws < 1:
        raise ValueError('无效分块参数')
    starts = rng.integers(0, n-block+1, size=(draws, (length+block-1)//block))
    return (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :length]


def bootstrap_null(rng, xtrain, xtest, ytrain, block, draws):
    beta, resid, _ = fit_ols(xtrain, ytrain)
    centered = resid - resid.mean()
    if effectively_constant(centered, np.max(np.abs(ytrain))):
        raise DegenerateDataError('训练残差为常量或数值上全零，bootstrap残差池退化')
    if not np.isfinite(xtest).all():
        raise NonFiniteStatisticError('bootstrap测试设计包含非有限值')
    n, h = len(xtrain), len(xtest)
    indices = moving_indices(rng, n, n+h, block, draws)
    errors = centered[indices]
    # 每列是一条训练伪样本；一次多右端项lstsq实际重新拟合B个beta。
    ystar = xtrain @ beta[:, None] + errors[:, :n].T
    betastar = np.linalg.lstsq(xtrain, ystar, rcond=None)[0]
    future = xtest @ beta[:, None] + errors[:, n:].T
    statistic = np.abs((future - xtest @ betastar).sum(axis=0))
    if not np.isfinite(statistic).all():
        raise NonFiniteStatisticError('bootstrap统计量包含非有限值')
    if effectively_constant(statistic):
        raise DegenerateDataError('bootstrap统计量零分布退化为单点，不提供常规p值')
    return statistic


def mc_pvalue(observed, null):
    null = np.asarray(null)
    if null.ndim != 1 or len(null) == 0:
        raise ValueError('模拟数组必须为非空一维数组')
    if not np.isfinite(null).all() or not np.isfinite(observed):
        raise NonFiniteStatisticError('观测或bootstrap统计量包含非有限值')
    if effectively_constant(null):
        raise DegenerateDataError('bootstrap统计量零分布退化为单点，不提供常规p值')
    exceed = int(np.count_nonzero(null >= observed))
    b = len(null)
    p = (1+exceed)/(b+1)
    # 带+1平滑后的插值MC标准误；不是有限样本有效性的保证。
    se = np.sqrt(b*p*(1-p))/(b+1)
    return {'p': float(p), 'exceed': exceed, 'B': b, 'mc_se_plugin': float(se)}


def wilson(k, n, level=0.95):
    if n < 1 or not 0 <= k <= n:
        raise ValueError('无效二项计数')
    z = norm.ppf(0.5+level/2)
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(max(0, center-half)), float(min(1, center+half))]


def shannon(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError('概率必须非负、有限且和为1')
    q = p[p > 0]
    return float(-np.sum(q*np.log2(q)))


def binned_entropy(values, edges):
    values, edges = np.asarray(values), np.asarray(edges)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('样本必须非空且有限')
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError('箱边界必须严格递增')
    if values.min() < edges[0] or values.max() > edges[-1]:
        raise ValueError('公共箱必须覆盖全部样本，不能静默丢弃尾部')
    counts = np.histogram(values, bins=edges)[0]
    return shannon(counts/counts.sum())


def chronological_split(dates, ntrain, horizon):
    parsed = [date.fromisoformat(d) for d in dates]
    if any(b <= a for a, b in zip(parsed, parsed[1:])):
        raise ValueError('日期重复或顺序不严格递增')
    if any((b-a).days > 7 for a, b in zip(parsed, parsed[1:])):
        raise ValueError('存在超过7自然日的日期缺口，拒绝跨缺口计算')
    if len(parsed) != ntrain+horizon or ntrain < 1 or horizon < 1:
        raise ValueError('训练与测试窗口长度不匹配')
    return slice(0, ntrain), slice(ntrain, ntrain+horizon)
