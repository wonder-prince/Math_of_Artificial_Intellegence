"""可选归档价格接口：下载证据门控与严格时间切分。"""
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
import numpy as np
from math_core import (fit_ols, bootstrap_null, mc_pvalue, chronological_split, binned_entropy,
                       effectively_constant, DegenerateDataError, NonFiniteStatisticError)


def load_verified(path):
    path = Path(path)
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('status') != 'verified_download':
        raise ValueError('真实数据未标识为verified_download')
    if not doc.get('source'):
        raise ValueError('缺少数据来源')
    evidence = doc.get('download_evidence', {})
    for field in ('url', 'retrieved_at', 'artifact', 'sha256'):
        if not evidence.get(field):
            raise ValueError('缺少下载证据字段: '+field)
    if not evidence['url'].startswith('https://'):
        raise ValueError('下载来源必须是HTTPS')
    retrieved = datetime.fromisoformat(evidence['retrieved_at'].replace('Z', '+00:00'))
    if retrieved.utcoffset() is None:
        raise ValueError('下载时间戳必须带时区')
    artifact = (path.parent/evidence['artifact']).resolve()
    if not artifact.is_relative_to(path.parent.resolve()):
        raise ValueError('证据原件必须归档在data目录内')
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != evidence['sha256']:
        raise ValueError('下载证据SHA256不匹配')
    rows = doc.get('rows', [])
    dates = [r['date'] for r in rows]
    chronological_split(dates, len(rows)-1, 1)
    prices = np.array([[r['NVDA'], r['SPY']] for r in rows], dtype=float)
    if prices.shape != (len(rows), 2) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('价格必须完整、正数且有限')
    return doc, dates, prices


def data_availability(root):
    path = root/'data'/'download_validation.json'
    if not path.exists():
        return {'status': 'no_download_record', 'download_validation': None}
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return {'status': 'invalid_download_record', 'reason': str(exc)}
    if not isinstance(record, dict):
        return {'status': 'invalid_download_record', 'reason': '取数记录必须为JSON对象'}
    return {'status': record.get('status', 'unknown'), 'reason': record.get('error'),
            'record_path': 'data/download_validation.json', 'download_validation': record,
            'network_attempted_by_experiments': False}


def optional_real(cfg, root, rng):
    availability = data_availability(root)
    path = root/'data'/'verified_prices.json'
    if not path.exists():
        return {'status': 'not_provided', 'reason': '未提供有归档下载证据的真实价格；只运行合成实验。',
                'data_availability': availability}
    try:
        doc, dates, prices = load_verified(path)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return {'status': 'rejected', 'reason': str(exc), 'data_availability': availability}
    returns = np.diff(np.log(prices), axis=0)
    return_dates = dates[1:]
    n, h = cfg['ntrain'], cfg['horizon']
    windows = []
    for split in range(n, len(returns)-h+1, h):
        start = split-n
        tr, te = chronological_split(return_dates[start:split+h], n, h)
        item = {'train_start': return_dates[start], 'train_end': return_dates[split-1],
                'test_start': return_dates[split], 'test_end': return_dates[split+h-1],
                'entropy_comparable_for_significance': False,
                'entropy_interpretation': '仅描述；训练与测试样本量不同，插件熵有样本量偏差；未校正也未进行熵显著性检验。'}
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                local = returns[start:split+h]
                x = np.column_stack((np.ones(len(local)), local[:, 1]))
                beta, residual, _ = fit_ols(x[tr], local[tr, 0])
                future_residual = local[te, 0]-x[te] @ beta
                statistic = float(abs(future_residual.sum()))
                if not np.isfinite(statistic):
                    raise NonFiniteStatisticError('真实窗口累计统计量非有限')
                item.update({'beta': beta.tolist(), 'test_residuals': future_residual.tolist(),
                             'T': statistic, 'train_std': float(residual.std(ddof=1)),
                             'test_std': float(future_residual.std(ddof=1))})
                if effectively_constant(residual, np.max(np.abs(local[tr, 0]))):
                    # 点质量离散熵为0；没有可识别的训练分箱尺度，不比较测试熵。
                    item.update({'train_entropy': 0.0, 'test_entropy': None,
                                 'entropy_status': 'constant_training_residual_point_mass',
                                 'bootstrap': None})
                    raise DegenerateDataError('训练残差为常量或数值上全零：H定义为0，测试熵不可比较，bootstrap不可用')
                internal = np.unique(np.linspace(residual.min(), residual.max(), 9)[1:-1])
                edges = np.r_[-np.inf, internal, np.inf]
                tests = {str(block): mc_pvalue(statistic, bootstrap_null(rng, x[tr], x[te], local[tr, 0], block, cfg['B']))
                         for block in cfg['blocks']}
                item.update({'status': 'computed', 'bootstrap': tests,
                             'train_entropy': binned_entropy(residual, edges),
                             'test_entropy': binned_entropy(future_residual, edges),
                             'entropy_status': 'descriptive_unequal_sample_sizes'})
        except DegenerateDataError as exc:
            item.update({'status': 'skipped_degenerate', 'reason': str(exc)})
        except (NonFiniteStatisticError, np.linalg.LinAlgError, FloatingPointError) as exc:
            item.update({'status': 'skipped_numerical', 'reason': str(exc)})
        # 不捕获普通ValueError、KeyError或程序错误，防止隐藏配置/逻辑缺陷。
        windows.append(item)
    weekdays_missing = []
    observed = set(dates)
    for d in np.arange(np.datetime64(dates[0]), np.datetime64(dates[-1])+1):
        text = str(d)
        if date.fromisoformat(text).weekday() < 5 and text not in observed:
            weekdays_missing.append(text)
    counts = {status: sum(w['status'] == status for w in windows)
              for status in ('computed', 'skipped_degenerate', 'skipped_numerical')}
    status = 'computed' if counts['computed'] else ('no_valid_windows' if windows else 'insufficient_rows')
    return {'status': status, 'source': doc['source'], 'data_availability': availability,
            'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'download_evidence': doc['download_evidence'], 'rows': len(prices), 'windows': windows,
            'window_status_counts': counts, 'missing_weekdays_review': weekdays_missing,
            'limitations': ['缺失工作日可能是交易所假日；未内置交易日历，不把工作日缺失自动认定为缺数。',
                            '仅校验归档原件哈希与声明来源；来源可信性及原件到rows的转换须由提供者独立验证。',
                            '实际测试期SPY回报作为固定设计条件，属于事后条件推断，不是事前预测。',
                            '训练200与测试20的经验熵受不同样本量偏差影响，仅作描述，不能视为显著性证据。',
                            '默认使用提供的价格口径；提供者应说明复权/拆股/股息处理。']}
