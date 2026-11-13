"""下载、原件归档与基本门控；不运行任何实验。"""
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
START = date(2023, 1, 1)
END = date(2025, 12, 31)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36', 'Accept': 'application/json'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, doc):
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fetch(url, name, records):
    # 原件逐字节归档，解析前先保存；拒绝访问立即终止，不切换来源。
    try:
        with urlopen(Request(url, headers=HEADERS), timeout=45) as response:
            raw = response.read()
            final_url = response.geturl()
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(f'端点拒绝访问，停止下载：HTTP {exc.code} {url}') from exc
        raise
    stamp = datetime.now(timezone.utc).isoformat()
    artifact = DATA / name
    if artifact.exists():
        raise FileExistsError(f'不覆盖已有原件：{artifact}')
    artifact.write_bytes(raw)
    records.append({'url': url, 'final_url': final_url, 'retrieved_at': stamp,
                    'artifact': name, 'sha256': sha(raw), 'bytes': len(raw)})
    return json.loads(raw)


def valid_price(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def yahoo(symbol, records):
    start = int(datetime.combine(START, datetime.min.time(), timezone.utc).timestamp())
    stop = int(datetime.combine(END + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp())
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + symbol + '?' + urlencode({
        'period1': start, 'period2': stop, 'interval': '1d', 'events': 'div,splits', 'includeAdjustedClose': 'true'})
    doc = fetch(url, f'yahoo_{symbol}_2023_2025.json', records)
    chart = doc['chart']
    if chart.get('error') or not chart.get('result'):
        raise ValueError(f'Yahoo返回错误：{chart.get("error")}')
    result = chart['result'][0]
    if result['meta']['symbol'] != symbol or result['meta'].get('currency') != 'USD':
        raise ValueError('标的或币种不匹配')
    timestamps = result['timestamp']
    prices = result['indicators']['adjclose'][0]['adjclose']
    if len(timestamps) != len(prices):
        raise ValueError('时间戳与复权价长度不一致')
    tz = ZoneInfo(result['meta']['exchangeTimezoneName'])
    rows = {}
    nulls = []
    for ts, price in zip(timestamps, prices):
        day = datetime.fromtimestamp(ts, tz).date()
        if not START <= day <= END:
            continue
        text = day.isoformat()
        if text in rows or text in nulls:
            raise ValueError('原件存在重复日期')
        if price is None:
            nulls.append(text)
        elif valid_price(price):
            rows[text] = float(price)
        else:
            raise ValueError('原件价格非法')
    return rows, {'field': 'adjclose', 'null_dates': nulls, 'events': result.get('events', {}),
                  'price_basis': 'Yahoo adjclose，供应商拆股和现金分红复权；不再进行拆股或股息调整'}


def nasdaq(symbol, records):
    url = 'https://api.nasdaq.com/api/quote/' + symbol + '/historical?' + urlencode({
        'assetclass': 'etf' if symbol == 'SPY' else 'stocks', 'fromdate': START.isoformat(),
        'todate': END.isoformat(), 'limit': 9999})
    doc = fetch(url, f'nasdaq_{symbol}_2023_2025.json', records)
    data = doc.get('data') or {}
    raw_rows = (data.get('tradesTable') or {}).get('rows') or []
    if not raw_rows:
        raise ValueError('Nasdaq历史记录为空')
    rows = {}
    for item in raw_rows:
        day = datetime.strptime(item['date'], '%m/%d/%Y').date()
        if not START <= day <= END:
            continue
        text = day.isoformat()
        price = float(item['close'].replace('$', '').replace(',', '').strip())
        if text in rows or not valid_price(price):
            raise ValueError('Nasdaq日期重复或价格非法')
        rows[text] = price
    split_check = None
    if symbol == 'NVDA':
        before, after = rows.get('2024-06-07'), rows.get('2024-06-10')
        if before is None or after is None:
            raise ValueError('缺少NVDA拆股前后核验日期')
        ratio = before / after
        split_check = {'before': before, 'after': after, 'ratio': ratio}
        if not 0.5 < ratio < 2:
            raise ValueError('NVDA跨拆股日不连续；无法确认统一拆股口径，不输出已验证价格')
    return rows, {'field': 'close', 'split_check': split_check,
                  'price_basis': 'Nasdaq close；跨2024-06-10检查拆股连续性；现金分红未复权'}


def main():
    if not DATA.is_dir():
        raise FileNotFoundError('项目data目录不存在')
    for name in ('verified_prices.json', 'download_bundle.json', 'download_validation.json'):
        if (DATA / name).exists():
            raise FileExistsError(f'不覆盖已有归档：{name}')
    records, failures = [], []
    series = details = None
    provider = None
    for provider, downloader in (('Yahoo Finance', yahoo), ('Nasdaq', nasdaq)):
        try:
            fetched = {symbol: downloader(symbol, records) for symbol in ('NVDA', 'SPY')}
            series = {symbol: value[0] for symbol, value in fetched.items()}
            details = {symbol: value[1] for symbol, value in fetched.items()}
            if len(set(series['NVDA']) & set(series['SPY'])) < 221:
                raise ValueError('共同交易日不足221')
            break
        except PermissionError as exc:
            save_json(DATA / 'download_validation.json', {
                'status': 'blocked_permission', 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'provider': provider, 'error': str(exc), 'attempts_failed': failures,
                'original_artifacts': records, 'rows': 0,
                'limitations': ['权限拒绝后停止；未重试或切换来源；未生成verified_prices.json']})
            raise
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as exc:
            failures.append({'provider': provider, 'error': str(exc)})
            series = None
    if series is None:
        save_json(DATA / 'download_validation.json', {'status': 'download_failed', 'attempts': failures, 'artifacts': records})
        raise RuntimeError('所有可用来源下载失败，详见download_validation.json')
    dates = sorted(set(series['NVDA']) & set(series['SPY']))
    if dates[0] != '2023-01-03' or dates[-1] < '2025-11-26':
        raise ValueError('日期覆盖范围不足')
    gaps = [(a, b) for a, b in zip(dates, dates[1:]) if (date.fromisoformat(b) - date.fromisoformat(a)).days > 7]
    if gaps:
        raise ValueError(f'出现超过7天缺口：{gaps}')
    rows = [{'date': day, 'NVDA': series['NVDA'][day], 'SPY': series['SPY'][day]} for day in dates]
    bundle = {'description': '原始响应索引；artifact文件为逐字节原件，payload为解析副本',
              'downloads': [{**r, 'payload': json.loads((DATA / r['artifact']).read_bytes())} for r in records]}
    save_json(DATA / 'download_bundle.json', bundle)
    raw_bundle = (DATA / 'download_bundle.json').read_bytes()
    doc = {'status': 'verified_download', 'source': provider + '；' + details['NVDA']['price_basis'],
           'download_evidence': {'url': records[-1]['url'], 'retrieved_at': max(r['retrieved_at'] for r in records),
                                 'artifact': 'download_bundle.json', 'sha256': sha(raw_bundle), 'original_downloads': records},
           'price_details': details, 'rows': rows}
    # 先验证全部输入，再写入契约文件。
    for r in records:
        if sha((DATA / r['artifact']).read_bytes()) != r['sha256']:
            raise ValueError('原件哈希不一致')
    save_json(DATA / 'verified_prices.json', doc)
    from real_data import load_verified
    loaded, loaded_dates, prices = load_verified(DATA / 'verified_prices.json')
    if loaded['rows'] != rows or loaded_dates != dates or prices.shape != (len(rows), 2):
        raise ValueError('现有接口读取结果不一致')
    missing = {symbol: sorted(set(series[symbol]) - set(dates)) for symbol in series}
    validation = {'status': 'passed', 'provider': provider, 'rows': len(rows), 'first_date': dates[0], 'last_date': dates[-1],
                  'price_details': details, 'unmatched_dates': missing, 'gaps_over_7_days': gaps,
                  'attempts_failed': failures, 'original_artifacts': records,
                  'verified_prices_sha256': sha((DATA / 'verified_prices.json').read_bytes()),
                  'checks': ['HTTPS原始响应逐字节归档及SHA256', '共同交易日不少于221', '日期唯一递增且间隔不超过7天',
                             '正数有限价格', '现有load_verified门控通过', '原始数据转换逐行复核'],
                  'limitations': ['未独立对照交易所交易日历，不将节假日当作缺数', '未进行第二供应商价格交叉验证',
                                  '复权价格按下载时供应商历史调整口径，未构建当时可得价格数据库']}
    save_json(DATA / 'download_validation.json', validation)
    print(json.dumps({k: validation[k] for k in ('status', 'provider', 'rows', 'first_date', 'last_date', 'unmatched_dates')}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
