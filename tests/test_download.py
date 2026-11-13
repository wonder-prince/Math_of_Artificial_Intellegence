"""下载传输和重跑回归测试；HTTP请求全部模拟，文件仅写入临时目录。"""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import download_data as downloader


class DownloadTests(unittest.TestCase):
    def test_fetch_uses_system_proxy_and_archives_exact_bytes(self):
        raw = b'{"chart": {"result": []}}\n'
        response = Mock(status_code=200, content=raw, url='https://query1.finance.yahoo.com/result')
        with TemporaryDirectory() as directory, patch.object(downloader, 'DATA', Path(directory)), \
                patch.object(downloader, 'getproxies', return_value={'https': 'http://127.0.0.1:7897'}), \
                patch.object(downloader, 'proxy_bypass', return_value=False), \
                patch.object(downloader.requests, 'get', return_value=response) as get:
            records = []
            doc = downloader.fetch('https://query1.finance.yahoo.com/chart', 'raw.json', records)
            self.assertEqual(doc, json.loads(raw))
            self.assertEqual(get.call_args.kwargs['proxy'], 'http://127.0.0.1:7897')
            self.assertEqual(get.call_args.kwargs['impersonate'], 'chrome')
            self.assertEqual((Path(directory) / 'raw.json').read_bytes(), raw)
            self.assertEqual(records[0]['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(records[0]['final_url'], response.url)

    def test_proxy_bypass_is_respected(self):
        response = Mock(status_code=200, content=b'{}', url='https://example.invalid/chart')
        with TemporaryDirectory() as directory, patch.object(downloader, 'DATA', Path(directory)), \
                patch.object(downloader, 'getproxies', return_value={'https': 'http://127.0.0.1:7897'}), \
                patch.object(downloader, 'proxy_bypass', return_value=True), \
                patch.object(downloader.requests, 'get', return_value=response) as get:
            downloader.fetch('https://example.invalid/chart', 'raw.json', [])
            self.assertIsNone(get.call_args.kwargs['proxy'])

    def test_permission_denied_does_not_archive_html(self):
        for code in (401, 403):
            with self.subTest(code=code), TemporaryDirectory() as directory, \
                    patch.object(downloader, 'DATA', Path(directory)), \
                    patch.object(downloader, 'getproxies', return_value={}), \
                    patch.object(downloader, 'proxy_bypass', return_value=False), \
                    patch.object(downloader.requests, 'get', return_value=Mock(status_code=code)) as get:
                records = []
                with self.assertRaisesRegex(PermissionError, str(code)):
                    downloader.fetch('https://example.invalid/chart', 'raw.json', records)
                self.assertEqual(records, [])
                self.assertFalse((Path(directory) / 'raw.json').exists())
                self.assertEqual(get.call_count, 1)

    def test_rate_limit_is_reported_without_retries(self):
        response = Mock(status_code=429, headers={'Retry-After': '60'})
        with TemporaryDirectory() as directory, patch.object(downloader, 'DATA', Path(directory)), \
                patch.object(downloader, 'getproxies', return_value={}), \
                patch.object(downloader, 'proxy_bypass', return_value=False), \
                patch.object(downloader.requests, 'get', return_value=response) as get:
            with self.assertRaisesRegex(HTTPError, 'Retry-After=60'):
                downloader.fetch('https://example.invalid/chart', 'raw.json', [])
            self.assertEqual(get.call_count, 1)

    def test_existing_raw_file_is_checked_before_network(self):
        with TemporaryDirectory() as directory, patch.object(downloader, 'DATA', Path(directory)), \
                patch.object(downloader.requests, 'get') as get:
            artifact = Path(directory) / 'raw.json'
            artifact.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                downloader.fetch('https://example.invalid/chart', 'raw.json', [])
            get.assert_not_called()
            self.assertEqual(artifact.read_bytes(), b'original')

    def test_failed_attempt_is_archived_and_can_run_again(self):
        with TemporaryDirectory() as directory, patch.object(downloader, 'DATA', Path(directory)), \
                patch.object(downloader, 'download', side_effect=URLError('offline')) as download:
            root = Path(directory)
            original = b'{"status":"blocked_permission"}\n'
            (root / 'download_validation.json').write_bytes(original)
            raw = root / 'yahoo_NVDA_2023_2025.json'
            raw.write_bytes(b'partial-download')
            for _ in range(2):
                with self.assertRaises(URLError):
                    downloader.main()
            self.assertEqual(download.call_count, 2)
            archives = sorted((root / 'download_history').iterdir())
            self.assertEqual(len(archives), 2)
            self.assertEqual((archives[0] / 'download_validation.json').read_bytes(), original)
            self.assertEqual((archives[0] / raw.name).read_bytes(), b'partial-download')
            latest = json.loads((root / 'download_validation.json').read_text())
            self.assertEqual(latest['status'], 'download_failed')
            self.assertIn('offline', latest['error'])

    def test_successful_download_is_validated_and_reused_without_network(self):
        with TemporaryDirectory() as directory, patch.object(downloader, 'DATA', Path(directory)), \
                patch.object(downloader, 'download') as download, \
                patch('real_data.load_verified', return_value=(
                    {'source': 'unit-test-only'}, ['2023-01-03', '2025-12-31'], [1, 2])) as load, \
                patch('builtins.print'):
            verified = Path(directory) / 'verified_prices.json'
            verified.write_text('{}')
            downloader.main()
            load.assert_called_once_with(verified)
            download.assert_not_called()
            self.assertFalse((Path(directory) / 'download_history').exists())


if __name__ == '__main__':
    unittest.main()