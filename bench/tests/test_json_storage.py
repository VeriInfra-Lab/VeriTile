"""Storage format is independent of JSON authority, paths, and raw byte hashes."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter_common import canonical, checked, hash_file, load, relative, sha, write_json


class JsonStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_plain_bytes_and_json_semantics_are_preserved(self):
        data = {'schema': 1, 'nested': [{'unicode': 'λ世界', 'value': None}], 'number': 1.25}
        plain, compressed = [self.root / name for name in ('plain.json', 'compressed.json.gz')]
        write_json(plain, data)
        write_json(compressed, data)
        expected = (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        self.assertEqual(plain.read_bytes(), expected)
        self.assertEqual(gzip.decompress(compressed.read_bytes()), expected)
        self.assertEqual(load(plain), data)
        self.assertEqual(load(compressed), data)
        self.assertEqual(sha(canonical(load(plain))), sha(canonical(load(compressed))))

    def test_gzip_bytes_have_no_time_or_filename_dependency(self):
        paths = [self.root / name for name in ('one.json.gz', 'nested/two.json.gz')]
        for path in paths:
            write_json(path, {'x': list(range(100))})
        before = paths[0].read_bytes()
        write_json(paths[0], {'x': list(range(100))})
        self.assertEqual(before, paths[0].read_bytes())
        self.assertEqual(before, paths[1].read_bytes())
        self.assertEqual(before[:4], b'\x1f\x8b\x08\x00')  # FNAME flag absent.
        self.assertEqual(before[4:8], b'\0' * 4)

    def test_duplicate_keys_fail_at_every_depth_in_both_formats(self):
        for text in ('{"schema":1,"schema":2}', '{"a":[{"x":1,"x":2}]}'):
            for suffix in ('.json', '.json.gz'):
                with self.subTest(text=text, suffix=suffix):
                    path = self.root / ('duplicate' + suffix)
                    raw = text.encode()
                    path.write_bytes(gzip.compress(raw) if suffix.endswith('.gz') else raw)
                    with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                        load(path)

    def test_corrupt_gzip_fails_before_returning_json(self):
        valid = gzip.compress(b'{"schema":1}', mtime=0)
        crc = bytearray(valid); crc[-8] ^= 1
        length = bytearray(valid); length[-4] ^= 1
        corruptions = {
            'plain masquerading as gzip': b'{"schema":1}',
            'truncated header': valid[:4],
            'truncated stream': valid[:-9],
            'missing trailer': valid[:-8],
            'partial trailer': valid[:-1],
            'crc': bytes(crc),
            'length': bytes(length),
            'invalid deflate': valid[:10] + b'\x07' + valid[-8:],
            'trailing junk': valid + b'junk',
        }
        path = self.root / 'bad.json.gz'
        for label, data in corruptions.items():
            with self.subTest(label=label):
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, 'invalid gzip JSON'):
                    load(path)

    def test_invalid_json_and_utf8_still_fail(self):
        path = self.root / 'bad.json.gz'
        for text in (b'{', b'{} {}', b'\xff', b''):
            with self.subTest(text=text):
                path.write_bytes(gzip.compress(text))
                with self.assertRaises(ValueError):
                    load(path)

    def test_format_is_explicit_without_path_fallback(self):
        data = {'x': 1}
        write_json(self.root / 'only.json.gz', data)
        with self.assertRaises(FileNotFoundError):
            load(self.root / 'only.json')
        for name in ('unrelated.gz', 'upper.JSON.GZ', 'payload.json.gz.backup'):
            path = self.root / name
            write_json(path, data)
            self.assertEqual(path.read_bytes()[:1], b'{')
            self.assertEqual(load(path), data)
            path.write_bytes(gzip.compress(b'{}'))
            with self.assertRaises(ValueError):
                load(path)

    def test_raw_hash_binding_does_not_hash_decompressed_json(self):
        path = self.root / 'catalog.json.gz'
        write_json(path, {'schema': 1})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(hash_file(path), digest)
        self.assertEqual(load(checked(self.root, path.name, digest)), {'schema': 1})
        decompressed_digest = sha(gzip.decompress(path.read_bytes()))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            checked(self.root, path.name, decompressed_digest)
        # Metadata-only rewrite has identical JSON but must invalidate receipts.
        before = load(path)
        raw = bytearray(path.read_bytes()); raw[4] = 1; path.write_bytes(raw)
        self.assertEqual(load(path), before)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            checked(self.root, path.name, digest)

    def test_strict_relative_and_symlink_rules_still_apply(self):
        path = self.root / 'catalog.json.gz'
        write_json(path, {'schema': 1})
        digest = hash_file(path)
        for name in ('../catalog.json.gz', '/catalog.json.gz', './catalog.json.gz',
                     'nested//catalog.json.gz', 'nested/../catalog.json.gz',
                     'nested\\catalog.json.gz', 'catalog.json.gz\0', '', '.'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    relative(name)
                with self.assertRaises(ValueError):
                    checked(self.root, name, digest)
        inside = self.root / 'inside'; inside.mkdir()
        (inside / 'escape.json.gz').symlink_to(path)
        with self.assertRaisesRegex(ValueError, 'escaping file'):
            checked(inside, 'escape.json.gz', digest)
        with self.assertRaisesRegex(ValueError, 'missing/invalid hash'):
            checked(self.root, path.name, digest.upper())


if __name__ == '__main__':
    unittest.main()
