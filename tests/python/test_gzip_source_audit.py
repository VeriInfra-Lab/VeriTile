"""Exercise compressed catalog forwarding and raw-byte audit postchecks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'bench'))
from tests.python import test_current_source_audit as fixtures
from tests.python import test_current_source_adapters as adapter_fixtures
from tests.python import test_current_value_metadata as value_fixtures
from tests.python import test_original_program_scope as scope_fixtures
from scripts.bench.adapter_common import canonical, hash_file, load, sha, write_json
from scripts.bench.current_source_audit import run
from scripts.bench.current_source_catalog import validate_supplement


class GzipSourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CurrentSourceAuditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args = self.fixture.args('gzip-audit')
        self.args.catalog = self.fixture.root / 'catalog.json.gz'
        self.args.supplement = self.fixture.root / 'supplement.json.gz'
        self.args.scope_ledger = self.fixture.root / 'scope.json.gz'
        write_json(self.args.catalog, self.fixture.catalog)

    def test_global_runner_preserves_explicit_compressed_paths(self):
        commands = {}
        def fixture_gate(root, output, name, command):
            commands[name] = command
            return {'status': 'passed', 'exit_code': 0}
        with patch('scripts.bench.current_source_audit.command_gate', side_effect=fixture_gate):
            result = run(self.args)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['catalog_sha256'], sha(canonical(self.fixture.catalog)))
        for name in ('catalog_reproduction', 'scope_review', 'legacy_source', 'value_metadata'):
            self.assertIn(str(self.args.catalog), commands[name])
        for name in ('catalog_reproduction', 'value_metadata'):
            self.assertIn(str(self.args.supplement), commands[name])
        self.assertIn(str(self.args.scope_ledger), commands['scope_review'])
        self.assertEqual(result['corpus_replay'], 'not_run_by_global_runner')

    def test_metadata_only_gzip_rewrite_fails_raw_byte_postcheck(self):
        def fixture_gate(root, output, name, command):
            if name == 'library_replay':
                raw = bytearray(self.args.catalog.read_bytes())
                raw[4] ^= 1
                self.args.catalog.write_bytes(raw)
            return {'status': 'passed', 'exit_code': 0}
        with patch('scripts.bench.current_source_audit.command_gate', side_effect=fixture_gate):
            result = run(self.args)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['gates']['unchanged_inputs']['status'], 'failed')
        self.assertIn('catalog file changed', result['gates']['unchanged_inputs']['errors'][0])

    def test_corrupt_catalog_runs_no_subprocess_gate(self):
        self.args.catalog.write_bytes(self.args.catalog.read_bytes()[:-1])
        with patch('scripts.bench.current_source_audit.command_gate') as gate:
            with self.assertRaisesRegex(ValueError, 'invalid gzip JSON'):
                run(self.args)
            gate.assert_not_called()


class GzipBoundReviewTests(unittest.TestCase):
    def compress_review(self, root, reference):
        plain = root / reference['path']
        compressed = root / (reference['path'] + '.gz')
        write_json(compressed, load(plain))
        reference.update(path=compressed.name, sha256=hash_file(compressed))
        plain.unlink()  # No fallback to the old path may hide a broken reader.
        return compressed

    def fixture(self, cls):
        fixture = cls()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_supplement_consumes_compressed_hash_bound_root_review(self):
        fixture = self.fixture(adapter_fixtures.AdapterTest)
        supplement = fixture.supplement()
        reference = supplement['tasks'][0]['review']
        path = self.compress_review(fixture.root, reference)
        result = validate_supplement(fixture.root, fixture.root, supplement, {'t'})
        self.assertEqual(result['t']['declarations'][0]['model_kind'], 'protocol_contract')
        raw = bytearray(path.read_bytes()); raw[4] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_supplement(fixture.root, fixture.root, supplement, {'t'})

    def test_value_scope_review_preserves_exact_scope_checks(self):
        fixture = self.fixture(value_fixtures.CurrentValueMetadataTests)
        before = fixture.check()
        self.compress_review(fixture.root, fixture.task['review'])
        self.assertEqual(fixture.check(), before)
        fixture.task['declarations'][0]['dtype_profile'] = 'unreviewed wider scope'
        with self.assertRaisesRegex(ValueError, 'lack exact current review'):
            fixture.check()

    def test_original_scope_review_keeps_canonical_catalog_binding(self):
        fixture = self.fixture(scope_fixtures.OriginalProgramScopeTests)
        self.compress_review(fixture.root, fixture.row['review'])
        for name, value in (('catalog', fixture.catalog), ('row', fixture.row)):
            path = fixture.root / (name + '.json.gz')
            write_json(path, value)
            setattr(fixture, name, load(path))
        self.assertEqual(fixture.validate(), 'verified')
        fixture.row['coverage']['branches'][0]['explanation'] = 'changed review'
        with self.assertRaisesRegex(ValueError, 'exact current obligations/evidence'):
            fixture.validate()


if __name__ == '__main__':
    unittest.main()
