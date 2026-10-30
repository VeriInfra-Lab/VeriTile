"""Regression tests for rejection of incomplete or stale acceptance evidence."""
# Make the repository packages available for direct invocation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from scripts.bench.acceptance_tritonbench_g import REQUIRED_GATES, changed, digest, fp_contract_surface, inputs, run_gate, source_profile_inputs, successful, unregistered_modules


class AcceptanceBundleTests(unittest.TestCase):
    def source_profile_fixture(self, root):
        registry = root / 'bench/tritonbench_g/source_dtype_constexpr_profiles.json'
        registry.parent.mkdir(parents=True)
        base = root / 'bench/review-checkpoints/native'
        artifacts = {'source.py': b'original source', 'fp16/kernel.cubin': b'\x00\xff\x80',
                     'fp16/kernel.ttir': b'compiler IR', 'bf16/error.txt': b'rejected',
                     '__pycache__/original.pyc': b'\x80\x00\xff', 'opaque': b'no suffix'}
        for name, data in artifacts.items():
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        receipt = {'artifact_sha256': {name: digest(data) for name, data in artifacts.items()}}
        (base / 'receipt.json').write_text(json.dumps(receipt))
        registry.write_text(json.dumps({'schema': 1, 'profiles': {'fixture': {
            'evidence': base.relative_to(root).as_posix(),
            'receipt_sha256': digest((base / 'receipt.json').read_bytes())}}}))
        return registry, base, receipt

    def bind_receipt(self, registry, base, receipt):
        (base / 'receipt.json').write_text(json.dumps(receipt))
        record = json.loads(registry.read_text())
        record['profiles']['fixture']['receipt_sha256'] = digest((base / 'receipt.json').read_bytes())
        registry.write_text(json.dumps(record))

    def test_source_profile_closure_survives_archive_relocation_including_binary_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'original'
            registry, base, receipt = self.source_profile_fixture(root)
            # Neither unrelated checkpoints nor unbound artifacts are dependencies.
            (base / 'unbound.json').write_text('{}')
            before = inputs(root)
            expected = {registry.relative_to(root).as_posix(),
                        (base / 'receipt.json').relative_to(root).as_posix(),
                        *((base / name).relative_to(root).as_posix()
                          for name in receipt['artifact_sha256'])}
            self.assertEqual(set(before), expected)
            data = io.BytesIO()
            with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name in before:
                    archive.writestr(name, (root / name).read_bytes())
            relocated = Path(directory) / 'relocated'
            with zipfile.ZipFile(data) as archive:
                archive.extractall(relocated)
            self.assertEqual(inputs(relocated), before)
            self.assertEqual((relocated / base.relative_to(root) / 'fp16/kernel.cubin').read_bytes(),
                             b'\x00\xff\x80')

    def test_source_profile_receipt_and_each_artifact_require_exact_digest(self):
        for target in ('receipt.json', 'fp16/kernel.cubin', '__pycache__/original.pyc'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, base, _ = self.source_profile_fixture(root)
                (base / target).write_bytes((base / target).read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    inputs(root)

    def test_source_profile_missing_receipt_or_artifact_fails_closed(self):
        for target in ('receipt.json', 'fp16/kernel.cubin', '__pycache__/original.pyc'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, base, _ = self.source_profile_fixture(root)
                (base / target).unlink()
                with self.assertRaisesRegex(ValueError, 'Missing or out-of-bounds'):
                    inputs(root)

    def test_source_profile_paths_cannot_escape_checkpoint_or_artifact_directory(self):
        cases = [('evidence', '../outside'), ('evidence', '/tmp/outside'),
                 ('evidence', 'bench/tritonbench_g'), ('evidence', ''),
                 ('artifact', '../outside'), ('artifact', '/tmp/outside'),
                 ('artifact', 'nested/../../outside'), ('artifact', 'nested\\outside'),
                 ('artifact', '.')]
        for kind, name in cases:
            with self.subTest(kind=kind, name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, base, receipt = self.source_profile_fixture(root)
                if kind == 'evidence':
                    record = json.loads(registry.read_text())
                    record['profiles']['fixture']['evidence'] = name
                    registry.write_text(json.dumps(record))
                else:
                    receipt['artifact_sha256'] = {name: digest(b'outside')}
                    self.bind_receipt(registry, base, receipt)
                with self.assertRaises(ValueError):
                    inputs(root)

    def test_source_profile_symlinks_cannot_escape_evidence_boundary(self):
        for target in ('evidence', 'receipt.json', 'fp16/kernel.cubin'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'repo'
                registry, base, receipt = self.source_profile_fixture(root)
                if target == 'evidence':
                    outside = Path(directory) / 'outside'
                    base.rename(outside)
                    base.symlink_to(outside, target_is_directory=True)
                else:
                    # Even a file inside the repo but outside this evidence
                    # directory must not satisfy an artifact/receipt reference.
                    source = base / target
                    outside = root / 'outside'
                    source.rename(outside)
                    source.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'out-of-bounds'):
                    inputs(root)

    def test_source_profile_missing_or_malformed_hash_bindings_are_rejected(self):
        for kind in ('receipt', 'artifact'):
            for value in (None, '', 'z' * 64, 123):
                with self.subTest(kind=kind, value=value), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    registry, base, receipt = self.source_profile_fixture(root)
                    if kind == 'receipt':
                        record = json.loads(registry.read_text())
                        record['profiles']['fixture']['receipt_sha256'] = value
                        registry.write_text(json.dumps(record))
                    else:
                        receipt['artifact_sha256']['source.py'] = value
                        self.bind_receipt(registry, base, receipt)
                    with self.assertRaises(ValueError):
                        inputs(root)

    def test_source_profile_empty_or_malformed_artifact_manifest_is_rejected(self):
        for value in (None, {}, [], {'source.py': []}):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, base, receipt = self.source_profile_fixture(root)
                receipt['artifact_sha256'] = value
                self.bind_receipt(registry, base, receipt)
                with self.assertRaises(ValueError):
                    inputs(root)

    def test_source_profile_duplicate_registry_or_receipt_keys_are_rejected(self):
        for target in ('registry', 'receipt'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, base, _ = self.source_profile_fixture(root)
                if target == 'registry':
                    registry.write_text('{"schema": 1, "profiles": {}, "profiles": {}}')
                else:
                    (base / 'receipt.json').write_text('{"artifact_sha256": {}, "artifact_sha256": {}}')
                    record = json.loads(registry.read_text())
                    record['profiles']['fixture']['receipt_sha256'] = digest((base / 'receipt.json').read_bytes())
                    registry.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, 'Duplicate source-profile'):
                    inputs(root)

    def test_inline_profile_evidence_does_not_become_an_external_file_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / 'bench/tritonbench_g/source_pointer_dtype_profiles.json'
            registry.parent.mkdir(parents=True)
            registry.write_text(json.dumps({'schema': 1, 'profiles': {'inline': {
                'receipt': {'native_source_path': '/not/a/dependency.py'},
                'receipt_sha256': 'inline receipt checked by source gate',
                'ttir': 'loc("/not/a/dependency.py")'}}}))
            self.assertEqual(set(inputs(root)), {registry.relative_to(root).as_posix()})

    def test_source_profile_registry_cannot_silently_drop_external_bindings(self):
        for change in ('schema', 'profiles', 'record', 'evidence'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                registry, _, _ = self.source_profile_fixture(root)
                record = json.loads(registry.read_text())
                if change == 'schema':
                    record['schema'] = True
                elif change == 'profiles':
                    record['profiles'] = []
                elif change == 'record':
                    record['profiles']['fixture'] = None
                else:
                    del record['profiles']['fixture']['evidence']
                registry.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    inputs(root)

    def test_source_profile_mutation_after_closure_cannot_enter_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, base, _ = self.source_profile_fixture(root)

            def change_after_validation(path):
                bound = source_profile_inputs(path)
                (base / 'fp16/kernel.cubin').write_bytes(b'changed after validation')
                return bound

            with patch('scripts.bench.acceptance_tritonbench_g.source_profile_inputs', change_after_validation):
                with self.assertRaisesRegex(ValueError, 'changed during snapshot'):
                    inputs(root)

    def test_math_or_cast_only_cannot_pass_fp_presence_even_without_extra_laws(self):
        for status in ('math_contract_only', 'rounding_contract_only',
                       'requires_numeric_model_revision', 'missing_source'):
            with self.subTest(status=status):
                result = fp_contract_surface([{'task': 'example', 'accepted_scope_review': {
                    'fp_adequacy_status': status, 'numerical_status': 'no_extra_numeric_law'}}])
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['tasks_with_fp_contracts'], 0)

    def test_relative_fp_presence_does_not_claim_native_functional_correctness(self):
        result = fp_contract_surface([{'task': 'example', 'accepted_scope_review': {
            'fp_adequacy_status': 'backend_equivalence_existing_reports'}}])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['native_functional_correctness'], 'not_established_by_this_gate')

    def test_one_typed_stage_cannot_hide_remaining_accepted_stage_gaps(self):
        for status in ('backend_value_contract_only', 'bit_interpreter_contract_only'):
            task = {'task': 'example', 'accepted_scope_review': {
                'fp_adequacy_status': status, 'accepted_scope_fp_coverage': 'partial'}}
            result = fp_contract_surface([task])
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['tasks_without_adequate_fp_contracts'][0]['reason'],
                             'previously_accepted_stages_still_lack_fp_contracts')
            task['accepted_scope_review']['accepted_scope_fp_coverage'] = 'complete_in_declared_profile'
            result = fp_contract_surface([task])
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['native_functional_correctness'], 'not_established_by_this_gate')

    def test_missing_fp_review_or_empty_corpus_cannot_pass_presence(self):
        with self.assertRaisesRegex(ValueError, 'Missing or invalid FP contract review'):
            fp_contract_surface([{'task': 'example'}])
        self.assertEqual(fp_contract_surface([])['status'], 'failed')

    def test_compiler_failure_stops_remaining_jobs_and_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, '-c',
                       "import time; print('error: Lean exited with code 1', flush=True); time.sleep(20)"]
            gate = run_gate('build', command, Path(directory))
            self.assertEqual(gate['status'], 'failed')
            self.assertNotEqual(gate['exit_code'], 0)
            self.assertTrue(gate['remaining_build_jobs_stopped_after_failure'])
            self.assertIn('error: Lean exited', (Path(directory) / 'build.log').read_text())

    def test_incomplete_failed_or_skipped_gates_never_pass(self):
        valid = {'gates': {g: {'status': 'passed', 'exit_code': 0} for g in REQUIRED_GATES},
                 'changed_inputs': [], 'missing_sources': []}
        self.assertTrue(successful(valid))
        for gate in REQUIRED_GATES:
            for replacement in (None, {'status': 'not_run'},
                                {'status': 'failed', 'exit_code': 1},
                                {'status': 'passed', 'exit_code': 1}):
                with self.subTest(gate=gate, replacement=replacement):
                    receipt = copy.deepcopy(valid)
                    if replacement is None:
                        del receipt['gates'][gate]
                    else:
                        receipt['gates'][gate] = replacement
                    self.assertFalse(successful(receipt))
        for field in ('changed_inputs', 'missing_sources'):
            receipt = copy.deepcopy(valid)
            receipt[field] = ['missing_or_changed']
            self.assertFalse(successful(receipt))

    def test_new_deleted_and_changed_proofs_invalidate_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proofs = root / 'bench/tritonbench_g/example'
            proofs.mkdir(parents=True)
            original = proofs / 'Original.lean'
            original.write_text('old')
            before = inputs(root)
            original.write_text('new')
            added = proofs / 'New.lean'
            added.write_text('new proof')
            self.assertEqual(len(changed(before, inputs(root))), 2)
            original.unlink()
            self.assertEqual(len(changed(before, inputs(root))), 2)
            archive = root / 'bench/review-checkpoints/inputs'
            archive.mkdir(parents=True)
            (archive / 'Historical.lean').write_text('old evidence')
            self.assertEqual(len(inputs(root)), 1)

    def test_unregistered_legacy_or_new_module_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lakefile.toml').write_text('''[[lean_lib]]
name = "TritonBenchGVerified"
globs = ["bench.tritonbench_g.covered.+", "bench.tritonbench_g.legacy.Old"]
''')
            for name in ('covered/Correct', 'covered/Proofs/Body', 'legacy/Old', 'missed/New'):
                p = root / ('bench/tritonbench_g/' + name + '.lean')
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('')
            self.assertEqual(unregistered_modules(root), ['bench.tritonbench_g.missed.New'])

    def test_numerical_review_captures_bound_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / 'bench/tritonbench_g/numerical_obligations_review.json'
            review.parent.mkdir(parents=True)
            name = 'bench/review-checkpoints/numerical-audit/observed.json'
            evidence = root / name
            evidence.parent.mkdir(parents=True)
            evidence.write_text('old evidence')
            review.write_text(json.dumps({'input_sha256': {name: 'checked by the review gate'}}))
            before = inputs(root)
            self.assertIn(name, before)
            evidence.write_text('changed evidence')
            self.assertEqual(changed(before, inputs(root)), [name])

    def test_numerical_review_cannot_archive_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            review = root / 'bench/tritonbench_g/numerical_obligations_review.json'
            review.parent.mkdir(parents=True)
            outside = Path(directory) / 'outside.json'
            outside.write_text('{}')
            review.write_text(json.dumps({'input_sha256': {'../outside.json': 'x'}}))
            with self.assertRaisesRegex(ValueError, 'Invalid bound numerical-review input'):
                inputs(root)

    def test_inventory_captures_excluded_generated_index_and_relocates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'original'
            report = root / 'bench/tritonbench_g/current_completion_inventory.json'
            name = 'documents/bench/spec-sheets/README.md'
            index = root / name
            index.parent.mkdir(parents=True)
            index.write_text('generated index explicitly fingerprinted by inventory')
            (index.parent / 'Unbound.md').write_text('not an inventory dependency')
            report.write_text(json.dumps({'input_fingerprints_sha256': {
                name: digest(index.read_bytes())}}))
            before = inputs(root)
            self.assertEqual(set(before), {name, report.relative_to(root).as_posix()})
            self.assertEqual(before[name], digest(index.read_bytes()))
            data = io.BytesIO()
            with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in before:
                    archive.writestr(path, (root / path).read_bytes())
            relocated = Path(directory) / 'relocated'
            with zipfile.ZipFile(data) as archive:
                archive.extractall(relocated)
            self.assertEqual(inputs(relocated), before)

    def test_inventory_dependency_paths_cannot_escape_workspace(self):
        for kind in ('parent', 'absolute', 'symlink'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'repo'
                report = root / 'bench/tritonbench_g/current_completion_inventory.json'
                report.parent.mkdir(parents=True)
                outside = Path(directory) / 'outside.md'
                outside.write_text('outside the snapshot root')
                if kind == 'parent':
                    name = '../outside.md'
                elif kind == 'absolute':
                    name = str(outside)
                else:
                    (root / 'linked.md').symlink_to(outside)
                    name = 'linked.md'
                report.write_text(json.dumps({'input_fingerprints_sha256': {
                    name: digest(outside.read_bytes())}}))
                with self.assertRaisesRegex(ValueError, 'Invalid bound inventory input'):
                    inputs(root)


if __name__ == '__main__':
    unittest.main()
