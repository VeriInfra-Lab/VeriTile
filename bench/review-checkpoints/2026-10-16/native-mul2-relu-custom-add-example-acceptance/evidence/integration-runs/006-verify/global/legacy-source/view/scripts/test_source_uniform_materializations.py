"""Complete direct Philox source materialization and production gate tests.

The fixture preserves the full upstream module and actual macro. Production
tests execute the unchanged extraction and 23 source gates in an explicitly
source-only harness, with no fabricated theorem files or proof receipts.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source
from bench import source_materializations as material
from bench import source_uniform_materializations as uniform
from bench.port_layout import source_text

FIXTURES = ROOT / 'scripts/fixtures/source_uniform_materializations'
PYTHON = (FIXTURES / 'uniform_sampling.py.txt').read_text()
LEAN = (FIXTURES / 'Kernels.lean.txt').read_text()
ADAPTER_FIXTURE = FIXTURES / 'RandomProfile-20261014-signed.lean.txt'


def fixture(storage='tl.float32'):
    args = {'storage': storage, 'version': 'original'}
    profile = {'kind': uniform.KIND, 'source_sha256': uniform.SOURCE_SHA256,
               'lean': {'source': uniform.SELECTOR, 'args': args},
               'regions': {'out_ptr': storage}, 'bindings': uniform.bindings(),
               'helper': uniform.HELPER.copy(), 'params': uniform.PARAMETER_MAP.copy()}
    header = '/-!\nSource-audit target: uniform_kernel\nSource-audit Lean: uniformSource%\n'
    header += 'Source-audit args: storage=' + storage + ', version=original\n'
    header += 'Source-audit params: to=upper\n'
    header += 'Source-audit materializations: ' + json.dumps(profile, separators=(',', ':')) + '\n-/\n'
    return PYTHON, header + LEAN[LEAN.index('import VeriTile.'):], profile


def replace_profile(lean, profile):
    return re.sub(r'^Source-audit materializations: .*$',
                  'Source-audit materializations: ' + json.dumps(profile), lean, flags=re.M)


class UniformMaterializationTests(unittest.TestCase):
    def setUp(self):
        adapter_patch = patch.object(material, '_adapter_path', return_value=ADAPTER_FIXTURE)
        adapter_patch.start()
        self.addCleanup(adapter_patch.stop)

    def reject(self, lean=None, python=PYTHON, profile=None):
        _, baseline, _ = fixture()
        text = baseline if lean is None else lean
        if profile is not None:
            text = replace_profile(text, profile)
        with self.assertRaises(ValueError):
            material.python_source_with_materializations(python, text)

    def test_frozen_complete_source_macro_and_reviewed_adapter(self):
        self.assertEqual(hashlib.sha256(PYTHON.encode()).hexdigest(), uniform.SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(LEAN.encode()).hexdigest(),
                         '67ba816fa56a773feb128b8b424529a7ba2b8aae1f4086745c0876959832e43c')
        self.assertEqual(hashlib.sha256((FIXTURES / 'SourceDomain.lean.txt').read_bytes()).hexdigest(),
                         'af632e51119b3d778eef8e847e800ffd01f9adb199f571b874c1bde7476cd2e3')
        self.assertEqual(hashlib.sha256(ADAPTER_FIXTURE.read_bytes()).hexdigest(), material.ADAPTER_SHA256)
        self.assertEqual(hashlib.sha256((ROOT / material.ADAPTER_PATH).read_bytes()).hexdigest(),
                         material.ADAPTER_SHA256)

    def test_complete_source_keeps_rebindings_tuple_helper_decorators_and_host(self):
        py, lean, _ = fixture()
        output = material.python_source_with_materializations(py, lean)
        original = source._explicit_python_audit_kernel(py, lean)
        changed = source._explicit_python_audit_kernel(output, lean)
        self.assertEqual(ast.dump(original.args), ast.dump(changed.args))
        self.assertEqual([ast.dump(x) for x in original.decorator_list],
                         [ast.dump(x) for x in changed.decorator_list])
        self.assertEqual(len(original.body), 20)
        self.assertEqual(len(changed.body), 23)
        for kind in (ast.AugAssign, ast.Tuple, ast.Call, ast.BinOp):
            self.assertEqual(sum(isinstance(x, kind) for x in ast.walk(original)),
                             sum(isinstance(x, kind) for x in ast.walk(changed)))
        self.assertEqual([ast.dump(x) for x in ast.parse(py).body if getattr(x, 'name', None) != original.name],
                         [ast.dump(x) for x in ast.parse(output).body if getattr(x, 'name', None) != original.name])
        self.assertIn('philox_seed = philox_seed_arg.to(tl.int64)', output)
        self.assertIn('philox_offset = philox_offset_arg.to(tl.int64)', output)
        self.assertIn('c0 += i4', output)
        self.assertIn('r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)', output)
        self.assertEqual(output.count("eviction_policy='evict_first'"), 4)
        self.assertIn('(to - from_)', output)  # original parameter is not erased/renamed here
        self.assertEqual(dict(source.source_parameter_map(output, lean)), {'to': 'upper'})

    def test_all_six_explicit_storage_selections_preserve_the_complete_ast(self):
        for storage in sorted(material.STORAGE_DTYPES):
            with self.subTest(storage=storage):
                py, lean, _ = fixture(storage)
                self.assertIn('c0 += i4', material.python_source_with_materializations(py, lean))

    def test_full_original_hash_rejects_helper_host_and_kernel_changes_even_rehashed(self):
        _, lean, profile = fixture()
        for before, after in [('4.6566127342e-10', '4.0e-10'),
                              ('1.0842020432385337e-19', '1.0e-19'),
                              ('except AttributeError:', 'except Exception:'),
                              ('c0 += i4', 'c0 = i4'),
                              ('return 512', 'return 256'),
                              ('"philox_seed", "philox_offset"', '"philox_seed"')]:
            with self.subTest(after=after):
                self.assertIn(before, PYTHON)
                changed = PYTHON.replace(before, after, 1)
                self.reject(lean, changed)
                bad = copy.deepcopy(profile)
                bad['source_sha256'] = hashlib.sha256(changed.encode()).hexdigest()
                self.reject(python=changed, profile=bad)

    def test_exact_schema_selection_parameter_map_and_composition(self):
        _, lean, profile = fixture()
        for bad in ({}, [], dict(profile, extra=True), dict(profile, kind='unknown'),
                    dict(profile, regions={}), dict(profile, bindings=[]),
                    dict(profile, helper={}), dict(profile, params={})):
            self.reject(profile=bad)
        for before, after in [('version=original', 'version=indexCommuted'),
                              ('Source-audit params: to=upper', 'Source-audit params: to=from_'),
                              ('Source-audit params: to=upper\n', ''),
                              ('Source-audit Lean: uniformSource%', 'Source-audit Lean: absent%')]:
            self.reject(lean=lean.replace(before, after, 1))
        for field in ('constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes'):
            self.reject(lean=lean.replace('Source-audit target:',
                                          'Source-audit ' + field + ': {}\nSource-audit target:', 1))

    def test_materialization_prefix_metadata_identity_and_carriers_are_exact(self):
        _, lean, profile = fixture()
        mutations = [
            ('tl.source_helper uint_to_uniform_float = tl.uint_to_uniform_float', ''),
            ('tl.source_helper uint_to_uniform_float', 'tl.source_helper other_helper'),
            ('tl.source_integer philox_seed_arg = tl.int64', 'tl.source_integer philox_seed_arg = tl.uint64'),
            ('$((philox_offset : Int))', '$((philox_seed : Int))'),
            ('i32ArangeSize BLOCK (arange_domain hBlock)', 'i32ArangeSize (BLOCK + 1) (arange_domain hBlock)'),
            ('i32ArangeSize BLOCK (arange_domain hBlock)', 'i32ArangeSize BLOCK (other hBlock)'),
            ('SourceDomain.LegalBlock BLOCK', 'SourceDomain.LegalBlock N'),
            ('philox_seed philox_offset : Int', 'philox_seed philox_offset : Nat'),
            ('$((upper : ℝ))', '$((upper + 1 : ℝ))'),
            ('$((upper : ℝ))', '$((upper : Nat))'),
        ]
        for before, after in mutations:
            with self.subTest(after=after):
                self.assertIn(before, lean)
                self.reject(lean=lean.replace(before, after, 1))
        for index, field, value in [(0, 'initial_cast', 'philox_offset'), (1, 'dtype', 'tl.uint64'),
                                    (2, 'adapter_sha256', '0' * 64)]:
            bad = copy.deepcopy(profile)
            bad['bindings'][index][field] = value
            self.reject(profile=bad)

    def test_complete_tree_rejects_bitwise_update_tuple_helper_store_and_hint_changes(self):
        _, lean, _ = fixture()
        mutations = [
            ('philox_seed_arg.to(tl.int64)', 'philox_seed.to(tl.int64)'),
            ('philox_offset >> 32', 'philox_offset >> 31'),
            ('& 0xFFFFFFFF', '& 0xFFFFFFFE'),
            ('c0 += i4', 'c0 = c0 + i4'),
            ('_O = c0 * 0', '_O = 0'),
            ('r0, r1, r2, r3 = tl.philox', 'r0, r2, r1, r3 = tl.philox'),
            ('tl.philox(philox_seed, c0, c1, _O, _O)', 'tl.philox(philox_seed, c0, c1, _O, c0)'),
            ('uint_to_uniform_float(r2)', 'uint_to_uniform_float(r1)'),
            ('$((upper : ℝ)) - $((from_ : ℝ))', '$((upper : ℝ)) + $((from_ : ℝ))'),
            ('off_2 = off_1 + BLOCK_arg', 'off_2 = off_0 + BLOCK_arg'),
            ('mask=off_3 < $(N)', 'mask=off_2 < $(N)'),
            ('eviction_policy="evict_first"', 'eviction_policy="evict_last"'),
            ('(r3).to($storage:tritonDType)', '(r2).to($storage:tritonDType)'),
            ('(r1).to($storage:tritonDType)', '(r1).to(tl.float64)'),
            ('  c0 += i4\n', ''),
            ('  _O = c0 * 0', '  extra = 0\n  _O = c0 * 0'),
        ]
        for before, after in mutations:
            with self.subTest(after=after):
                self.assertIn(before, lean)
                self.reject(lean=lean.replace(before, after, 1))

    def test_adapter_source_or_self_updated_metadata_cannot_bless_a_change(self):
        _, _, profile = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'RandomProfile.lean'
            path.write_text(ADAPTER_FIXTURE.read_text().replace(
                '(_h : ArangeSizeDomain n) : Nat := n', '(_h : ArangeSizeDomain n) : Nat := n + 1'))
            with patch.object(material, '_adapter_path', return_value=path):
                self.reject()
                bad = copy.deepcopy(profile)
                bad['bindings'][2]['adapter_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.reject(profile=bad)

    def test_absent_profile_does_not_transform_the_original(self):
        py, lean, _ = fixture()
        lean = re.sub(r'^Source-audit materializations: .*\n', '', lean, flags=re.M)
        self.assertEqual(material.python_source_with_materializations(py, lean), py)


class ProductionUniformMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.temporary.name) / 'repo'
        (cls.repo / 'bench/tritonbench_g').mkdir(parents=True)
        for path in (ROOT / 'bench').glob('*.py'):
            shutil.copyfile(path, cls.repo / 'bench' / path.name)
        adapter = cls.repo / material.ADAPTER_PATH
        adapter.parent.mkdir(parents=True)
        shutil.copyfile(ADAPTER_FIXTURE, adapter)
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        section = script[script.index('# A source view must be extractable'):
                         script.index('# ---- Machine-checkable trust audit')]
        cls.command = 'failures=0\nPORTS_ROOT="$1"\n' + section + '\nexit "$failures"\n'

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def production(self, lean, python=PYTHON):
        with tempfile.TemporaryDirectory(dir=self.repo / 'bench/tritonbench_g') as temporary:
            port = Path(temporary) / 'uniform_sampling'
            port.mkdir()
            (port / 'UniformSampling.lean').write_text(lean)
            (port / 'uniform_sampling.py').write_text(python)
            # Preserve the actual local import, even in this source-only
            # harness. The imported proof source is frozen, never fabricated.
            (port / 'Proofs').mkdir()
            shutil.copyfile(FIXTURES / 'SourceDomain.lean.txt', port / 'Proofs/SourceDomain.lean')
            return subprocess.run(['bash', '-c', self.command, 'uniform-source-only-harness', temporary],
                                  cwd=self.repo, capture_output=True, text=True, timeout=120)

    def test_complete_source_passes_all_production_source_gates(self):
        _, lean, _ = fixture()
        result = self.production(lean)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sum(line.startswith('ok ') for line in result.stdout.splitlines()), 24)
        self.assertIn('1 extractable source views, 0 unresolved source views', result.stdout)

    def test_actual_port_entry_all_storage_asts_and_selected_production_gates(self):
        directory = ROOT / 'bench/tritonbench_g/uniform_sampling'
        # Public correctness entries are a separate port-layout obligation.
        # Read this actual source entry and its imports without fabricating
        # Correct/FPEquiv files merely to satisfy split-port discovery.
        entry = directory / 'Kernels.lean'
        self.assertTrue(entry.is_file())
        self.assertEqual(entry.resolve().parent, directory.resolve())
        lean = source_text(entry)
        python = (directory / 'uniform_sampling.py').read_text()
        profile = source._source_profile_json(
            source.source_audit_metadata(lean, 'materializations'), 'materializations')
        self.assertEqual(profile['kind'], uniform.KIND)
        self.assertEqual(profile['lean'], {'source': uniform.SELECTOR,
                                         'args': {'storage': 'tl.float32', 'version': 'original'}})
        # Each selection is an explicit in-memory view of the current entry;
        # no source statement, runtime parameter or helper branch is removed.
        for storage in sorted(material.STORAGE_DTYPES):
            with self.subTest(storage=storage):
                changed = copy.deepcopy(profile)
                changed['lean']['args']['storage'] = storage
                changed['regions']['out_ptr'] = storage
                view = re.sub(r'^Source-audit args: .*$',
                              'Source-audit args: storage=' + storage + ', version=original', lean, flags=re.M)
                view = replace_profile(view, changed)
                result = material.python_source_with_materializations(python, view)
                kernel = source._explicit_python_audit_kernel(result, view)
                self.assertEqual(len(kernel.body), 23)
        # The actual declared selection remains fp32. These 24 source gates
        # do not constitute all-storage/native/ABI or theorem acceptance.
        result = self.production(lean, python)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sum(line.startswith('ok ') for line in result.stdout.splitlines()), 24)
        self.assertIn('1 extractable source views, 0 unresolved source views', result.stdout)

    def test_absent_profile_does_not_hide_original_assignments(self):
        _, lean, _ = fixture()
        lean = re.sub(r'^Source-audit materializations: .*\n', '', lean, flags=re.M)
        result = self.production(lean)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL statement lhs sequence scan', result.stdout)

    def test_translation_marker_cannot_bypass_the_complete_ast_gate(self):
        _, lean, _ = fixture()
        lean = lean.replace('/-!\n', '/-!\nTranslation-surface blocker: adversarial diagnostic\n', 1)
        lean = lean.replace('c0 += i4', 'c0 = i4')
        result = self.production(lean)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL source-view extraction scan', result.stdout)
        self.assertIn('complete source statement/expression AST differs', result.stdout)


if __name__ == '__main__':
    unittest.main()
