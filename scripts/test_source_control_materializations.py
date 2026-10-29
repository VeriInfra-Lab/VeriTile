"""Full-branch layer-norm source identity audit, including adversarial inputs.

The frozen fixture is the actual complete source/macro, not a miniature
straight-line surrogate. Production tests run the unchanged source gates in a
clearly identified source-only legacy harness. No theorem files are fabricated.
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
from bench import source_control_materializations as guarded
from bench.port_layout import source_file, source_text

FIXTURES = ROOT / 'scripts/fixtures/source_control_materializations'
PYTHON = (FIXTURES / 'layer_norm_fwd.py.txt').read_text()
LEAN = (FIXTURES / 'Kernels.lean.txt').read_text()
ADAPTER_FIXTURE = FIXTURES / 'RandomProfile-20261014-signed.lean.txt'


def fixture(storage='tl.float32', **overrides):
    args = dict.fromkeys(guarded.DTYPE_ARGUMENTS, storage)
    args.update(overrides)
    args['version'] = 'original'
    profile = {
        'kind': guarded.KIND,
        'source_sha256': guarded.SOURCE_SHA256,
        'lean': {'source': guarded.SELECTOR, 'args': args},
        'regions': {name: args[arg] for name, arg in guarded.FLOAT_REGIONS.items()}
                   | guarded.FIXED_REGIONS,
        'binding': {'python': 'BLOCK_N', 'local': 'BLOCK_N_arg', 'kind': 'constexpr_identity',
                    'adapter': material.ADAPTER, 'adapter_sha256': material.ADAPTER_SHA256,
                    'proof': 'hBlock', 'before': 'cols'},
    }
    header = '/-!\nSource-audit target: ' + guarded.TARGET + '\n'
    header += 'Source-audit Lean: ' + guarded.SELECTOR + '\n'
    header += 'Source-audit args: ' + ', '.join(k + '=' + v for k, v in args.items()) + '\n'
    header += 'Source-audit materializations: ' + json.dumps(profile, separators=(',', ':')) + '\n-/\n'
    return PYTHON, header + LEAN, profile


def replace_profile(lean, profile):
    return re.sub(r'^Source-audit materializations: .*$',
                  'Source-audit materializations: ' + json.dumps(profile), lean, flags=re.M)


class GuardedMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.adapter_patch = patch.object(material, '_adapter_path', return_value=ADAPTER_FIXTURE)
        self.adapter_patch.start()
        self.addCleanup(self.adapter_patch.stop)

    def reject(self, lean=None, python=PYTHON, profile=None):
        _, baseline, _ = fixture()
        text = baseline if lean is None else lean
        if profile is not None:
            text = replace_profile(text, profile)
        with self.assertRaises(ValueError):
            material.python_source_with_materializations(python, text)

    def test_frozen_actual_source_macro_and_reviewed_adapter(self):
        self.assertEqual(hashlib.sha256(PYTHON.encode()).hexdigest(), guarded.SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(LEAN.encode()).hexdigest(),
                         'edacf9764b98ebb7e84940217aed41cecd2adf1ae91a2d14b4f529f517ff0f8a')
        self.assertEqual(hashlib.sha256(ADAPTER_FIXTURE.read_bytes()).hexdigest(),
                         material.ADAPTER_SHA256)
        self.assertEqual(hashlib.sha256((ROOT / material.ADAPTER_PATH).read_bytes()).hexdigest(),
                         material.ADAPTER_SHA256)

    def test_complete_source_keeps_signature_decorators_all_branches_and_host(self):
        py, lean, _ = fixture()
        output = material.python_source_with_materializations(py, lean)
        original = source._explicit_python_audit_kernel(py, lean)
        changed = source._explicit_python_audit_kernel(output, lean)
        self.assertEqual(ast.dump(original.args), ast.dump(changed.args))
        self.assertEqual([ast.dump(x) for x in original.decorator_list],
                         [ast.dump(x) for x in changed.decorator_list])
        self.assertEqual(len(changed.body), len(original.body) + 1)
        for kind in (ast.If, ast.IfExp, ast.AugAssign, ast.Call):
            self.assertEqual(sum(isinstance(x, kind) for x in ast.walk(original)),
                             sum(isinstance(x, kind) for x in ast.walk(changed)))
        self.assertEqual({x.id for x in ast.walk(changed) if isinstance(x, ast.Name)} & set(guarded.FLAGS),
                         set(guarded.FLAGS))
        self.assertEqual([ast.dump(x) for x in ast.parse(py).body if getattr(x, 'name', None) != original.name],
                         [ast.dump(x) for x in ast.parse(output).body if getattr(x, 'name', None) != original.name])
        self.assertIn('tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7)', output)
        self.assertIn('tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7)', output)
        self.assertIn('BLOCK_N_arg = BLOCK_N\n    cols = tl.arange(0, BLOCK_N_arg)', output)
        self.assertIn('x += residual', output)
        self.assertIn('x1 *= rowscale', output)

    def test_destination_storage_choices_require_complete_explicit_macro_arguments(self):
        for kwargs in ({}, {'output': 'tl.float16', 'residualOut': 'tl.bfloat16'},
                       {'input': 'tl.float64', 'input1': 'tl.float8e5', 'weight': 'tl.float8e4nv'}):
            with self.subTest(kwargs=kwargs):
                py, lean, _ = fixture(**kwargs)
                self.assertIn('cols = tl.arange(0, BLOCK_N_arg)',
                              material.python_source_with_materializations(py, lean))

    def test_absent_metadata_keeps_original_unmodified(self):
        py, lean, _ = fixture()
        lean = re.sub(r'^Source-audit materializations: .*\n', '', lean, flags=re.M)
        self.assertEqual(material.python_source_with_materializations(py, lean), py)
        self.assertIn('BLOCK_N_arg =', source.lean_first_triton_body(source.select_lean_audit_source(lean)))

    def test_full_original_hash_cannot_be_changed_or_self_rehashed(self):
        _, lean, profile = fixture()
        for changed in (PYTHON + '\n', PYTHON.replace('n_rounds=7', 'n_rounds=10'),
                        PYTHON.replace('x += residual', 'x = residual'),
                        PYTHON.replace('num_warps=32', 'num_warps=64')):
            self.reject(lean=lean, python=changed)
            updated = copy.deepcopy(profile)
            updated['source_sha256'] = hashlib.sha256(changed.encode()).hexdigest()
            self.reject(python=changed, profile=updated)

    def test_profile_schema_binding_and_selector_are_exact(self):
        _, lean, profile = fixture()
        for bad in ({}, [], dict(profile, extra=True), dict(profile, kind='unknown'),
                    dict(profile, regions={}), dict(profile, binding={}),
                    dict(profile, source_sha256=False)):
            self.reject(profile=bad)
        for field, value in [('python', 'N'), ('local', 'other'), ('before', 'row'),
                             ('proof', 'otherProof'), ('adapter_sha256', '0' * 64)]:
            bad = copy.deepcopy(profile)
            bad['binding'][field] = value
            self.reject(profile=bad)
        for changed in (lean.replace('version=original', 'version=indexCommuted', 1),
                        lean.replace(', version=original', '', 1),
                        lean.replace('Source-audit Lean: layerNormFwdSource%', 'Source-audit Lean: absent%'),
                        lean.replace('"kind":', '"kind":"other","kind":', 1),
                        lean.replace('Source-audit materializations:',
                                     'Source-audit materializations: {}\nSource-audit materializations:', 1)):
            self.reject(lean=changed)

    def test_no_constexpr_or_other_profile_composition(self):
        _, lean, _ = fixture()
        for field in ('params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes'):
            self.reject(lean=lean.replace('Source-audit target:',
                                          'Source-audit ' + field + ': {}\nSource-audit target:', 1))

    def test_identity_position_carrier_and_proof_cannot_drift(self):
        _, lean, _ = fixture()
        identity = '  BLOCK_N_arg = $(' + material.ADAPTER + ' BLOCK_N hBlock)\n'
        self.assertIn(identity, lean)
        for changed in (lean.replace(identity, ''), lean.replace(identity, identity + identity),
                        lean.replace('i32ArangeSize BLOCK_N hBlock', 'i32ArangeSize (BLOCK_N + 1) hBlock'),
                        lean.replace('i32ArangeSize BLOCK_N hBlock', 'i32ArangeSize BLOCK_N otherProof'),
                        lean.replace('Random.ArangeSizeDomain BLOCK_N', 'Random.ArangeSizeDomain N'),
                        lean.replace('SEEDS : Region .int', 'SEEDS : Region .nat'),
                        lean.replace('$((N : Nat))', '$((N : Int))'),
                        lean.replace('$(eps)', '$(eps + 1)'),
                        lean.replace(identity, '').replace('  row = tl.program_id(0)',
                                                            identity + '  row = tl.program_id(0)')):
            self.reject(lean=changed)

    def test_full_computation_tree_rejects_control_update_and_numeric_changes(self):
        _, lean, _ = fixture()
        mutations = [
            ('if HAS_X1 {', 'if HAS_W1 {'),
            ('tl.if not IS_RMS_NORM {', 'tl.if IS_RMS_NORM {'),
            ('tl.if HAS_BIAS {', 'tl.if HAS_B1 {'),
            ('x += residual', 'x = residual'),
            ('x1 *= rowscale', 'x1 += rowscale'),
            ('x += x1', 'x -= x1'),
            ('x - mean', 'x + mean'),
            ('1 / tl.sqrt(var + $(eps))', '1 / tl.sqrt(var)'),
            ('xbar * xbar', 'xbar + xbar'),
            ('tl.sum(x, axis=0)', 'tl.sum(x, axis=1)'),
            ('x / (1.0 - $(dropout_p))', 'x'),
            ('> $(dropout_p)', '>= $(dropout_p)'),
            ('other=0.0', 'other=1.0'),
            ('mask=mask)', 'mask=cols < 1)'),
            ('tl.store(Y1 + cols', 'tl.store(Y + cols'),
            ('  x += residual\n', ''),
            ('  mask = cols <', '  extra = 0\n  mask = cols <'),
        ]
        for before, after in mutations:
            with self.subTest(after=after):
                self.assertIn(before, lean)
                self.reject(lean=lean.replace(before, after, 1))

    def test_both_nested_signed_seed_loads_uint32_casts_rounds_and_masks_are_exact(self):
        _, lean, profile = fixture()
        for before, after in [
            ('tl.load(SEEDS + row).to(tl.uint32)', 'tl.load(SEEDS + row)'),
            ('tl.load(SEEDS + $((M : Nat)) + row)', 'tl.load(SEEDS + row)'),
            ('.to(tl.uint32)', '.to(tl.float32)'),
            ('n_rounds=7', 'n_rounds=10'),
            ('cols, n_rounds=7', 'cols + 1, n_rounds=7'),
            ('SEEDS = tl.int64', 'SEEDS = tl.uint32'),
            ('DROPOUT_MASK = tl.int1', 'DROPOUT_MASK = tl.int32'),
            ('STORE_DROPOUT_MASK {', 'HAS_DROPOUT {'),
        ]:
            with self.subTest(after=after):
                self.assertIn(before, lean)
                self.reject(lean=lean.replace(before, after, 1))
        bad = copy.deepcopy(profile)
        bad['regions']['SEEDS'] = 'tl.uint32'
        self.reject(lean=lean.replace('SEEDS = tl.int64', 'SEEDS = tl.uint32'), profile=bad)

    def test_implicit_store_normalization_cannot_hide_a_wrong_cast(self):
        _, lean, _ = fixture(output='tl.float16', residualOut='tl.bfloat16')
        for before, after in [
            ('(y).to($output:tritonDType)', '(y).to(tl.float32)'),
            ('(x).to($residualOut:tritonDType)', '(x).to(tl.float16)'),
            ('(mean).to(tl.float32)', '(mean).to(tl.float64)'),
            ('(rstd).to(tl.float32)', '(rstd + 1).to(tl.float32)'),
        ]:
            self.reject(lean=lean.replace(before, after, 1))

    def test_branch_grammar_cannot_hide_unsupported_dead_code_or_forward_reads(self):
        guarded._validate_tree(ast.parse(
            'if HAS_BIAS:\n    b = 1\ny = b if HAS_BIAS else 0').body, guarded.PARAMETERS)
        for program in (
            'if HAS_BIAS:\n    b = 1\ny = b',
            'if HAS_BIAS:\n    if not HAS_BIAS:\n        x = bad()',
            'x = 1 if HAS_BIAS else bad() if HAS_W1 else 0',
            'x = tl.where(HAS_BIAS, missing, 0)',
            'if False:\n    x = 0',
            'for row in range(4):\n    x = 0',
            'def hidden():\n    return 0',
            'x = (lambda: 0)()',
        ):
            with self.subTest(program=program), self.assertRaises(ValueError):
                guarded._validate_tree(ast.parse(program).body, guarded.PARAMETERS)

    def test_adapter_cannot_be_changed_and_rehashed_in_metadata(self):
        _, _, profile = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'RandomProfile.lean'
            path.write_text(ADAPTER_FIXTURE.read_text().replace(
                '(_h : ArangeSizeDomain n) : Nat := n', '(_h : ArangeSizeDomain n) : Nat := n + 1'))
            with patch.object(material, '_adapter_path', return_value=path):
                self.reject()
                bad = copy.deepcopy(profile)
                bad['binding']['adapter_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.reject(profile=bad)


class ProductionGuardedMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.temporary.name) / 'repo'
        (cls.repo / 'bench').mkdir(parents=True)
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
        with tempfile.TemporaryDirectory(dir=self.temporary.name) as temporary:
            port = Path(temporary) / 'layer_norm_fwd'
            port.mkdir()
            (port / 'LayerNormFwd.lean').write_text(lean)
            (port / 'layer_norm_fwd.py').write_text(python)
            return subprocess.run(['bash', '-c', self.command, 'guarded-source-only-harness', temporary],
                                  cwd=self.repo, capture_output=True, text=True, timeout=120)

    def test_full_symbolic_source_passes_all_production_source_gates(self):
        _, lean, _ = fixture()
        result = self.production(lean)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sum(line.startswith('ok ') for line in result.stdout.splitlines()), 24)
        self.assertIn('1 extractable source views, 0 unresolved source views', result.stdout)

    def test_absent_profile_does_not_silently_erase_materialization(self):
        _, lean, _ = fixture()
        lean = re.sub(r'^Source-audit materializations: .*\n', '', lean, flags=re.M)
        result = self.production(lean)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL tl.arange call scan', result.stdout)
        self.assertIn('FAIL statement lhs sequence scan', result.stdout)

    def test_translation_marker_cannot_bypass_full_ast_mismatch(self):
        _, lean, _ = fixture()
        lean = lean.replace('/-!\n', '/-!\nTranslation-surface blocker: adversarial diagnostic\n', 1)
        lean = lean.replace('x += residual', 'x = residual')
        result = self.production(lean)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL source-view extraction scan', result.stdout)
        self.assertIn('complete source statement/expression AST differs', result.stdout)


if __name__ == '__main__':
    unittest.main()
