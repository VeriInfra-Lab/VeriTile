"""Full original profile product and adversarial source-computation tests.

The fixture is the actual compiled seeded-dropout source snapshot. Production
gates use a clearly identified legacy source-only harness, never fabricated
Correct/FPEquiv entries. Adapter bytes are the reviewed frozen frontend source;
no new native compilation, GPU execution or complete-port acceptance is claimed.
"""
import ast
import copy
import hashlib
import itertools
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
from bench.port_layout import source_text

FIXTURES = ROOT / 'scripts/fixtures/source_materializations'
ADAPTER_FIXTURE = FIXTURES / 'RandomProfile-20261014-signed.lean.txt'
STORAGES = sorted(material.STORAGE_DTYPES)
SEEDS = sorted(material.INTEGER_CARRIERS)
PYTHON = (FIXTURES / 'seeded_dropout.py.txt').read_text()
LEAN = (FIXTURES / 'Kernels.lean.txt').read_text()


def fixture(storage='tl.float32', seed='tl.int32', python=PYTHON, lean=LEAN):
    args = {'storage': storage, 'seedDType': seed, 'version': 'original'}
    profile = {
        'source_sha256': hashlib.sha256(python.encode()).hexdigest(),
        'lean': {'source': 'seededDropoutSource%', 'args': args},
        'regions': {'x_ptr': storage, 'output_ptr': storage},
        'bindings': [
            {'python': 'seed', 'local': 'seed_arg', 'kind': 'runtime_integer', 'dtype': seed},
            {'python': 'BLOCK_SIZE', 'local': 'BLOCK_SIZE_arg', 'kind': 'constexpr_identity',
             'adapter': material.ADAPTER, 'adapter_sha256': material.ADAPTER_SHA256,
             'proof': {'function': 'RandomDomain.arange_domain', 'binder': 'hBlock'}},
        ],
    }
    header = '/-!\nSource-audit target: _seeded_dropout\nSource-audit Lean: seededDropoutSource%\n'
    header += 'Source-audit args: ' + ', '.join(k + '=' + v for k, v in args.items()) + '\n'
    header += 'Source-audit materializations: ' + json.dumps(profile, separators=(',', ':')) + '\n-/\n'
    return python, header + lean, profile


def replace_profile(lean, profile):
    return re.sub(r'^Source-audit materializations: .*$',
                  'Source-audit materializations: ' + json.dumps(profile), lean, flags=re.M)


def write_harness(root, lean):
    port = root / 'seeded_dropout'
    port.mkdir(parents=True)
    (port / 'SeededDropout.lean').write_text(lean)
    (port / 'seeded_dropout.py').write_text(PYTHON)
    (port / 'Proofs').mkdir()
    for path in (FIXTURES / 'Proofs').glob('*.txt'):
        shutil.copyfile(path, port / 'Proofs' / path.name.removesuffix('.txt'))
    return port


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.adapter_patch = patch.object(material, '_adapter_path', return_value=ADAPTER_FIXTURE)
        self.adapter_patch.start()
        self.addCleanup(self.adapter_patch.stop)

    def reject(self, lean=None, python=None, profile=None):
        py, text, _ = fixture()
        if python is not None:
            py = python
        if lean is not None:
            text = lean
        if profile is not None:
            text = replace_profile(text, profile)
        with self.assertRaises(ValueError):
            material.python_source_with_materializations(py, text)

    def test_actual_original_sources_and_adapter_are_pinned(self):
        self.assertEqual(hashlib.sha256(PYTHON.encode()).hexdigest(),
                         '2fd4f4e6aff35d5720156b9cf046f2d112173a431d886106a6441382f63ab570')
        self.assertEqual(hashlib.sha256(LEAN.encode()).hexdigest(),
                         'd22dd1a19598ac8576e225a32dec2c70aea618bcffd374e84ed29103bf60dd30')
        self.assertEqual(hashlib.sha256(ADAPTER_FIXTURE.read_bytes()).hexdigest(),
                         material.ADAPTER_SHA256)

    def test_production_frontend_matches_reviewed_adapter_checkpoint(self):
        self.assertEqual(hashlib.sha256((ROOT / material.ADAPTER_PATH).read_bytes()).hexdigest(),
                         material.ADAPTER_SHA256)

    def test_all_24_original_profiles_preserve_parameters_and_every_statement(self):
        for storage, seed in itertools.product(STORAGES, SEEDS):
            with self.subTest(storage=storage, seed=seed):
                py, lean, _ = fixture(storage, seed)
                result = material.python_source_with_materializations(py, lean)
                original = source._explicit_python_audit_kernel(py, lean)
                changed = source._explicit_python_audit_kernel(result, lean)
                self.assertEqual(ast.dump(original.args), ast.dump(changed.args))
                self.assertEqual(len(changed.body), len(original.body) + 2)
                self.assertIn('tl.rand(seed_arg, offsets)', result)
                self.assertIn('random > p', result)
                self.assertIn('tl.where(x_keep, x / (1 - p), 0.0)', result)
                self.assertIn('tl.load(x_ptr + offsets, mask=mask)', result)
                self.assertIn('tl.store(output_ptr + offsets, output, mask=mask)', result)
                self.assertEqual([ast.dump(node) for node in ast.parse(py).body
                                  if getattr(node, 'name', None) != original.name],
                                 [ast.dump(node) for node in ast.parse(result).body
                                  if getattr(node, 'name', None) != original.name])

    def test_absent_metadata_does_not_infer_or_remove_materializations(self):
        py, lean, _ = fixture()
        lean = re.sub(r'^Source-audit materializations: .*\n', '', lean, flags=re.M)
        self.assertEqual(material.python_source_with_materializations(py, lean), py)
        self.assertIn('seed_arg =', source.lean_first_triton_body(source.select_lean_audit_source(lean)))

    def test_complete_source_hash_includes_jit_wrapper_and_host_code(self):
        for py in [PYTHON + '\n', PYTHON.replace('random > p', 'random >= p'),
                   PYTHON.replace('BLOCK_SIZE=1024', 'BLOCK_SIZE=32')]:
            self.reject(python=py)

    def test_profile_schema_duplicate_keys_and_all_selector_arguments_are_strict(self):
        _, lean, profile = fixture()
        for bad in [{}, [], dict(profile, extra=True), dict(profile, bindings=[]),
                    dict(profile, regions={}), dict(profile, source_sha256=False)]:
            self.reject(profile=bad)
        for changed in [lean.replace('"source_sha256":', '"source_sha256":"x","source_sha256":', 1),
                        lean.replace('Source-audit materializations:',
                                     'Source-audit materializations: {}\nSource-audit materializations:', 1),
                        lean.replace('Source-audit Lean: seededDropoutSource%', 'Source-audit Lean: missing%'),
                        lean.replace('version=original', 'version=indexCommuted', 1),
                        lean.replace(', version=original', '', 1),
                        lean.replace('seedDType=tl.int32', 'seedDType=tl.uint32', 1)]:
            self.reject(lean=changed)

    def test_no_unreviewed_profile_composition(self):
        _, lean, _ = fixture()
        for field in ['params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes']:
            self.reject(lean=lean.replace('Source-audit target:',
                                          'Source-audit ' + field + ': {}\nSource-audit target:', 1))

    def test_direct_seed_and_adapter_identity_are_required(self):
        _, lean, _ = fixture()
        replacements = [
            ('seed_arg = $((seed : $seedTy))', 'seed_arg = $((seed + 1 : $seedTy))'),
            ('seed_arg = $((seed : $seedTy))', 'seed_arg = $((0 : $seedTy))'),
            ('seed_arg = $((seed : $seedTy))', 'seed_arg = seed'),
            ('i32ArangeSize BLOCK_SIZE (', 'i32ArangeSize (BLOCK_SIZE * 2) ('),
            ('i32ArangeSize BLOCK_SIZE (', 'i32ArangeSize 32 ('),
            (material.ADAPTER, 'Random.i32ArangeSize'),
            ('RandomDomain.arange_domain hBlock', 'RandomDomain.arange_domain otherProof'),
        ]
        for before, after in replacements:
            with self.subTest(after=after):
                self.assertIn(before, lean)
                self.reject(lean=lean.replace(before, after))

    def test_source_integer_metadata_is_exact_unique_and_not_computation(self):
        _, lean, _ = fixture()
        line = 'tl.source_integer seed_arg = $seedDType'
        for replacement in ['', 'tl.source_integer seed_arg = tl.uint64',
                            'tl.source_integer other = $seedDType', line + '\n  ' + line,
                            'tl.source_integer seed_arg = $seedDType\n  seed_arg = 0']:
            self.reject(lean=lean.replace(line, replacement))

    def test_full_rhs_rejects_seed_shape_rand_threshold_and_eager_division_changes(self):
        _, lean, _ = fixture()
        for before, after in [
            ('tl.rand(seed_arg, offsets)', 'tl.rand(0, offsets)'),
            ('tl.rand(seed_arg, offsets)', 'tl.rand(seed_arg, offsets + 1)'),
            ('pid * BLOCK_SIZE_arg)', 'pid + BLOCK_SIZE_arg)'),
            ('block_start + tl.arange', 'block_start - tl.arange'),
            ('random > $(p)', 'random >= $(p)'),
            ('random > $(p)', 'random > 0.5'),
            ('x / (1 - $(p))', 'x'),
            ('x / (1 - $(p))', 'x / (1 + $(p))'),
            ('tl.where(x_keep, x / (1 - $(p)), 0.0)', 'x / (1 - $(p)) if x_keep else 0.0'),
            ('tl.where(x_keep, x / (1 - $(p)), 0.0)', 'tl.where(x_keep, x, 0.0) / (1 - $(p))'),
            ('mask=mask)', 'mask=mask, other=0.0)'),
        ]:
            with self.subTest(after=after):
                self.assertIn(before, lean)
                self.reject(lean=lean.replace(before, after))

    def test_no_original_statement_or_runtime_parameter_can_disappear(self):
        _, lean, _ = fixture()
        for changed in [lean.replace('  x_keep = random > $(p)\n', ''),
                        lean.replace('  random = tl.rand(seed_arg, offsets)\n', ''),
                        lean.replace(' (p : ℝ)', ''),
                        lean.replace(' (n_elements : Nat)', ''),
                        lean.replace('  x_keep = random > $(p)\n', '  x_keep = random > $(p)\n  random = 0\n'),
                        lean.replace('  mask = offsets < $(n_elements)', '  extra = 0\n  mask = offsets < $(n_elements)')]:
            self.reject(lean=changed)

    def test_region_profile_and_store_cast_remain_checked(self):
        _, lean, _ = fixture()
        for changed in [lean.replace('x_ptr = $storage', 'x_ptr = tl.float64'),
                        lean.replace('output_ptr = $storage', 'output_ptr = tl.float64'),
                        lean.replace('(output).to($storage:tritonDType)', '(output).to(tl.float64)'),
                        lean.replace('tl.store(output_ptr + offsets', 'tl.store(x_ptr + offsets')]:
            self.reject(lean=changed)

    def test_adapter_provenance_is_fresh_and_cannot_be_self_rehashed(self):
        _, _, profile = fixture()
        bad = copy.deepcopy(profile)
        bad['bindings'][1]['adapter_sha256'] = '0' * 64
        self.reject(profile=bad)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'adapter.lean'
            path.write_text(ADAPTER_FIXTURE.read_text().replace(
                '(_h : ArangeSizeDomain n) : Nat := n', '(_h : ArangeSizeDomain n) : Nat := n + 1'))
            with patch.object(material, '_adapter_path', return_value=path):
                self.reject()
                bad['bindings'][1]['adapter_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.reject(profile=bad)
            with patch.object(material, '_adapter_path', return_value=path.with_name('missing')):
                self.reject()

    def test_original_python_nested_scopes_branches_and_namespace_changes_reject_after_rehash(self):
        for text in [PYTHON.replace('    pid =', '    seed = seed + 1\n    pid =', 1),
                     PYTHON.replace('    pid =', '    if False:\n        seed = 0\n    pid =', 1),
                     PYTHON.replace('    pid =', '    hidden = lambda: seed\n    pid =', 1),
                     PYTHON.replace('    pid =', '    def hidden():\n        return seed\n    pid =', 1),
                     PYTHON + '\ntl = other\n', PYTHON + '\ntl.rand = other\n']:
            py, lean, _ = fixture(python=text)
            self.reject(python=py, lean=lean)

    def test_carrier_quotations_validate_every_arm_and_admission_before_selection(self):
        _, lean, _ = fixture()
        for changed in [lean.replace('=> `(Nat)', '=> `(Real)'),
                        lean.replace('=> `(Nat)', '=> `(Nat + 1)'),
                        lean.replace('=> `(Nat)', '=> arbitraryTerm'),
                        lean.replace('=> `(Nat)', '=> `(tritonExpr| 0)'),
                        lean.replace('let seedTy : TSyntax `term', 'let seedTy : Nat'),
                        lean.replace('let seedTy : TSyntax `term', 'let mut seedTy : TSyntax `term'),
                        lean.replace('let seedTy : TSyntax `term', 'let seedTy : Array (TSyntax `term)'),
                        lean.replace('    let seedTy', '    IO.println "effect"\n    let seedTy', 1),
                        lean.replace('`original || version.getId == `indexCommuted', '`other'),
                        lean.replace('then `(tritonStmt| block_start = BLOCK_SIZE_arg * pid)',
                                     'then unsupportedComputation')]:
            self.reject(lean=changed)

    def test_carrier_quotation_cannot_be_spliced_as_computation_syntax(self):
        _, lean, _ = fixture()
        self.reject(lean=lean.replace('$(p)', '$seedTy:tritonExpr'))
        self.reject(lean=lean.replace('$(p)', '$seedTy*'))




if __name__ == '__main__':
    unittest.main()
