"""Recorded Pow source comparison, not numerical or GPU acceptance.

Synthetic AST tests refresh MOCKED source/evidence hashes to reach structural
validators. Those records are not native evidence. Production tests retain the
reviewed on-disk native record unchanged, including all rejection cases.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source
from bench import source_extern as extern
from bench import source_orders as orders


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def metadata(lean, field, value):
    return re.sub(r'^Source-audit ' + field + r': .*$',
                  'Source-audit ' + field + ': ' + json.dumps(value), lean, flags=re.M)


def strip_metadata(lean, field):
    return re.sub(r'^Source-audit ' + field + r': .*\n', '', lean, flags=re.M)


class SourceExternOrderTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / 'bench/tritonbench_g/pow_scalar_tensor'
        self.python = (directory / 'pow_scalar_tensor.py').read_text()
        self.lean = (directory / 'Kernels.lean').read_text()
        self.profile = json.loads(source.source_audit_metadata(self.lean, 'extern'))
        self.orders = json.loads(source.source_audit_metadata(self.lean, 'orders'))
        self.record = copy.deepcopy(extern._registry()[self.profile['native']['profile']])

    def synthetic(self, python=None, lean=None, record=None):
        """MOCKED evidence only: never write these records to the registry."""
        python = self.python if python is None else python
        lean = self.lean if lean is None else lean
        record = copy.deepcopy(self.record if record is None else record)
        tree = ast.parse(python)
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == orders.KERNEL)
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == extern.HELPER)
        chain = next(n for n in tree.body if isinstance(n, ast.Try))
        record['source_sha256'] = extern._sha(python)
        for key, node in [('kernel_source_sha256', kernel), ('helper_source_sha256', helper),
                          ('import_chain_sha256', chain)]:
            record[key] = extern._sha(extern._excerpt(python, node))
        generated = ('import triton\nfrom triton import language as tl\n' + extern._excerpt(python, chain)
                     + '\n' + extern._excerpt(python, helper) + '\n' + extern._excerpt(python, kernel) + '\n')
        record['generated_module_sha256'] = extern._sha(generated)
        profile = copy.deepcopy(self.profile)
        profile['native']['sha256'] = digest(record)
        order_profile = copy.deepcopy(self.orders)
        order_profile['wrapper']['source_sha256'] = extern._sha(python)
        lean = metadata(metadata(lean, 'extern', profile), 'orders', order_profile)
        return python, lean, {profile['native']['profile']: record}

    def reject_synthetic(self, python=None, lean=None):
        py, view, registry = self.synthetic(python, lean)
        with patch.object(extern, '_registry', return_value=registry), self.assertRaises(ValueError):
            extern.python_source_with_extern_orders(py, view)

    def production(self, cases, *, full=False):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        start = script.index('# A source view must be extractable')
        if full:
            end = script.index('# ---- Machine-checkable trust audit')
        else:
            end = script.index("  printf 'FAIL source-view extraction scan", start)
            end = script.index('\nfi\n', end) + len('\nfi\n')
        command = 'failures=0\nPORTS_ROOT="$1"\n' + script[start:end] + '\nexit "$failures"\n'
        with tempfile.TemporaryDirectory() as directory:
            for index, (python, lean) in enumerate(cases):
                port = Path(directory) / ('case_' + str(index))
                port.mkdir()
                (port / 'source.py').write_text(python)
                (port / 'Source.lean').write_text(lean)
            return subprocess.run(['bash', '-c', command, 'extern-order-regression', directory],
                                  cwd=ROOT, text=True, capture_output=True, timeout=60)

    def test_real_profile_preserves_twelve_parameters_and_three_source_casts(self):
        self.assertEqual(orders.source_order_profile(self.python, self.lean), tuple(orders.ORDERS.items()))
        self.assertEqual(source.python_source_parameter_names(self.python, self.lean), list(orders.ALIASES.values()))
        transformed = extern.python_source_with_extern_orders(self.python, self.lean)
        before = source._explicit_python_audit_kernel(self.python, self.lean)
        after = source._explicit_python_audit_kernel(transformed, self.lean)
        self.assertEqual(ast.dump(before.args), ast.dump(after.args))
        self.assertEqual([ast.dump(n) for n in before.decorator_list], [ast.dump(n) for n in after.decorator_list])
        self.assertEqual(transformed.count('tl.extra.cuda.libdevice.pow(val0.to(tl.float32), in0)'), 2)
        self.assertIn('return _pow(x.to(tl.float32), exponent)', transformed)
        body = source.python_audit_kernel_body(self.python, self.lean)
        resolved = source.resolve_source_dtype_attributes(body, source.lean_first_triton_body(self.lean))
        self.assertEqual(source.cast_sequence(resolved), ['float32'] * 3)
        self.assertNotIn('to(tl.float32).to', transformed)
        self.assertNotIn('Source-audit runtime:', self.lean)
        self.assertEqual(self.record['native']['base_conversion_lines'], [])

    def test_no_profile_is_inferred_without_metadata(self):
        lean = strip_metadata(strip_metadata(self.lean, 'orders'), 'extern')
        self.assertEqual(extern.python_source_with_extern_orders(self.python, lean), self.python)
        self.assertEqual(orders.source_order_profile(self.python, lean), ())
        with self.assertRaises(ValueError):
            extern.python_source_with_extern_orders(self.python, strip_metadata(self.lean, 'orders'))

    def test_order_values_schema_and_wrapper_profile_are_exact(self):
        for value in [False, True, 0.0, 1, -1, '0', None, [], {}]:
            profile = copy.deepcopy(self.orders)
            profile['python']['in0_stride_order0'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                extern.python_source_with_extern_orders(self.python, metadata(self.lean, 'orders', profile))
        for key, value in [('name', 'other'), ('launch', 'other'), ('rank', True),
                           ('rank', 1.0), ('rank', 2), ('source_sha256', '0' * 64)]:
            profile = copy.deepcopy(self.orders)
            profile['wrapper'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                extern.python_source_with_extern_orders(self.python, metadata(self.lean, 'orders', profile))
        for field in ['orders', 'extern']:
            for value in [{}, [], None, {'python': {}}, {'extra': 0}]:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    extern.python_source_with_extern_orders(self.python, metadata(self.lean, field, value))
            for bad in [self.lean.replace('Source-audit ' + field + ':', 'Source-audit ' + field + ': {}\nSource-audit ' + field + ':'),
                        self.lean.replace('"python":', '"python":{},"python":', 1)]:
                with self.subTest(field=field, duplicate=True), self.assertRaises(ValueError):
                    extern.python_source_with_extern_orders(self.python, bad)

    def test_exact_lean_selector_arguments_and_all_nine_aliases_are_required(self):
        changes = [('Source-audit target: ' + orders.KERNEL, 'Source-audit target: other'),
                   ('Source-audit Lean: powScalarFloating%', 'Source-audit Lean: other%'),
                   ('single=0, candidate=0', 'single=1, candidate=0'),
                   ('output=tl.float32,', 'output=tl.float16,'),
                   (', candidate=0', ''), (', num_tasks=tasks', ''),
                   ('val0=base', 'val0=tasks'), ('s0=N', 's0=unknown')]
        for before, after in changes:
            with self.subTest(before=before), self.assertRaises(ValueError):
                extern.python_source_with_extern_orders(self.python, self.lean.replace(before, after, 1))
        for field in ['orders', 'extern']:
            profile = copy.deepcopy(getattr(self, 'orders' if field == 'orders' else 'profile'))
            profile['lean']['args']['single'] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                extern.python_source_with_extern_orders(self.python, metadata(self.lean, field, profile))

    def test_original_signature_and_do_not_specialize_are_preserved(self):
        mutations = [('    num_tasks: int,\n', ''), ('    val0,\n', '    val0: int,\n'),
                     ('in0_stride_order0: tl.constexpr', 'in0_stride_order0: int'),
                     ('in0_stride_order0: tl.constexpr', 'in0_stride_order0'),
                     ('tile_size0: tl.constexpr', 'tile_size0: int'),
                     ("@triton.jit(do_not_specialize=['val0'])", '@triton.jit'),
                     ("do_not_specialize=['val0']", 'do_not_specialize=[]')]
        for before, after in mutations:
            with self.subTest(before=before):
                self.reject_synthetic(self.python.replace(before, after, 1))

    def test_wrapper_tuple_index_target_alias_and_argument_mutations_reject(self):
        mutations = [('in0_stride_order = (0,)', 'in0_stride_order = (False,)'),
                     ('in0_stride_order = (0,)', 'in0_stride_order = (0.0,)'),
                     ('in0_stride_order = (0,)', 'in0_stride_order = (1,)'),
                     ('in0_stride_order = (0,)', 'in0_stride_order = (0, 1)'),
                     ('in0_stride_order[0]', 'in0_stride_order[1]'),
                     ('out0_stride_order[0]', 'in0_stride_order[0]'),
                     ('val0, in0, out0,', 'val0, out0, in0,'),
                     ('one_tile_per_cta=one_tile_per_cta,', 'one_tile_per_cta=True,'),
                     (orders.KERNEL + '[grid]', orders.KERNEL + '[(1,)]'),
                     (orders.KERNEL + '[grid]', 'other_kernel[grid]'),
                     ('in0_stride_order = (0,)', 'in0_stride_order = (0,)\n    alias = in0_stride_order'),
                     ('in0_stride_order = (0,)', 'in0_stride_order = (0,)\n    del in0_stride_order'),
                     ('wrapper_rank_1(val0, in0: Union', 'wrapper_rank_1(val0, unexpected, in0: Union')]
        for before, after in mutations:
            with self.subTest(before=before, after=after):
                self.reject_synthetic(self.python.replace(before, after, 1))

    def test_wrapper_order_must_be_unconditional_before_original_guarded_launch(self):
        statement = '    in0_stride_order = (0,)\n'
        delayed = self.python.replace(statement, '').replace('    return out0', statement + '    return out0', 1)
        self.reject_synthetic(delayed)
        conditional = self.python.replace(statement, '    if False:\n    ' + statement, 1)
        self.reject_synthetic(conditional)
        tree = ast.parse(self.python)
        wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == orders.WRAPPER)
        guard = next(n for n in wrapper.body if isinstance(n, ast.With))
        text = extern._excerpt(self.python, guard)
        for header in ['    def launch_later():\n', '    if enabled:\n', '    for _ in items:\n',
                       '    while enabled:\n']:
            changed = self.python.replace(text, header + ''.join('    ' + line for line in text.splitlines(keepends=True)))
            with self.subTest(header=header):
                self.reject_synthetic(changed)
        self.reject_synthetic(self.python.replace('torch.cuda._DeviceGuard(in0.device.index)', 'other_guard()', 1))

    def test_indirect_wrapper_bindings_are_rejected(self):
        changes = ['global in0_stride_order', 'from foreign import in0_stride_order',
                   'import foreign as in0_stride_order',
                   'try:\n        pass\n    except Exception as in0_stride_order:\n        pass',
                   'match other:\n        case in0_stride_order:\n            pass',
                   'def inner(in0_stride_order):\n        return 0']
        for statement in changes:
            with self.subTest(statement=statement):
                self.reject_synthetic(self.python.replace('    in0_stride_order = (0,)',
                    '    ' + statement + '\n    in0_stride_order = (0,)', 1))

    def test_orders_validate_both_original_arms_and_reject_unsupported_uses(self):
        mutations = [('order=(in0_stride_order0,)', 'order=(out0_stride_order0,)'),
                     ('boundary_check=(in0_stride_order0,)', 'boundary_check=(0,)'),
                     ('order=(in0_stride_order0,)', 'order=(in0_stride_order0 + 0,)'),
                     ('boundary_check=(out0_stride_order0,)', 'boundary_check=(out0_stride_order0, 0)')]
        for before, after in mutations:
            for mode in ['first', 'last']:
                if mode == 'first':
                    changed = self.python.replace(before, after, 1)
                else:
                    index = self.python.rindex(before)
                    changed = self.python[:index] + self.python[index:].replace(before, after, 1)
                with self.subTest(before=before, arm=mode):
                    self.reject_synthetic(changed)
        for statement in ['value = in0_stride_order0', 'in0_stride_order0 = 0',
                          'del out0_stride_order0', 'value = f(in0_stride_order0)',
                          'def nested():\n        return in0_stride_order0']:
            with self.subTest(statement=statement):
                self.reject_synthetic(self.python.replace('    pid = tl.program_id(0)',
                    '    ' + statement + '\n    pid = tl.program_id(0)', 1))

    def test_helper_cast_effects_argument_order_and_all_call_sites_are_checked(self):
        for expression in ['_pow(x, exponent)', '_pow(x.to(tl.float16), exponent)',
                           '_pow(x.to(tl.float32).to(tl.float32), exponent)',
                           '_pow(exponent, x.to(tl.float32))', '_pow(x.to(tl.float32), exponent) + 0',
                           'tl.load(x)', '_pow(x.to(tl.float32), exponent=exponent)']:
            with self.subTest(expression=expression):
                self.reject_synthetic(self.python.replace('_pow(x.to(tl.float32), exponent)', expression, 1))
        for extra in ['    tl.store(x, exponent)\n', '    x = replacement\n',
                      '    _pow = replacement\n', '    alias = tl\n']:
            with self.subTest(extra=extra):
                self.reject_synthetic(self.python.replace('    return _pow', extra + '    return _pow', 1))
        for expression in ['pow_func_scalar_tensor(in0, val0)', 'pow_func_scalar_tensor(val0 + 0, in0)',
                           'pow_func_scalar_tensor(val0, in0) + 0', 'pow_func_scalar_tensor(val0.to(tl.float32), in0)',
                           'pow_func_scalar_tensor(val0, exponent=in0)']:
            for count in [1, 2]:
                with self.subTest(expression=expression, sites=count):
                    self.reject_synthetic(self.python.replace('pow_func_scalar_tensor(val0, in0)', expression, count))

    def test_original_import_chain_and_builtin_importerror_are_protected(self):
        mutations = [('except ImportError:', 'except Exception:'),
                     ('except ImportError:', 'except ImportError as error:'),
                     ('from triton.language.math import pow as _pow', 'from triton.language.libdevice import pow as _pow'),
                     ('from triton.language.extra.cuda.libdevice import pow as _pow', 'from triton.language.math import pow as _pow')]
        for before, after in mutations:
            with self.subTest(before=before, after=after):
                self.reject_synthetic(self.python.replace(before, after, 1))
        missing_fallback = ('try:\n    from triton.language.extra.cuda.libdevice import pow as _pow\n'
                            'except ImportError:\n    from triton.language.math import pow as _pow\n')
        self.reject_synthetic(self.python.replace(extern.IMPORT_CHAIN, missing_fallback, 1))
        reordered = extern.IMPORT_CHAIN.replace('language.math', 'language.TEMP').replace(
            'language.libdevice', 'language.math').replace('language.TEMP', 'language.libdevice')
        self.reject_synthetic(self.python.replace(extern.IMPORT_CHAIN, reordered, 1))
        for suffix in ['\nImportError = Exception\n', '\nfrom foreign import ImportError\n',
                       '\nclass ImportError(Exception):\n    pass\n',
                       '\ndef shadow(ImportError):\n    pass\n', '\nalias = ImportError\n',
                       '\ntry:\n    pass\nexcept Exception as ImportError:\n    pass\n']:
            with self.subTest(suffix=suffix):
                self.reject_synthetic(self.python + suffix)

    def test_namespace_aliases_deep_mutations_and_dynamic_access_fail_closed(self):
        suffixes = ['\n_pow = replacement\n', '\nalias = _pow\n', '\ntl = replacement\n',
                    '\nfrom triton import language as alias\n', '\nimport triton.language as alias\n',
                    '\nalias = tl.extra\n', '\nalias = triton.language\n',
                    '\ntl.extra.cuda.libdevice.pow = replacement\n', '\n_pow.__code__ = replacement\n',
                    '\ntriton.language.extra.cuda.libdevice.pow.__code__ = replacement\n',
                    '\nsetattr(tl, "pow", replacement)\n', '\nglobals()["_pow"] = replacement\n',
                    '\n__import__("triton.language").pow = replacement\n',
                    '\nimport importlib\nimportlib.import_module("triton.language").pow = replacement\n',
                    '\nimport sys\nsys.modules["triton.language"].pow = replacement\n',
                    '\n__builtins__["globals"]()["_pow"] = replacement\n',
                    '\ndef host():\n    global _pow\n    _pow = replacement\n',
                    '\n' + orders.WRAPPER + ' = replacement\n',
                    '\nalias = ' + orders.WRAPPER + '\n',
                    '\nmatch value:\n    case {"x": _pow}:\n        pass\n']
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                self.reject_synthetic(self.python + suffix)

    def test_no_runtime_materialization_or_arithmetic_helper_overlap(self):
        for field in ['runtime', 'helpers']:
            lean = self.lean.replace('Source-audit extern:', 'Source-audit ' + field + ': {}\nSource-audit extern:', 1)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'overlap'):
                extern.python_source_with_extern_orders(self.python, lean)
        for statement in ['val0 = val0.to(tl.float32)', 'val0 += 0', 'del val0',
                          'value = [val0 for val0 in items]']:
            with self.subTest(statement=statement):
                self.reject_synthetic(self.python.replace('    pid = tl.program_id(0)',
                    '    ' + statement + '\n    pid = tl.program_id(0)', 1))

    def test_lean_complete_rhs_casts_and_memory_shape_fields_are_compared(self):
        mutations = [('$((base : ℝ)).to(tl.float32), in0', 'in0, $((base : ℝ)).to(tl.float32)'),
                     ('$((base : ℝ)).to(tl.float32)', '$((base : ℝ))'),
                     ('$((base : ℝ)).to(tl.float32)', '$((base : ℝ)).to(tl.float32).to(tl.float32)'),
                     (').to(X.type.element_ty)', ')'),
                     ('(out0).to($output:tritonDType)', '(out0)'),
                     ('(out0).to($output:tritonDType)', '(out0).to(tl.float16)'),
                     ('shape=($(N))', 'shape=($(B))'), ('strides=($(sx))', 'strides=($(sy))'),
                     ('offsets=(offset0)', 'offsets=(0)'), ('block_shape=($(B))', 'block_shape=($(N))'),
                     ('order=(0)', 'order=(1)'), ('boundary_check=([0] : List Nat)', 'boundary_check=([1] : List Nat)'),
                     ('tl.load(in0_bptr,', 'tl.load(in0_bptr, padding_option=1,'),
                     ('tl.store(out0_bptr,', 'tl.store(in0_bptr,'),
                     ('tile_id = pid + j * num_ctas', 'tile_id = pid + num_ctas * j')]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaises(ValueError):
                extern.python_source_with_extern_orders(self.python, self.lean.replace(before, after))

    def test_selected_python_pointer_fields_and_tensor_casts_are_compared(self):
        mutations = [('(s0,), (in0_stride0,)', '(tile_size0,), (in0_stride0,)'),
                     ('(in0_stride0,), (offset0,)', '(out0_stride0,), (offset0,)'),
                     ('(offset0,), (tile_size0,)', '(0,), (tile_size0,)'),
                     (').to(in0_ptr.type.element_ty)', ')'),
                     ('out0.to(out0_bptr.type.element_ty)', 'out0'),
                     ('tl.load(in0_bptr,', 'tl.load(in0_bptr, padding_option="zero",')]
        for before, after in mutations:
            for arm in ['both', 'unselected']:
                changed = self.python.replace(before, after) if arm == 'both' else self.python.replace(before, after, 1)
                with self.subTest(before=before, arm=arm):
                    self.reject_synthetic(changed)

    def test_registry_schema_duplicates_and_read_failures_are_errors(self):
        for raw in ['{}', '[]', '{"schema":true,"profiles":{}}', '{"schema":1,"profiles":[],"extra":0}',
                    '{"schema":1,"schema":1,"profiles":{}}', '{"schema":1,"profiles":{"x":{},"x":{}}}']:
            with self.subTest(raw=raw), patch.object(Path, 'read_text', return_value=raw), self.assertRaises(ValueError):
                extern._registry()
        with patch.object(Path, 'read_text', side_effect=OSError), self.assertRaises(ValueError):
            extern._registry()
        for registry in [{}, {self.profile['native']['profile']: {}}]:
            with self.subTest(registry=registry), patch.object(extern, '_registry', return_value=registry), self.assertRaises(ValueError):
                extern.python_source_with_extern_orders(self.python, self.lean)

    def test_all_native_fingerprints_and_digest_must_match(self):
        for field in ['source_sha256', 'kernel_source_sha256', 'helper_source_sha256',
                      'import_chain_sha256', 'generated_module_sha256', 'core_sha256', 'cuda_libdevice_sha256']:
            record = copy.deepcopy(self.record)
            record[field] = '0' * 64
            for refresh in [False, True]:
                profile = copy.deepcopy(self.profile)
                if refresh:
                    profile['native']['sha256'] = digest(record)
                with self.subTest(field=field, refreshed=refresh), patch.object(extern, '_registry', return_value={profile['native']['profile']: record}), self.assertRaises(ValueError):
                    extern.python_source_with_extern_orders(self.python, metadata(self.lean, 'extern', profile))

    def test_native_configuration_import_outcomes_signature_and_dispatch_are_strict(self):
        changes = [('compiler', 'triton_version', '3.5.1'), ('compiler', 'target', 'cuda sm80 warp32'),
                   ('compiler', 'kernel_launched', True), ('compiler', 'options', {'num_warps': 8, 'num_stages': 3}),
                   ('selected_import', 'branch', 'fallback'), ('runtime_base_profile', 'object', 'int'),
                   ('runtime_base_profile', 'abi', 'fp64'), ('runtime_base_specialization', '7', ['fp32', None]),
                   ('native', 'profile', 'base_fp32_exp_fp32_single_1'),
                   ('native', 'constexprs', {**self.record['native']['constexprs'], 'in0_stride_order0': False}),
                   ('native', 'signature', {**self.record['native']['signature'], 'val0': 'i32'}),
                   ('native', 'signature', {**self.record['native']['signature'], 'in0_ptr': '*i32'}),
                   ('native', 'ttir_signature', self.record['native']['ttir_signature'].replace('%num_tasks: i32', '%num_tasks: i64')),
                   ('native', 'base_conversion_lines', ['arith.sitofp']),
                   ('native', 'extra', 0), ('native', 'ttir_sha256', '0' * 64)]
        records = []
        for section, key, value in changes:
            record = copy.deepcopy(self.record)
            record[section][key] = value
            records.append((section + '.' + key, record))
        for change in ['stale_ttir', 'wrong_symbol', 'wrong_operand', 'extra_extern', 'bad_import', 'bool_import', 'malformed_import']:
            record = copy.deepcopy(self.record)
            if change == 'stale_ttir':
                record['ttir'] += '\n'
            elif change == 'wrong_symbol':
                record['ttir'] = record['ttir'].replace('__nv_powf', '__nv_powif')
            elif change == 'wrong_operand':
                record['ttir'] = record['ttir'].replace('(tensor<32xf32>, tensor<32xf32>)', '(tensor<32xf32>, tensor<32xi32>)')
            elif change == 'extra_extern':
                record['ttir'] += record['native']['extern_lines'][0] + '\n'
            elif change == 'bad_import':
                record['import_evidence'][1]['status'] = 'accepted'
            elif change == 'bool_import':
                record['import_evidence'][0]['same_as_selected'] = 1
            else:
                record['import_evidence'][1] = None
            if change in ['wrong_symbol', 'wrong_operand', 'extra_extern']:
                record['native']['ttir_sha256'] = extern._sha(record['ttir'])
                record['native']['extern_lines'] = [line.strip() for line in record['ttir'].splitlines() if 'tt.extern_elementwise' in line]
            records.append((change, record))
        for name, record in records:
            profile = copy.deepcopy(self.profile)
            profile['native']['sha256'] = digest(record)
            with self.subTest(name=name), patch.object(extern, '_registry', return_value={profile['native']['profile']: record}), self.assertRaises(ValueError):
                extern.python_source_with_extern_orders(self.python, metadata(self.lean, 'extern', profile))

    def test_utf8_spans_comments_strings_other_functions_and_signatures_are_preserved(self):
        extra = '\n@triton.jit\ndef unrelated(in0_stride_order0):\n    """pow_func_scalar_tensor(val0, in0) 中文"""\n    return in0_stride_order0\n'
        python = self.python.replace('    return _pow', '    """中文 helper docs"""\n    # keep val0 and in0_stride_order0\n    return _pow', 1) + extra
        py, lean, registry = self.synthetic(python)
        with patch.object(extern, '_registry', return_value=registry):
            transformed = extern.python_source_with_extern_orders(py, lean)
        self.assertTrue(transformed.endswith(extra))
        self.assertIn('"""中文 helper docs"""\n    # keep val0 and in0_stride_order0', transformed)
        self.assertEqual(transformed[:transformed.index('@triton.jit(do_not_specialize=')], py[:py.index('@triton.jit(do_not_specialize=')])
        text = '文本 = "字"; answer = helper(value)  # helper(value)\n'
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call))
        self.assertEqual(extern._apply_edits(text, [(node, 'value')]), '文本 = "字"; answer = value  # helper(value)\n')
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            extern._apply_edits(text, [(node, 'value'), (node.args[0], '0')])

    def test_real_production_all_twenty_four_source_gates_pass(self):
        result = self.production([(self.python, self.lean)], full=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(re.findall(r'^ok ', result.stdout, re.M)), 24, result.stdout)
        self.assertIn('1 extractable source views, 0 unresolved', result.stdout)

    def test_real_production_mutations_fail_mandatory_gate_with_and_without_blocker(self):
        pairs = [(self.python + '\n', self.lean),
                 (self.python.replace('    num_tasks: int,\n', '', 1), self.lean),
                 (self.python.replace("do_not_specialize=['val0']", 'do_not_specialize=[]', 1), self.lean),
                 (self.python.replace('_pow(x.to(tl.float32), exponent)', '_pow(x, exponent)', 1), self.lean),
                 (self.python.replace('in0_stride_order = (0,)', 'in0_stride_order = (1,)', 1), self.lean),
                 (self.python.replace('pow_func_scalar_tensor(val0, in0)', 'pow_func_scalar_tensor(in0, val0)', 1), self.lean),
                 (self.python.replace('except ImportError:', 'except Exception:', 1), self.lean),
                 (self.python + '\nImportError = Exception\n', self.lean),
                 (self.python + '\nalias = tl\n', self.lean),
                 (self.python, self.lean.replace('shape=($(N))', 'shape=($(B))')),
                 (self.python, self.lean.replace('(out0).to($output:tritonDType)', '(out0)')),
                 (self.python, self.lean.replace('in0_stride_order0\":0', 'in0_stride_order0\":false')),
                 (self.python, self.lean.replace(self.profile['native']['sha256'], '0' * 64)),
                 (self.python, strip_metadata(self.lean, 'orders')),
                 (self.python, self.lean.replace(', num_tasks=tasks', ''))]
        for marker in [False, True]:
            cases = [(py, lean.replace('/-!', '/-!\nTranslation-surface blocker: existing fixture blocker', 1)
                      if marker else lean) for py, lean in pairs]
            result = self.production(cases)
            with self.subTest(blocker=marker):
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('0 extractable source views, ' + str(len(cases)) + ' unresolved', result.stdout)
                self.assertIn('FAIL source-view extraction scan', result.stdout)
                self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
