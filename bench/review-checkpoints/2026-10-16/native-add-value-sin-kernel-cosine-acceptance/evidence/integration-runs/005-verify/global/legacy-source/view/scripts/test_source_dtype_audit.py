"""Adversarial dtype-provenance checks used by the production source audit."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.audit_source import declared_region_dtypes, resolve_source_dtype_attributes as resolve


class SourceDtypeAuditTests(unittest.TestCase):
    def test_production_memory_gate_resolves_both_pointer_element_spellings(self):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        gates = [body for body in re.findall(r"<<'PY'\n(.*?)\nPY\nthen", script, re.S)
                 if 'tl.{fn} call expression mismatch' in body]
        self.assertEqual(len(gates), 1)
        python = ('@triton.jit\ndef kernel(O, N, B, value):\n'
                  '    p = tl.make_block_ptr(base=O, shape=(N,), strides=(1,), '
                  'offsets=(0,), block_shape=(B,), order=(0,))\n'
                  '    tl.store(p, value.to(p.dtype.element_ty), boundary_check=(0,))\n')
        for dtype in ['tl.float16', 'tl.bfloat16', 'tl.float32', 'tl.float64', 'OUT_DTYPE']:
            wrong = 'tl.float16' if dtype == 'tl.float32' else 'tl.float32'
            for cast, value, boundary, expected in [
                    ('p.dtype.element_ty', 'value', '0', 0),
                    (dtype, 'value', '0', 0),
                    (wrong, 'value', '0', 1),
                    ('p.dtype.element_ty', 'value + 1', '0', 1),
                    ('p.dtype.element_ty', 'value', '1', 1)]:
                with self.subTest(dtype=dtype, cast=cast, value=value, boundary=boundary), \
                        tempfile.TemporaryDirectory() as temp:
                    port = Path(temp) / 'fixture'
                    port.mkdir()
                    (port / 'fixture.py').write_text(python)
                    (port / 'Fixture.lean').write_text(
                        '/-!\nSource-audit target: kernel\n-/\n'
                        'def source (O : RegionName) (N B : Nat) (value : Real) := triton {\n'
                        f'tl.region O = {dtype}\n'
                        'p = tl.make_block_ptr(base=O, shape=($(N)), strides=($(1)), '
                        'offsets=($(0)), block_shape=($(B)), order=(0))\n'
                        f'tl.store(p, ({value}).to({cast}), boundary_check=([{boundary}] : List Nat))\n'
                        '}\n')
                    result = subprocess.run([sys.executable, '-', temp], input=gates[0],
                                            text=True, cwd=ROOT, capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    if expected:
                        self.assertIn('tl.store call expression mismatch', result.stdout)

    def test_declared_pointer_self_advances_retain_only_the_declared_type(self):
        for dtype in ['tl.float16', 'tl.float32', 'tl.int32', '$storage']:
            body = (f'tl.region O = {dtype}\n'
                    'O = O + row * stride +\n  col\n'
                    'O += step\nO = O + offset\nO -= step\n')
            with self.subTest(dtype=dtype):
                self.assertEqual(resolve('v.to(O.dtype.element_ty)', body), f'v.to({dtype})')
        self.assertEqual(resolve('v.to(O.dtype.element_ty)', 'O = O + offset\n'),
                         'v.to(O.dtype.element_ty)')

    def test_self_advances_do_not_hide_rebinding_or_unknown_updates(self):
        body = 'tl.region O = tl.float16, P = tl.float32\nO = O + offset\n'
        for update in ['O = P + offset', 'O = choose(O, P)', 'O *= stride',
                       'O //= stride', 'O = O + offset; O = P + other',
                       'if active { O = P + other }']:
            with self.subTest(update=update):
                self.assertEqual(resolve('v.to(O.dtype.element_ty)', body + update),
                                 'v.to(O.dtype.element_ty)')

    def test_load_arithmetic_casts_and_same_line_rebindings_stay_unknown(self):
        for rhs in ['tl.load(P) + wide', 'tl.load(P) * scale',
                    'tl.load(P).to(tl.float32)', 'tl.load(P); v = other',
                    'tl.load(P) + tl.load(Q)', 'tl.load(P']:
            body = 'tl.region P = tl.float16, Q = tl.float32\nv = ' + rhs + '\n'
            with self.subTest(rhs=rhs):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')
        body = 'tl.region P = tl.float16\nv = tl.load(P, mask=active, other=0.0)\n'
        self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(tl.float16)')
        self.assertEqual(resolve('x.to(v.dtype)', body + 'if active { v = wide }\n'),
                         'x.to(v.dtype)')

    def test_symbolic_tensor_negation_is_not_assumed_dtype_preserving(self):
        body = 'tl.region P = $storage\nv = tl.load(P)\nif backward {\n v = -v\n}\n'
        self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')

    def test_concrete_floating_self_negations_retain_loaded_dtype(self):
        for dtype in ['tl.float16', 'tl.bfloat16', 'tl.float32', 'tl.float64']:
            body = f'tl.region P = {dtype}\nv = tl.load(P)\nif backward {{\n v = -v\n}}\n'
            with self.subTest(dtype=dtype):
                self.assertEqual(resolve('x.to(v.dtype)', body), f'x.to({dtype})')
                self.assertEqual(resolve('x.to(v.dtype)', body + 'v = -v\n'), f'x.to({dtype})')
        for dtype in ['tl.int1', 'tl.int8', 'tl.uint8', 'tl.int32', '$storage', 'UNKNOWN']:
            body = f'tl.region P = {dtype}\nv = tl.load(P)\nif backward {{\n v = -v\n}}\n'
            with self.subTest(dtype=dtype):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')

    def test_self_negation_cannot_hide_other_writes_or_continuations(self):
        for mutation in ['v = -v + wide', 'v = -v\n + wide', 'v = -v\n .to(tl.float64)',
                         'v = -other', 'v = tl.load(P)', 'v += -v',
                         'v = -v; v = wide', 'if active { v = -v }', 'v = (-v).to(tl.float32)']:
            body = 'tl.region P = tl.float16\nv = tl.load(P)\n' + mutation + '\n'
            with self.subTest(mutation=mutation):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')

    def test_production_cast_gate_checks_negation_and_destination_dtypes(self):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        bodies = re.findall(r"<<'PY'\n(.*?)\nPY\nthen", script, re.S)
        gates = [body for body in bodies if '.to(...) cast sequence mismatch' in body]
        self.assertEqual(len(gates), 1)
        python = ('@triton.jit\ndef kernel(P, X, BACKWARD):\n'
                  '    v = tl.load(P)\n    if BACKWARD:\n        v = -v\n'
                  '    q = tl.load(X).to(v.dtype)\n')
        for dtype in ['tl.float16', 'tl.bfloat16', 'tl.float32', 'tl.float64']:
            for update, cast, expected in [('v = -v', dtype, 0),
                                            ('v = -v + wide', dtype, 1),
                                            ('v = -v\n + wide', dtype, 1),
                                            ('v = -v', 'tl.int32', 1)]:
                with self.subTest(dtype=dtype, update=update, cast=cast), tempfile.TemporaryDirectory() as temp:
                    port = Path(temp) / 'fixture'
                    port.mkdir()
                    (port / 'fixture.py').write_text(python)
                    (port / 'Fixture.lean').write_text(
                        '/-!\nSource-audit target: kernel\n-/\n'
                        'def source (P X : RegionName) (BACKWARD : Bool) := triton {\n'
                        f'tl.region P = {dtype}, X = tl.float32\n'
                        'v = tl.load(P)\nif BACKWARD {\n' + update + '\n}\n'
                        f'q = tl.load(X).to({cast})\n' + '}\n')
                    result = subprocess.run([sys.executable, '-', temp], input=gates[0],
                                            text=True, cwd=ROOT, capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    if expected:
                        self.assertIn('.to(...) cast sequence mismatch', result.stdout)

    def test_multiline_load_suffixes_do_not_establish_storage_dtype(self):
        for suffix in ['+ wide', '* scale', '.to(tl.float32)', '[indices]',
                       '(argument)', 'unknown_continuation']:
            body = 'tl.region P = tl.float16\nv = tl.load(P)\n  ' + suffix + '\n'
            with self.subTest(suffix=suffix):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')
                commented = body.replace('\n  ' + suffix, '\n/- continue -/\n  ' + suffix)
                self.assertEqual(resolve('x.to(v.dtype)', commented), 'x.to(v.dtype)')
        for following in ['other = v + 1', 'tl.store(O, v)', 'if active {\n tl.store(O, v)\n}', '}']:
            body = 'tl.region P = tl.float16\nv = tl.load(P)\n' + following + '\n'
            with self.subTest(following=following):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(tl.float16)')

    def test_balanced_multiline_loads_keep_their_declared_storage_dtype(self):
        for dtype in ['tl.float16', 'tl.float32', '$storage']:
            load = ('tl.load(P + offsets,\n'
                    ' mask=(rows < count) &\n  (cols < $(width)),\n'
                    ' other=0.0)')
            for suffix, expected in [('', dtype), (' + wide', 'v.dtype'),
                                      ('\n + wide', 'v.dtype'),
                                      ('.to(tl.float64)', 'v.dtype'),
                                      ('\n .to(tl.float64)', 'v.dtype')]:
                body = f'tl.region P = {dtype}\nv = ' + load + suffix + '\nq = v\n'
                with self.subTest(dtype=dtype, suffix=suffix):
                    self.assertEqual(resolve('x.to(v.dtype)', body), f'x.to({expected})')
            body = f'tl.region P = {dtype}\nv = ' + load[:-1] + '\nq = v\n'
            self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')

    def test_load_pointer_casts_and_postfixes_do_not_keep_original_storage(self):
        for pointer in ['P.to(tl.pointer_type(tl.float32))', 'P.to(tl.float32)',
                        'P.reinterpret(tl.float32)', 'P(indices)', 'P[indices]',
                        'P if active else other']:
            body = f'tl.region P = tl.float16\nv = tl.load({pointer})\nq = v\n'
            with self.subTest(pointer=pointer):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')
        for pointer in ['P', 'P + offset', 'P - offset', 'P + offset.to(tl.int32)']:
            body = (f'tl.region P = tl.float16\nv = tl.load({pointer},\n'
                    ' mask=active, other=padding.to(tl.float32))\nq = v\n')
            with self.subTest(pointer=pointer):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(tl.float16)')

    def test_strings_and_comments_cannot_supply_load_or_pointer_provenance(self):
        for decoy in ['v = tl.load(P)', 'p = tl.make_block_ptr(base=P)']:
            body = ('tl.region P = tl.float16\nv = tl.load(unknown)\np = choose(P, other)\n'
                    'message = "' + decoy + '"\n')
            with self.subTest(decoy=decoy):
                self.assertEqual(resolve('x.to(v.dtype)', body), 'x.to(v.dtype)')
                self.assertEqual(resolve('x.to(p.dtype.element_ty)', body), 'x.to(p.dtype.element_ty)')

    def test_raw_strings_cannot_declare_regions_or_rebind_pointer_dtypes(self):
        for opening, closing in [('r#"', '"#'), ('r##"', '"##'), ('"', '"')]:
            def message(decoy):
                return 'tl.static_print(' + opening + 'message\n' + decoy + '\n' + closing + ')\n'
            with self.subTest(opening=opening):
                fake_region = message('tl.region O = tl.float16')
                self.assertEqual(declared_region_dtypes(fake_region), {})
                self.assertEqual(resolve('v.to(O.dtype.element_ty)', fake_region),
                                 'v.to(O.dtype.element_ty)')
                body = 'tl.region O = tl.float16, P = tl.float32\n' + message('O = P + off')
                self.assertEqual(resolve('v.to(O.dtype.element_ty)', body), 'v.to(tl.float16)')
                self.assertEqual(declared_region_dtypes(body),
                                 {'O': {'tl.float16'}, 'P': {'tl.float32'}})
                body = 'tl.region O = tl.float16\n' + message('p = O + off')
                self.assertEqual(resolve('v.to(p.dtype.element_ty)', body),
                                 'v.to(p.dtype.element_ty)')


if __name__ == '__main__':
    unittest.main()
