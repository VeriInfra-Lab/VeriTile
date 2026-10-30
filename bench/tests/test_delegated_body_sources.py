"""An indirect Python wrapper cannot hide unchecked JIT functions."""
import json
from pathlib import Path
import tempfile
import unittest
from bench.check_body_corpus import inspect
from bench.jit_correspondence import check_unary_imports

WRAPPER = '''import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location("source", Path(__file__).with_name("body.py"))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
def launch():
    return _module.first
'''
SOURCE = '''import triton
import triton.language as tl
@triton.jit
def first(X, Y):
    x = tl.load(X)
    tl.store(Y, x)
'''
LEAN = '''def signature : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  scalars := []
  constexpr := []
def body (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    tl.store(Y, x)
  }
'''

class DelegatedBodySourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'wrapper.py').write_text(WRAPPER)
        (self.root / 'body.py').write_text(SOURCE)
        (self.root / 'Kernels.lean').write_text(LEAN)
        (self.root / 'jit-correspondence.json').write_text(json.dumps([{
            'python': 'body.py', 'lean': 'Kernels.lean', 'python_function': 'first',
            'lean_definition': 'body', 'signature_definition': 'signature',
            'expected_signature': {'pointers': [['X', 'fp32'], ['Y', 'fp32']], 'scalars': [], 'constexpr': []}}]))

    def row(self):
        return inspect(self.root, 'sample', 'wrapper.py', 'Kernels.lean', 'task')

    def test_complete_delegation_pins_wrapper_and_source(self):
        row = self.row()
        self.assertEqual(row['status'], 'selected_body_copy_checked')
        self.assertIn('wrapper.py', row['source_sha256'])
        self.assertIn('body.py', row['source_sha256'])
        self.assertEqual(row['python_jits'][0]['python'], 'body.py')

    def test_every_delegated_jit_must_be_covered(self):
        (self.root / 'body.py').write_text(SOURCE + SOURCE.replace('def first(', 'def second('))
        row = self.row()
        self.assertEqual(row['status'], 'uncovered_jit_functions')
        self.assertEqual(row['uncovered_selected_jits'], [{'python': 'body.py', 'function': 'second'}])

    def test_loader_rebinding_or_mutation_is_rejected(self):
        for mutation in ['_module = object()\n', 'Path = object()\n', '_spec = object()\n',
                         '_spec.loader.exec_module = replacement\n', 'import other as importlib\n', 'from other import *\n']:
            (self.root / 'wrapper.py').write_text(WRAPPER + mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.row()

    def test_unresolved_loader_cannot_use_a_manifest_as_coverage(self):
        for source in [WRAPPER.replace('exec_module(_module)', 'unrecognized(_module)'),
                       WRAPPER.replace('with_name("body.py")', 'with_name(filename)'),
                       WRAPPER.replace('import importlib.util\n', '') + 'import importlib.util\n',
                       WRAPPER.replace('from pathlib import Path', 'from .pathlib import Path')]:
            (self.root / 'wrapper.py').write_text(source)
            self.assertEqual(self.row()['status'], 'no_selected_jit_resolved')

class UnaryImportAliasTests(unittest.TestCase):
    target = {'f': 'triton.language.extra.libdevice.tanh'}

    def test_direct_and_module_attribute_imports(self):
        for source in ['from triton.language.extra.libdevice import tanh as f',
                       'from triton.language.extra import libdevice\nf = libdevice.tanh',
                       'import triton.language.extra.libdevice as device\nf = device.tanh']:
            check_unary_imports(source, self.target)

    def test_computed_mutated_or_rebound_aliases_are_rejected(self):
        prefix = 'from triton.language.extra import libdevice\n'
        for body in ['f = libdevice.cos', 'f = make_tanh()', 'libdevice = other\nf = libdevice.tanh',
                     'libdevice.tanh = other\nf = libdevice.tanh',
                     'f = libdevice.tanh\nf = other', 'f = libdevice.tanh\ndel f',
                     'from .triton.language.extra.libdevice import tanh as f']:
            with self.subTest(body=body), self.assertRaises(ValueError):
                check_unary_imports(prefix + body, self.target)
