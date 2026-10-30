"""Release selection must fail closed on omissions, drift and status promotion."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import verified_scope as gate


class VerifiedScope(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scope = self.root / 'bench/verified-scope'
        self.scope.mkdir(parents=True)
        (self.root / 'bench/tritonbench_g').mkdir()
        (self.root / 'bench/tests').mkdir()
        (self.root / 'Main.lean').write_text('import Helper\n')
        (self.root / 'Helper.lean').write_text('theorem valid : True := True.intro\n')
        self.sha = lambda p: hashlib.sha256((self.root / p).read_bytes()).hexdigest()
        (self.scope / 'roots.txt').write_text('Main\n')
        build = ['lean-toolchain', 'lakefile.toml', 'lake-manifest.json']
        for p in build:
            (self.root / p).write_text('fixture\n')
        self.kernel = {'public_entries': ['Main.lean'], 'source': 'main.py', 'kernel_source': 'Main.lean'}
        self.ledger = {'tasks': [{'task': 'fixture', 'kernel_revision': self.kernel}]}
        self.bodies = {'tasks': [{'name': 'fixture', 'status': 'uncovered_jit_functions'}], 'counts': {}, 'examples': []}
        self.data = {'schema': 1, 'root_count': 1,
                     'roots_sha256': self.sha('bench/verified-scope/roots.txt'),
                     'source_sha256': {p: self.sha(p) for p in ['Main.lean', 'Helper.lean']},
                     'build_input_sha256': {p: self.sha(p) for p in build},
                     'tasks': [{'task': 'fixture', 'proof_scope': 'selected',
                                'body_status': 'uncovered_jit_functions', 'entries': ['Main.lean'],
                                'python': 'main.py', 'kernel': 'Main.lean'}],
                     'body_counts': {}, 'examples': [], 'selected_python_tests': [], 'pending_python_tests': {}}
        for name, value in [('bench/tritonbench_g/task_progress.json', self.ledger),
                            ('bench/source-correspondence.json', self.bodies)]:
            (self.root / name).write_text(json.dumps(value))
        self.save()
        self.addCleanup(patch.stopall)
        patch.object(gate, 'ROOT', self.root).start()
        patch.object(gate, 'SCOPE', self.scope).start()

    def save(self):
        (self.scope / 'manifest.json.gz').write_bytes(gzip.compress(json.dumps(self.data).encode()))

    def test_unchanged_scope_preserves_pending_body(self):
        data, roots = gate.read_scope()
        self.assertEqual(['Main'], roots)
        self.assertEqual('uncovered_jit_functions', data['tasks'][0]['body_status'])

    def test_source_change_fails(self):
        (self.root / 'Helper.lean').write_text('axiom invalid : False\n')
        with self.assertRaisesRegex(ValueError, 'Changed selected proof dependency'):
            gate.read_scope()

    def test_omitted_dependency_fails(self):
        del self.data['source_sha256']['Helper.lean']
        self.save()
        with self.assertRaisesRegex(ValueError, 'dependency closure'):
            gate.read_scope()

    def test_omitted_root_or_task_fails(self):
        (self.scope / 'roots.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'roots changed'):
            gate.read_scope()
        (self.scope / 'roots.txt').write_text('Main\n')
        self.data['tasks'] = []
        self.save()
        with self.assertRaisesRegex(ValueError, 'task was added or omitted'):
            gate.read_scope()

    def test_pending_body_cannot_be_promoted(self):
        self.data['tasks'][0]['body_status'] = 'selected_body_copy_checked'
        self.save()
        with self.assertRaisesRegex(ValueError, 'disposition changed'):
            gate.read_scope()

    def test_new_test_or_changed_toolchain_requires_review(self):
        (self.root / 'bench/tests/test_new.py').write_text('')
        with self.assertRaisesRegex(ValueError, 'explicit disposition'):
            gate.read_scope()
        (self.root / 'bench/tests/test_new.py').unlink()
        (self.root / 'lean-toolchain').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'build configuration changed'):
            gate.read_scope()
