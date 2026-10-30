"""Split ports must not lose sources, FP entries or nested proof audit targets."""
from pathlib import Path
import tempfile
import unittest

from bench.port_layout import (
    source_file, source_files, proof_files, specification_files,
    source_import_files, source_text,
)
from bench.audit_source import lean_first_triton_body


class PortLayoutTests(unittest.TestCase):
    def test_supplemental_fp_entry_preserves_source_and_discovers_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port = root / 'legacy'
            (port / 'Proofs').mkdir(parents=True)
            for name in ('Legacy.lean', 'FPCorrect.lean', 'Proofs/Execution.lean'):
                (port / name).touch()
            self.assertEqual(source_file(port), port / 'Legacy.lean')
            self.assertEqual(set(specification_files(root)), {
                port / 'Legacy.lean', port / 'FPCorrect.lean'})
            self.assertEqual(len(proof_files(port)), 3)
            (port / 'Other.lean').touch()
            with self.assertRaisesRegex(ValueError, 'Ambiguous legacy'):
                source_file(port)
            (port / 'Other.lean').unlink()
            (port / 'Legacy.lean').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing source'):
                source_file(port)

    def test_split_supplement_does_not_replace_required_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port = root / 'split'
            port.mkdir()
            for name in ('Kernels.lean', 'Correct.lean', 'FPEquiv.lean', 'FPCorrect.lean'):
                (port / name).touch()
            self.assertEqual(source_file(port), port / 'Kernels.lean')
            self.assertEqual(len(specification_files(root)), 3)
            (port / 'FPEquiv.lean').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing split-port entry'):
                source_file(port)

    def test_source_discovery_uses_only_the_local_import_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp) / 'fixture'
            proofs = port / 'Proofs'
            proofs.mkdir(parents=True)
            entry = port / 'Kernels.lean'
            profile, template, decoy = [proofs / name for name in
                                        ('Profile.lean', 'Template.lean', 'Legacy.lean')]
            entry.write_text('import VeriTile.Triton\n'
                             'import bench.tritonbench_g.fixture.Proofs.Profile\n')
            profile.write_text('import bench.tritonbench_g.fixture.Proofs.Template\n')
            template.write_text('def actual := triton { tl.store(Y, x, mask=active) }\n')
            decoy.write_text('def decoy := triton { tl.store(Y, wrong) }\n')
            self.assertEqual(source_import_files(entry), [entry, profile, template])
            self.assertIn('mask=active', lean_first_triton_body(source_text(entry)))
            self.assertNotIn('wrong', source_text(entry))
            template.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing local source import'):
                source_text(entry)

    def test_source_discovery_rejects_cycles_and_ambiguous_imported_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp) / 'fixture'
            (port / 'Proofs').mkdir(parents=True)
            entry = port / 'Kernels.lean'
            a, b = port / 'Proofs/A.lean', port / 'Proofs/B.lean'
            entry.write_text('import bench.tritonbench_g.fixture.Proofs.A\n')
            a.write_text('import bench.tritonbench_g.fixture.Kernels\n')
            with self.assertRaisesRegex(ValueError, 'cycle'):
                source_text(entry)
            entry.write_text('import bench.tritonbench_g.fixture.Proofs.A\n'
                             'import bench.tritonbench_g.fixture.Proofs.B\n')
            a.write_text('def a := triton { x = tl.load(X) }\n')
            b.write_text('def b := triton { x = tl.load(Y) }\n')
            with self.assertRaisesRegex(ValueError, 'Ambiguous imported'):
                source_text(entry)
            entry.write_text('/-!\nSource-audit Lean: b\n-/\n' + entry.read_text())
            self.assertEqual(lean_first_triton_body(source_text(entry)), ' x = tl.load(Y) ')

    def test_legacy_before_typed_macro_requires_an_explicit_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = Path(tmp) / 'Kernels.lean'
            entry.write_text('''def legacy := triton { x = tl.load(X) }
macro_rules
  | `(typed% $t:tritonDType) => `(triton { x = tl.load(X).to($t) })
''')
            with self.assertRaisesRegex(ValueError, 'Ambiguous literal and macro'):
                source_text(entry)
            entry.write_text('/-!\nSource-audit Lean: typed%\n-/\n' + entry.read_text())
            self.assertIn('.to($t)', lean_first_triton_body(source_text(entry)))

    def test_mixed_corpus_selects_both_contracts_and_nested_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy, split = root / 'legacy', root / 'split'
            legacy.mkdir()
            (legacy / 'Legacy.lean').touch()
            (split / 'Proofs').mkdir(parents=True)
            for name in ('Kernels.lean', 'Correct.lean', 'FPEquiv.lean', 'Proofs/Real.lean'):
                (split / name).touch()
            self.assertEqual(source_file(split), split / 'Kernels.lean')
            self.assertEqual(len(source_files(root)), 2)
            self.assertEqual(set(specification_files(root)), {
                legacy / 'Legacy.lean', split / 'Correct.lean', split / 'FPEquiv.lean'})
            self.assertEqual(len(proof_files(split)), 4)
            self.assertIn(split / 'Proofs/Real.lean', proof_files(split))

    def test_missing_or_ambiguous_entry_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(source_file(root))
            (root / 'Kernels.lean').touch()
            with self.assertRaisesRegex(ValueError, 'Missing split-port entry'):
                source_file(root)
            for name in ('Correct.lean', 'FPEquiv.lean', 'Old.lean'):
                (root / name).touch()
            with self.assertRaisesRegex(ValueError, 'Unexpected split-port'):
                source_file(root)
            (root / 'Kernels.lean').unlink()
            with self.assertRaisesRegex(ValueError, 'Ambiguous legacy'):
                source_file(root)


if __name__ == '__main__':
    unittest.main()
