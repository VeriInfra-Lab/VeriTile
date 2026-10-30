"""Spec sheets follow reviewed readers and preserve distinct source revisions."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import spec_sheet


class SpecificationSheets(unittest.TestCase):
    def test_selected_readers_replace_legacy_filename_inference(self):
        rows = [{'file': 'bench/repairs/task/KernelOnly/Correct.lean'},
                {'file': 'bench/repairs/task/KernelOnly/Correct.lean'},
                {'file': 'bench/repairs/task/FPEquivBound.lean'}]
        with patch('bench.final_inventory.coverage', return_value={'rows': rows}):
            self.assertEqual(spec_sheet.selected_files(), sorted({
                str(Path(spec_sheet.ROOT) / row['file']) for row in rows}))
        with patch('bench.final_inventory.coverage', side_effect=ValueError('reviewed source changed')):
            with self.assertRaisesRegex(ValueError, 'reviewed source changed'):
                spec_sheet.selected_files()

    def test_distinct_nested_revisions_cannot_overwrite_each_other(self):
        names = ['bench/repairs/a/KernelOnly/Correct.lean',
                 'bench/repairs/b/KernelOnly/Correct.lean',
                 'bench/tritonbench_g/a/Correct.lean',
                 'bench/repairs/a/Correct.lean']
        spec_sheet.check_output_names(names)
        self.assertEqual(len({spec_sheet.sheet_name(p) for p in names}), len(names))
        with self.assertRaisesRegex(ValueError, 'collision'):
            spec_sheet.check_output_names(['port/Correct.lean', 'Port/correct.lean'])

    def test_bound_entry_preserves_public_headline_and_shared_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'Kernels.lean').write_text('def kernel := 1\n')
            entry = root / 'FPEquivBound.lean'
            entry.write_text('specification bound (h : True) : kernel = kernel := by rfl\n')
            sheet, stats = spec_sheet.make_sheet(str(entry), {})
            self.assertEqual(stats['headline'], 1)
            self.assertIn('bound', sheet)
            self.assertIn('def kernel := 1', sheet)


if __name__ == '__main__':
    unittest.main()
