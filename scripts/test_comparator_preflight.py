"""The proof gate rejects unavailable isolation before snapshot preparation."""
import subprocess
import unittest
from unittest.mock import patch

from scripts import comparator_common


class ComparatorPreflightTests(unittest.TestCase):
    def probe(self, systemd=0, landlock=0):
        results = [
            subprocess.CompletedProcess([], systemd, stderr='no user bus' if systemd else ''),
            subprocess.CompletedProcess([], landlock, stderr='no Landlock support' if landlock else ''),
        ]
        return patch('comparator_common.subprocess.run', side_effect=results)

    @patch('comparator_common.shutil.which', side_effect=lambda name: '/tools/' + name)
    def test_both_sandboxes_are_required(self, _which):
        for systemd, landlock in [(1, 0), (0, 1), (1, 1)]:
            with self.subTest(systemd=systemd, landlock=landlock), self.probe(systemd, landlock) as run:
                with self.assertRaises(ValueError) as failure:
                    comparator_common.require_tools()
                self.assertEqual(run.call_count, 2)
                self.assertEqual('no user bus' in str(failure.exception), bool(systemd))
                self.assertEqual('no Landlock support' in str(failure.exception), bool(landlock))
                self.assertIn('--property=RestrictAddressFamilies=~AF_UNIX', run.call_args_list[0].args[0])
                self.assertEqual(run.call_args_list[1].args[0],
                                 ['landrun', '--rox', '/', '/tools/true'])

    @patch('comparator_common.shutil.which', side_effect=lambda name: '/tools/' + name)
    def test_success_requires_both_real_probes(self, _which):
        with self.probe() as run:
            self.assertEqual(comparator_common.require_tools(), '/tools/comparator')
            self.assertEqual(run.call_count, 2)

    @patch('comparator_common.shutil.which', return_value=None)
    @patch('comparator_common.subprocess.run')
    def test_missing_tool_cannot_reach_a_sandbox_probe(self, run, _which):
        with self.assertRaisesRegex(ValueError, 'Official comparator is required'):
            comparator_common.require_tools()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
