"""The proof gate rejects unavailable isolation before snapshot preparation."""
import subprocess
import unittest
from unittest.mock import patch

from scripts import comparator_common


class ComparatorPreflightTests(unittest.TestCase):
    def probe(self, systemd=0, filesystem=0, network=0):
        results = [
            subprocess.CompletedProcess([], systemd, stderr='no user bus' if systemd else ''),
            subprocess.CompletedProcess([], filesystem, stderr='no filesystem sandbox' if filesystem else ''),
            subprocess.CompletedProcess([], network, stderr='no TCP sandbox' if network else ''),
        ]
        return patch('comparator_common.subprocess.run', side_effect=results)

    @patch('comparator_common.shutil.which', side_effect=lambda name: '/tools/' + name)
    def test_user_bus_filesystem_and_tcp_controls_are_required(self, _which):
        for systemd, filesystem, network in [
                (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
            with self.subTest(systemd=systemd, filesystem=filesystem, network=network), \
                    self.probe(systemd, filesystem, network) as run:
                with self.assertRaises(ValueError) as failure:
                    comparator_common.require_tools()
                self.assertEqual(run.call_count, 3)
                self.assertEqual('no user bus' in str(failure.exception), bool(systemd))
                self.assertEqual('no filesystem sandbox' in str(failure.exception), bool(filesystem))
                self.assertEqual('no TCP sandbox' in str(failure.exception), bool(network))
                self.assertIn('--property=RestrictAddressFamilies=~AF_UNIX', run.call_args_list[0].args[0])
                self.assertEqual(run.call_args_list[1].args[0],
                                 ['landrun', '--best-effort', '--rox', '/', '/tools/true'])
                self.assertEqual(run.call_args_list[2].args[0],
                                 ['landrun', '--unrestricted-filesystem', '--rox', '/', '/tools/true'])

    @patch('comparator_common.shutil.which', side_effect=lambda name: '/tools/' + name)
    def test_success_requires_all_three_probes(self, _which):
        with self.probe() as run:
            self.assertEqual(comparator_common.require_tools(), '/tools/comparator')
            self.assertEqual(run.call_count, 3)

    @patch('comparator_common.shutil.which', return_value=None)
    @patch('comparator_common.subprocess.run')
    def test_missing_tool_cannot_reach_a_sandbox_probe(self, run, _which):
        with self.assertRaisesRegex(ValueError, 'Official comparator is required'):
            comparator_common.require_tools()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
