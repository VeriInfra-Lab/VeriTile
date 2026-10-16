from pathlib import Path
import unittest
from unittest_summary import observed_unittest_count

SUCCESS='----------------------------------------------------------------------\nRan 3 tests in 0.123s\n\nOK\n'

class SummaryTests(unittest.TestCase):
    def test_success_with_buffered_stdout(self):
        self.assertEqual(3, observed_unittest_count(SUCCESS+'{"inventory_tasks": 184}\n',0))
    def test_ordinary_success(self):
        self.assertEqual(3,observed_unittest_count(SUCCESS,0))
    def test_nonzero_or_unobserved_exit_is_rejected(self):
        for code in [1,127,None,False]:
            with self.subTest(code=code),self.assertRaises(ValueError):observed_unittest_count(SUCCESS,code)
    def test_failed_missing_duplicate_or_nonpositive_summary_is_rejected(self):
        for data in [SUCCESS.replace('OK','FAILED (failures=1)'),SUCCESS+SUCCESS,
                     SUCCESS.replace('Ran 3','Ran 0'),SUCCESS.replace('OK',''),
                     SUCCESS+'FAILED (failures=1)\n',SUCCESS.replace('OK','OK (skipped=1)'),
                     'Ran 3 tests in 0.1s\n\nintervening data\nOK\n']:
            with self.subTest(data=data),self.assertRaises(ValueError):observed_unittest_count(data,0)
    def test_actual_completed_logs(self):
        base=Path('/tmp/veritile-standardization-20261015/next-three-source-integration-run-001/005-verify')
        import json
        commands={r['name']:r for r in json.loads((base/'run.json').read_text())['commands']}
        for name in ['bench-tests','shard-tests']:
            row=commands[name];self.assertEqual(row['state'],'completed')
            count=observed_unittest_count((base/row['log']).read_text(),row['exit_code'])
            self.assertGreater(count,0)
            if name=='bench-tests':self.assertEqual(count,283)

if __name__=='__main__': unittest.main(verbosity=2)
