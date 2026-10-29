from pathlib import Path
from unittest.mock import patch
import gzip,json,tempfile,unittest,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import integration as m

class IntegrationStorageTests(unittest.TestCase):
 def test_rebind_is_exact_byte_splice_and_retains_task_rows(self):
  ledger={'schema':1,'deferred_queue':{'path':m.PLAIN,'sha256':'a'*64},'tasks':[{'task':'one','review':{'sha256':'b'*64},'status':'pending','note':'已完成的证据保留'}],'other':['keep',{'x':1}]};raw=m.text_bytes(ledger);out=m.rebind_ledger_bytes(raw,'c'*64);actual=json.loads(out)
  self.assertEqual(actual,m.rebind_ledger(ledger,'c'*64));self.assertEqual(actual['tasks'],ledger['tasks'])
  self.assertEqual(out.replace(m.COMPRESSED.encode(),m.PLAIN.encode()).replace(('c'*64).encode(),('a'*64).encode()),raw)
 def test_rebind_unknown_prior_path_and_unexpected_format_fail(self):
  with self.assertRaises(ValueError):m.rebind_ledger({'deferred_queue':{'path':'elsewhere'},'tasks':[]},'a'*64)
  ledger={'deferred_queue':{'path':m.PLAIN,'sha256':'a'*64},'tasks':[]}
  with self.assertRaisesRegex(ValueError,'formatting'):m.rebind_ledger_bytes(json.dumps(ledger,separators=(',',':')).encode(),'c'*64)
 def test_archive_verifies_bytes_without_deleting_source(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);src=root/'queue.json';src.write_bytes(b'{"payload":"test"}\n'*40);archive=root/'before/queue.gz';r=m.archive_plain(src,archive)
   self.assertTrue(src.exists());self.assertEqual(gzip.decompress(archive.read_bytes()),src.read_bytes());self.assertEqual(r['decompressed_sha256'],m.file_hash(src))
   with self.assertRaises(ValueError):m.archive_plain(src,archive)
 def test_corrupted_archive_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);src=root/'q';src.write_bytes(b'queue');archive=root/'q.gz';m.archive_plain(src,archive);archive.write_bytes(archive.read_bytes()[:-3])
   with self.assertRaises((ValueError,OSError,EOFError)):m.decompressed_hash(archive)
 def test_explicit_full_commit_required(self):
  for bad in ['HEAD','97e11a9','',None,'a'*39]:
   with self.subTest(bad=bad),self.assertRaises((ValueError,TypeError)):m.commit_check(Path('/tmp'),bad)
 def test_wrong_or_dirty_base_is_rejected(self):
  h='a'*40
  with patch.object(m,'git',return_value='b'*40),self.assertRaisesRegex(ValueError,'HEAD differs'):m.commit_check(Path('/tmp'),h)
  with patch.object(m,'git',side_effect=[h,'commit',' M bench/a.py']),self.assertRaisesRegex(ValueError,'dirty'):m.commit_check(Path('/tmp'),h,clean=True)
  with patch.object(m,'git',side_effect=[h,'commit','']):m.commit_check(Path('/tmp'),h,clean=True)
 def test_relative_path_escape_rejected(self):
  for name in ['../x','/tmp/x','a/../x','a\\b']:
   with self.assertRaises(ValueError):m.relative(Path('/tmp/a'),name)
 def test_doc_checkpoint_render_is_explicit(self):
  raw=b'policy @MIGRATION_CHECKPOINT@/before/bench/source.md'
  self.assertEqual(m.render_doc('bench/source.md',raw,'bench/review-checkpoints/new'),b'policy review-checkpoints/new/before/bench/source.md')
if __name__=='__main__':unittest.main()
