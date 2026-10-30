"""The standalone original-scope gate must enforce the single active queue."""
from pathlib import Path
import gzip,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapter_common import hash_file,write_json
import check_original_program_scope as scope
import deferred_numerical_experiments as producer

class NumericalQueueMigrationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.gz=self.root/scope.QUEUE;self.plain=self.root/(producer.OUTPUT+'.json');self.data={'rules':[],'tasks':[]};write_json(self.gz,self.data);self.ref={'path':scope.QUEUE,'sha256':hash_file(self.gz)}
 def test_scope_reads_exact_gzip(self):self.assertEqual(scope.load_active_queue(self.root,self.ref),self.data)
 def test_scope_rejects_both_even_when_bytes_identical(self):
  self.plain.write_bytes(gzip.decompress(self.gz.read_bytes()))
  with self.assertRaisesRegex(ValueError,'Competing plain'):scope.load_active_queue(self.root,self.ref)
 def test_scope_rejects_plain_only(self):
  self.plain.write_bytes(gzip.decompress(self.gz.read_bytes()));self.gz.unlink()
  with self.assertRaisesRegex(ValueError,'Competing plain'):scope.load_active_queue(self.root,self.ref)
 def test_dangling_plain_name_is_not_silently_reused(self):
  self.plain.symlink_to(self.root/'missing')
  with self.assertRaisesRegex(ValueError,'Competing plain'):scope.load_active_queue(self.root,self.ref)
  for check in [False,True]:
   with self.assertRaisesRegex(ValueError,'Competing plain'):producer.write_outputs(self.root,self.data,check=check)
 def test_old_ledger_path_or_stale_hash_rejected(self):
  for bad in [{'path':producer.OUTPUT+'.json','sha256':self.ref['sha256']},{'path':scope.QUEUE,'sha256':'0'*64}]:
   with self.assertRaises(ValueError):scope.load_active_queue(self.root,bad)
 def test_invalid_gzip_rejected_even_after_repin(self):
  original=self.gz.read_bytes();crc=bytearray(original);crc[-8]^=1
  for raw in [original[:-4],bytes(crc),original+b'not-gzip']:
   self.gz.write_bytes(raw)
   with self.assertRaises((ValueError,OSError,EOFError)):scope.load_active_queue(self.root,{'path':scope.QUEUE,'sha256':hash_file(self.gz)})
 def test_duplicate_json_key_rejected_after_repin(self):
  self.gz.write_bytes(gzip.compress(b'{"rules":[],"rules":[1]}',mtime=0))
  with self.assertRaisesRegex(ValueError,'duplicate JSON'):scope.load_active_queue(self.root,{'path':scope.QUEUE,'sha256':hash_file(self.gz)})
if __name__=='__main__':unittest.main()
