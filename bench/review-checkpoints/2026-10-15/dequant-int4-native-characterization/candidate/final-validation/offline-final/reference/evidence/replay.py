"""Fresh private direct-Lean replay. Does not modify W, R or the frozen candidate."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_util import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 out=args.out.resolve()
 if out.exists():raise SystemExit('refusing existing output directory')
 subprocess.run([sys.executable,'-B',str(HERE/'verify.py')],check=True)
 receipt=json.loads((HERE/'receipt.json').read_text())
 records=json.loads((HERE/'last-build.json').read_text())
 out.mkdir(parents=True)
 for row in records:
  rel=Path(row['source']).relative_to(HERE);dst=out/rel
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/rel,dst)
 shutil.copyfile(HERE/'build.py',out/'build.py')
 roots=[Path(x) for x in receipt['ordered_roots'][1:]]
 # Lean resolves a top-level namespace against its first object directory.
 # Mirror the explicitly reused bench objects; all new targets must be absent.
 target=out/'.lake/build/lib/lean/bench'
 for root in roots:
  base=root/'bench'
  if not base.exists():continue
  for src in base.rglob('*'):
   if src.is_file():
    dst=target/src.relative_to(base)
    if dst.exists():continue
    dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src)
 for row in records:
  dst=out/'.lake/build/lib/lean'/(row['module'].replace('.','/')+'.olean')
  if dst.exists():raise RuntimeError('new module collides with baseline '+row['module'])
 command=[sys.executable,'-B',str(out/'build.py')]+[r['module'] for r in records]
 log=out/'replay.log'
 with log.open('w') as f: code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
 result={'candidate_receipt':str(HERE/'receipt.json'),'candidate_receipt_sha256':sha(HERE/'receipt.json'),
  'command':command,'exit_code':code,'log':str(log),'log_sha256':sha(log),'new_module_count':len(records)}
 if code==0:
  actual=json.loads((out/'last-build.json').read_text())
  result['actual_records']=str(out/'last-build.json');result['actual_records_sha256']=sha(out/'last-build.json')
  result['source_equal']=all(a['source_after_sha256']==b['source_after_sha256'] for a,b in zip(actual,records))
  result['object_equal']=all(a['object_sha256']==b['object_sha256'] for a,b in zip(actual,records))
  assert result['source_equal']
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));return code
if __name__=='__main__':sys.exit(main())
