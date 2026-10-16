from pathlib import Path
import concurrent.futures,datetime,hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=W/'native-attempts/20261015T214958399467Z'
work=W/'native-attempts'/('fp8-blocks-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
work.mkdir();(work/'source.py').write_bytes((original/'source.py').read_bytes());(work/'probe_native.py').write_bytes((W/'probe_native.py').read_bytes());(work/'probe_fp8_blocks.py').write_bytes(Path(__file__).read_bytes())
jobs=[(k,d,b) for k in ['mul2_kernel','mul2_inplace_kernel'] for d in ['float8_e4m3fn','float8_e5m2'] for b in [1,512,1024]]
def run(job):
 k,d,b=job;name=f'{k}-{d}-b{b}';cmd=[sys.executable,'-B',str(work/'probe_native.py'),'worker',str(work),name,k,d,str(b),'full']
 with (work/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 p=work/(name+'.json');row={'name':name,'command':cmd,'exit_code':r.returncode,'log':name+'.log','log_sha256':sha(work/(name+'.log')),'result':json.loads(p.read_text()),'record_sha256':sha(p)}
 print(name,row['result']['status'],row['result'].get('failure_stage'),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(run,jobs))
assert all(r['exit_code']==0 and r['result']['status']=='rejected' and r['result']['failure_stage']=='backend' for r in records)
receipt={'task':'triton_mul2','kind':'fp8_block_backend_rejection_supplement','original_receipt_sha256':sha(original/'receipt.json'),'records':records,'gpu_launches':0,'artifacts':{str(p.relative_to(work)):sha(p) for p in sorted(work.iterdir()) if p.is_file()}}
p=work/'receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(p),'sha256':sha(p)}))
