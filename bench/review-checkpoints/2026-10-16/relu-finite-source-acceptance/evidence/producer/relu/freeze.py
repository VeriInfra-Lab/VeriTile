"""Freeze the candidate without mutating the repository or earlier packages."""
import hashlib
import json
from pathlib import Path
import re
import sys

root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
dump=lambda p,d:p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
modules=['bench/tritonbench_g/relu_triton_kernel/'+n+'.lean' for n in
  ['Proofs/FiniteIndices','Proofs/FiniteBoundary','Proofs/FiniteHost','Proofs/FiniteOutcomes','FiniteCorrect']]

def after():
    data=read(root/'evidence/dependencies-before.json')
    for rel,row in data['project'].items():
        assert sha(Path(row['source']))==row['sha256'],rel
        assert sha(Path(row['olean']))==row['olean_sha256'],rel
    for rel,row in data['external_objects'].items():assert sha(Path(row['olean']))==row['sha256'],rel
    for p,h in data['input_sha256'].items():assert sha(Path(p))==h,p
    dump(root/'evidence/dependencies-after.json',data)
    return data

def finalize():
    receipt=root/'evidence/receipt.json'
    assert not receipt.exists(),'Never modify a frozen package'
    after()
    history=read(root/'evidence/verification-attempts.json')
    final=read(root/'evidence/verification-results.json')
    assert [r['module'] for r in final]==modules and history[-len(modules):]==final
    for r in history:
        assert sha(root/r['source_snapshot'])==r['snapshot_sha256']==r['source_sha256_before']
        assert sha(root/r['log'])==r['log_sha256']
    for r in final:
        assert r['exit_code']==0
        assert r['source_sha256_before']==r['source_sha256_after']==sha(root/r['module'])
        assert not re.search(r'\berror:|sorryAx|DISALLOWED|warning:',(root/r['log']).read_text())
    # Metadata generator source hashes must describe the final Lean state.
    authority=read(root/'evidence/numerical-authority-delta.json')
    for p in authority['provenance'].values():
        assert p['lean_source_sha256']=={f:sha(root/f) for f in modules}
    artifacts={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
      if p.is_file() and not p.is_symlink() and p.suffix not in ['.olean','.ilean']
      and p!=receipt and str(p.relative_to(root)) not in modules}
    dump(receipt,{'schema_version':1,'frozen':True,'acceptance_increment':0,
      'candidate':'complete finite source-scope proposal, independent review/integration pending',
      'modules':{f:sha(root/f) for f in modules},
      'checkpoint_module_source_paths':{f:f+'.txt' for f in modules},
      'module_objects':{str(Path(f).with_suffix('.olean')):sha((root/f).with_suffix('.olean')) for f in modules},
      'artifacts':artifacts,'verification_attempt_count':len(history),
      'failed_attempts_preserved':sum(r['exit_code']!=0 for r in history),'final_clean_attempts':final,
      'independent_review_status':'pending root substantive review and exact queue/catalog/ledger integration',
      'numerical_conformance_validated':False,'native_backend_compiled':False,'gpu_executed':False})
    print(json.dumps({'receipt':str(receipt),'receipt_sha256':sha(receipt),
      'modules':len(modules),'attempts':len(history),'failed':sum(r['exit_code']!=0 for r in history)}))

if __name__=='__main__':
    if sys.argv[1]=='after':after()
    elif sys.argv[1]=='finalize':finalize()
    else:raise SystemExit('expected after or finalize')
