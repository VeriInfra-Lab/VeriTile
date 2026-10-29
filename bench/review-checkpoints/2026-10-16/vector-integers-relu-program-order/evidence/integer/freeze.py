import hashlib, json, re, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
old=Path('/tmp/veritile-standardization-20261015/vector-addition-wip')
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
mods=['Proofs/IntegerSource','Proofs/IntegerValue','Proofs/IntegerProfiles','Proofs/IntegerGrid','IntegerValueCorrect']
paths=['bench/tritonbench_g/vector_addition/'+m+'.lean' for m in mods]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def deps():
    seen={}; queue=paths.copy(); ext=set()
    while queue:
        rel=queue.pop()
        if rel in seen: continue
        p=root/rel
        source_kind='package'
        if not p.is_file(): p=old/rel; source_kind='frozen_vector_package'
        if not p.is_file(): p=repo/rel; source_kind='repository'
        if not p.is_file(): ext.add(rel); continue
        seen[rel]={'source_kind':source_kind,'sha256':sha(p)}
        for line in p.read_text().splitlines():
            if line.startswith('import '):
                for imp in line[7:].split():
                    if imp.startswith('--'): break
                    queue.append(imp.replace('.','/')+'.lean')
    anchors=['bench/tritonbench_g/vector_addition/vector_addition.py','lakefile.lean','lake-manifest.json','lean-toolchain']
    for rel in anchors:
        p=repo/rel
        if p.is_file(): seen[rel]={'source_kind':'repository','sha256':sha(p)}
    return {'files':dict(sorted(seen.items())),'external_imports':sorted(ext),
      'frozen_vector_receipt_sha256':sha(old/'evidence/receipt.json'),
      'frozen_vector_scope_sha256':sha(old/'evidence/scope-review.json')}
mode=sys.argv[1]
if mode=='before':
    (root/'evidence/final-dependencies-before.json').write_text(json.dumps(deps(),indent=2)+'\n')
    print(json.dumps({'modules':paths,'dependency_count':len(deps()['files'])}))
elif mode=='after':
    before=json.loads((root/'evidence/final-dependencies-before.json').read_text()); after=deps()
    (root/'evidence/final-dependencies-after.json').write_text(json.dumps(after,indent=2)+'\n')
    assert before==after, 'dependencies changed during final verification'
    results=json.loads((root/'evidence/verification-results.json').read_text())
    assert [r['module'] for r in results]==paths
    for r in results:
        assert r['exit_code']==0 and r['source_sha256_before']==r['source_sha256_after']==sha(root/r['module'])
        assert r['log_sha256']==sha(root/r['log'])
        assert not re.search(r'\b(error|sorryAx|DISALLOWED)\b', (root/r['log']).read_text())
    print(json.dumps({'modules':len(results),'all_exit_zero':True,'dependency_count':len(after['files']),'dependency_stable':True,'axiom_checks':sum((root/r['log']).read_text().count('axiom footprint') for r in results),'noncircularity_checks':sum((root/r['log']).read_text().count('non-circular ✓') for r in results)}))
