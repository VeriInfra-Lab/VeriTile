import hashlib, json, re, subprocess, sys, tempfile, shutil
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
prior=[root.parent/n for n in ['vector-addition-mixed-fp32-revision1-wip','vector-addition-dtype-extension-wip','vector-addition-wip']]
mods=['Proofs/MixedSmallSource','Proofs/MixedSmallValue','Proofs/MixedSmallGrid','Proofs/MixedSmallDefined','Proofs/MixedSmallProfiles','MixedSmallValueCorrect']
paths=['bench/tritonbench_g/vector_addition/'+m+'.lean' for m in mods]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def deps():
    seen={}; queue=paths.copy(); ext=set()
    while queue:
        rel=queue.pop()
        if rel in seen: continue
        for base in [root,*prior,repo]:
            p=base/rel
            if p.is_file(): break
        else: ext.add(rel); continue
        row={'source_path':str(p),'sha256':sha(p)}
        if rel not in paths:
            for base in [root,*prior,repo/'.lake/build/lib/lean']:
                o=(base/rel).with_suffix('.olean')
                if o.is_file():
                    row.update(olean_path=str(o.resolve()),olean_sha256=sha(o)); break
        seen[rel]=row
        for line in p.read_text().splitlines():
            if line.startswith('import '):
                for imp in line[7:].split():
                    if imp.startswith('--'): break
                    queue.append(imp.replace('.','/')+'.lean')
    for rel in ['bench/tritonbench_g/vector_addition/vector_addition.py','lakefile.toml','lakefile.lean','lake-manifest.json','lean-toolchain']:
        p=repo/rel
        if p.is_file(): seen[rel]={'source_path':str(p),'sha256':sha(p)}
    return {'files':dict(sorted(seen.items())),'external_imports':sorted(ext),
      'prior_receipts':{str(p/'evidence/receipt.json'):sha(p/'evidence/receipt.json') for p in prior},
      'prior_scopes':{str(p/'evidence/scope-review.json'):sha(p/'evidence/scope-review.json') for p in prior}}
mode=sys.argv[1]
if mode=='before':
    d=deps(); dump(root/'evidence/final-dependencies-before.json',d)
    print(json.dumps({'modules':paths,'dependency_count':len(d['files'])}))
elif mode=='after':
    before=json.loads((root/'evidence/final-dependencies-before.json').read_text()); after=deps()
    dump(root/'evidence/final-dependencies-after.json',after)
    assert before==after, 'source/compiled dependencies changed during verification'
    results=json.loads((root/'evidence/verification-results.json').read_text())
    assert [r['module'] for r in results]==paths
    for r in results:
        assert r['exit_code']==0 and r['source_sha256_before']==r['source_sha256_after']==sha(root/r['module'])
        assert r['log_sha256']==sha(root/r['log'])
        assert not re.search(r'\b(error|warning|sorryAx|DISALLOWED)\b', (root/r['log']).read_text())
    print(json.dumps({'modules':len(results),'all_exit_zero':True,'dependency_count':len(after['files']),'dependency_stable':True,'axiom_checks':sum((root/r['log']).read_text().count('axiom footprint') for r in results),'noncircularity_checks':sum((root/r['log']).read_text().count('non-circular ✓') for r in results)}))
elif mode=='whitespace':
    staged=Path(tempfile.mkdtemp(prefix='veritile-mixed-small-whitespace-'))
    for rel in paths:
        d=staged/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root/rel,d)
        assert sha(d)==sha(root/rel)
    commands=[]
    for args in [['git','init','--quiet'],['git','add','--',*paths],['git','diff','--cached','--check']]:
        r=subprocess.run(args,cwd=staged,text=True,capture_output=True)
        commands.append({'command':args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        assert r.returncode==0,r.stderr+r.stdout
    dump(root/'evidence/whitespace-check.json',{'method':'actual git diff --cached --check on exact source bytes','staging_directory':str(staged),'files':{p:sha(root/p) for p in paths},'commands':commands})
    print(json.dumps({'staged_whitespace_exit_code':0,'files':len(paths)}))
