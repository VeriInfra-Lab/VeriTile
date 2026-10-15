"""Check exact source/olean dependencies and retain every real compile attempt."""
from pathlib import Path
import hashlib
import json
import re
import sys

root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
base=root.parent
cache=Path('/tmp/veritile-tritonbench-lake/build/lib/lean')
modules=['bench/tritonbench_g/relu_triton_kernel/'+p+'.lean' for p in
         ['Proofs/ByteCodec','Proofs/ByteStores','Proofs/ByteView','ByteGridCorrect']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
frozen=[('relu-finite-address-wip','fc98bfa9d776e2b5592506905eda9241efb16a4809c7ad112f500afc83d815f3',
         '6eaaf3bfa779c674ad6a6a3436354b9684a8d67e55c4e1d7d58ce892491c6e6e'),
        ('relu-integer-word-refinement-wip','bf37c9bb89af701ca3737ce96fbc21d9a707d8a3d7c41e1a47d11a2e73c31276',
         'da4d9317436b3f73bfc2aac6e1c0a795501b0af18904e7e065a185175ba60e57'),
        ('relu-original-wip','ed4a852afafce5efd027f62565e32afe42146eb9c0b140d872990a5da211fd90',
         'f484d80a8fbb1ff238a4a2ebb360b5d0991938d6b79437adb9c1b6f0b8d49453')]
for name,r,s in frozen:
    old=base/name
    assert sha(old/'evidence/receipt.json')==r and sha(old/'evidence/scope-review.json')==s
    for p,h in read(old/'evidence/receipt.json')['modules'].items():assert sha(old/p)==h
for relative in modules:
    text=(root/relative).read_text()
    assert text.endswith('\n') and all(line==line.rstrip(' \t') for line in text.splitlines())

def dependencies():
    seen,external={},set()
    queue=modules[:]
    while queue:
        rel=queue.pop()
        if rel in seen:continue
        candidates=[(root,'package')]+[(base/name,name) for name,_,_ in frozen]+[(repo,'repository')]
        found=next(((directory/rel,kind,directory) for directory,kind in candidates if (directory/rel).is_file()),None)
        if found is None:external.add(rel);continue
        path,kind,directory=found
        row={'source_kind':kind,'source_path':str(path),'sha256':sha(path)}
        if kind!='package':
            olean=(cache if kind=='repository' else directory)/Path(rel).with_suffix('.olean')
            assert olean.is_file(),str(olean)
            row.update(olean_path=str(olean),olean_sha256=sha(olean))
        seen[rel]=row
        for line in path.read_text().splitlines():
            if line.startswith('import '):
                for name in line[7:].split():
                    if name.startswith('--'):break
                    queue.append(name.replace('.','/')+'.lean')
    for rel in ['bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py',
                'lakefile.toml','lake-manifest.json','lean-toolchain']:
        seen[rel]={'source_kind':'repository','source_path':str(repo/rel),'sha256':sha(repo/rel)}
    route=base/'relu-finite-address-wip/evidence/address-route-review.json'
    ttirs={r['path']:r['sha256'] for r in read(route)['profiles']}
    for p,h in ttirs.items():assert sha(Path(p))==h
    return {'files':dict(sorted(seen.items())),'external_imports':sorted(external),
        'frozen_receipts':{str(base/name):{'receipt':r,'scope':s} for name,r,s in frozen},
        'inherited_route_review':{'path':str(route),'sha256':sha(route)},'audited_ttir_sha256':ttirs}

mode=sys.argv[1];assert mode in ('before','after')
current=dependencies()
path=root/('evidence/final-dependencies-'+mode+'.json')
path.write_text(json.dumps(current,indent=2)+'\n')
if mode=='after':
    assert read(root/'evidence/final-dependencies-before.json')==current
    rows=read(root/'evidence/verification-results.json')
    assert [row['module'] for row in rows]==modules
    for row in rows:
        assert row['exit_code']==0
        assert row['source_sha256_before']==row['source_sha256_after']==sha(root/row['module'])
        assert row['log_sha256']==sha(root/row['log'])
        assert not re.search(r'\b(error|sorryAx|DISALLOWED)\b|warning:',(root/row['log']).read_text())
    history=read(root/'evidence/verification-attempts.json')
    assert history[-len(rows):]==rows
    for row in history:
        assert row['log_sha256']==sha(root/row['log'])
        assert row['source_sha256_before']==row['snapshot_sha256']==sha(root/row['source_snapshot'])
print(json.dumps({'mode':mode,'bound_files':len(current['files']),
    'bound_oleans':sum('olean_sha256' in r for r in current['files'].values()),'inherited_ttir_count':len(current['audited_ttir_sha256'])}))
