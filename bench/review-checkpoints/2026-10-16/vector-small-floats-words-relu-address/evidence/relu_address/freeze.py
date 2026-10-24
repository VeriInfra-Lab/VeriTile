"""Bind final checks to unchanged new, frozen and repository source inputs."""
from pathlib import Path
import hashlib
import json
import re
import sys

root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
original=root.parent/'relu-original-wip'
words=root.parent/'relu-integer-word-refinement-wip'
modules=['bench/tritonbench_g/relu_triton_kernel/'+p+'.lean' for p in
         ['Proofs/FiniteAddress','Proofs/ByteAddress','AddressGridCorrect']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
frozen=[(original,'ed4a852afafce5efd027f62565e32afe42146eb9c0b140d872990a5da211fd90',
         'f484d80a8fbb1ff238a4a2ebb360b5d0991938d6b79437adb9c1b6f0b8d49453'),
        (words,'bf37c9bb89af701ca3737ce96fbc21d9a707d8a3d7c41e1a47d11a2e73c31276',
         'da4d9317436b3f73bfc2aac6e1c0a795501b0af18904e7e065a185175ba60e57')]
for p,r,s in frozen:
    assert sha(p/'evidence/receipt.json')==r
    assert sha(p/'evidence/scope-review.json')==s
    for f,h in read(p/'evidence/receipt.json')['modules'].items():assert sha(p/f)==h
for relative in modules:
    text=(root/relative).read_text()
    assert text.endswith('\n') and all(line==line.rstrip(' \t') for line in text.splitlines())

def dependencies():
    seen,external={},set()
    queue=modules[:]
    while queue:
        rel=queue.pop()
        if rel in seen:continue
        candidates=[(root,'package'),(words,'frozen_words_package'),
                    (original,'frozen_original_package'),(repo,'repository')]
        found=next(((base/rel,kind) for base,kind in candidates if (base/rel).is_file()),None)
        if found is None:external.add(rel);continue
        path,kind=found
        seen[rel]={'source_kind':kind,'sha256':sha(path)}
        for line in path.read_text().splitlines():
            if line.startswith('import '):
                for name in line[7:].split():
                    if name.startswith('--'):break
                    queue.append(name.replace('.','/')+'.lean')
    for rel in ['bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py',
                'lakefile.toml','lake-manifest.json','lean-toolchain']:
        seen[rel]={'source_kind':'repository','sha256':sha(repo/rel)}
    routes=read(root/'evidence/address-route-review.json')
    for entry in routes['profiles']:
        assert sha(Path(entry['path']))==entry['sha256']
    return {'files':dict(sorted(seen.items())),'external_imports':sorted(external),
        'frozen_receipts':{str(p):{'receipt':r,'scope':s} for p,r,s in frozen},
        'source_route_sha256':sha(root/'evidence/address-route-review.json'),
        'audited_ttir_sha256':{entry['path']:entry['sha256'] for entry in routes['profiles']}}

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
print(json.dumps({'mode':mode,'bound_files':len(current['files']),'audited_ttir':len(current['audited_ttir_sha256'])}))
