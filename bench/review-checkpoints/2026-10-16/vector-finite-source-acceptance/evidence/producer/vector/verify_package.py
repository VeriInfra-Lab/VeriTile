from pathlib import Path
import hashlib,json
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
r=load(W/'evidence/receipt.json')
assert sha(W/'evidence/receipt.json')==(W/'evidence/receipt.sha256').read_text().split()[0]
assert r['frozen']
for rel,h in r['modules'].items():assert sha(W/r['producer_module_source_paths'][rel])==h,rel
for rel,h in r['artifacts'].items():assert sha(W/rel)==h,rel
for rel,e in load(W/'evidence/dependencies-before.json')['project'].items():assert sha(R/rel)==e['sha256'],rel
print(json.dumps({'status':'passed','receipt_sha256':sha(W/'evidence/receipt.json'),'artifacts':len(r['artifacts']),'modules':len(r['modules']),'dependency_sources':r['project_dependency_count']}))
