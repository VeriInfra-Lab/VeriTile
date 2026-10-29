import sys;sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
W=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
r=load(W/'evidence/receipt.json');assert sha(W/'evidence/receipt.json')==(W/'evidence/receipt.sha256').read_text().split()[0] and r['frozen']
for rel,h in r['modules'].items():assert sha(W/r['producer_module_source_paths'][rel])==h,rel
for rel,h in r['artifacts'].items():assert sha(W/rel)==h,rel
before=load(W/'evidence/dependencies-before.json');assert before==load(W/'evidence/dependencies-after.json')
for e in before['project'].values():assert sha(e['source'])==e['sha256'] and sha(e['olean'])==e['olean_sha256']
for e in before['external_objects'].values():assert sha(e['olean'])==e['sha256']
for m in load(W/'evidence/final-compiles.json')['modules']:assert sha(W/m['object'])==m['object_sha256']
print(json.dumps({'status':'passed','receipt_sha256':sha(W/'evidence/receipt.json'),'artifacts':len(r['artifacts']),'modules':len(r['modules']),'dependency_sources':r['project_dependency_count'],'dependency_objects':r['dependency_object_count']}))
