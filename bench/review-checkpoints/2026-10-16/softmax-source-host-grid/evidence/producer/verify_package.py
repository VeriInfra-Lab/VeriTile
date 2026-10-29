from pathlib import Path
import hashlib,json
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((W/'evidence/receipt.json').read_text())
assert sha(W/'evidence/receipt.json')==(W/'evidence/receipt.sha256').read_text().split()[0]
for rel,h in d['frozen_files'].items():assert sha(W/rel)==h,rel
for dep in json.loads((W/'evidence/repo-dependencies.json').read_text())['project_import_closure']:
 p=(R if dep['origin']=='repository' else W/'after')/dep['path'];assert sha(p)==dep['source_sha256'],str(p)
print(json.dumps({'status':'passed','frozen_files':len(d['frozen_files']),'receipt_sha256':sha(W/'evidence/receipt.json')}))
