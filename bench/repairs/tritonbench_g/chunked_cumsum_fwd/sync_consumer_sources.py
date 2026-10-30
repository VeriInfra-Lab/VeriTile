"""Synchronize only this task's new proof sources into its isolated consumer.

Does not compile, touch shared staging, or replace historical public entries.
"""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[4]
task=Path(__file__).resolve().parent
consumer=root/'.veritile-cache/minimum-staging/consumers/chunked_cumsum_fwd/src'
target=consumer/task.relative_to(root)
(target/'Proofs').mkdir(parents=True,exist_ok=True)
records={}
for p in sorted((task/'Proofs').glob('Kernel*.lean')):
 q=target/'Proofs'/p.name
 data=p.read_bytes()
 if not q.exists() or q.read_bytes()!=data:q.write_bytes(data)
 records[str(q.relative_to(root))]={'origin':str(p.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest()}
for name in ['Kernels','Contract','Correct','FPEquiv']:
 p=target/(name+'.lean'); assert p.is_file()
 records[str(p.relative_to(root))]={'origin':'consumer_standard_entry','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
ev=root/'.veritile-cache/kernel-evidence/chunked_cumsum_fwd'; ev.mkdir(parents=True,exist_ok=True)
(ev/'draft-effective-sources.json').write_text(json.dumps({'status':'uncompiled_draft','files':records},indent=2)+'\n')
print(len(records),'task sources synchronized; no Lean invocation')
