"""Read-only hash/receipt/archive verification of the frozen task packet."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');E=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(r):assert sha(Path(r['path']))==r['sha256'],r['path']
d=json.loads((E/'delivery.json').read_text());f=json.loads((E/'freeze-result.json').read_text())
check(f['delivery']);check(f['archive'])
for r in d['frozen_files']:check(r)
b=json.loads((E/'task-builds.json').read_text());assert b['count']==48
for row in b['receipts']:
 for k in ['source','receipt','original_receipt']:check(row[k])
 receipt=json.loads(Path(row['receipt']['path']).read_text());assert receipt['source_sha256']==row['source']['sha256'] and receipt['exit_code']==0
 for r in row['objects']:check(r)
c=json.loads((E/'import-closure.json').read_text())
for r in c['records']:
 check(r['source']);check(r['effective_olean'])
 if r['effective_ilean']:check(r['effective_ilean'])
 rel=r['module'].replace('.','/')+'.olean'
 selected=next(Path(p)/rel for p in b['effective_search_order'] if (Path(p)/rel).is_file())
 assert str(selected)==r['effective_olean']['path']
with tarfile.open(E/'packet.tar.gz') as arc:
 expected={str(Path(r['path']).relative_to(ROOT)):r['sha256'] for r in d['frozen_files']};expected[str((E/'delivery.json').relative_to(ROOT))]=sha(E/'delivery.json')
 assert set(arc.getnames())==set(expected)
 for path,digest in expected.items():assert hashlib.sha256(arc.extractfile(path).read()).hexdigest()==digest,path
log=(E/'statement-audit.log').read_text();assert log.count('axiom footprint ⊆ standard base ✓')==8 and log.count('non-circular ✓')==2
assert all(x['status']=='pass' for x in json.loads((E/'native-static.json').read_text())['cases'])
print(json.dumps({'status':'pass','frozen_files':len(d['frozen_files']),'archive_members':len(expected),'task_build_receipts':48,'repository_closure':len(c['records']),'available_ilean':c['available_ilean_count'],'missing_reused_ilean':len(c['missing_reused_ilean_modules']),'public_axiom_audits':8,'noncircularity_audits':2,'source_or_object_mutations':False},indent=2))
