import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import hashlib,json
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
def save(n,x):(W/'evidence'/n).write_text(json.dumps(x,indent=2)+'\n')
before=load(W/'evidence/dependencies-before.json');assert before==load(W/'evidence/dependencies-after.json')
for name,e in before['project'].items():
 assert sha(e['source'])==e['sha256'] and sha(e['olean'])==e['olean_sha256'],name
for name,e in before['external_objects'].items():assert sha(e['olean'])==e['sha256'],name
for p,h in before['input_sha256'].items():assert sha(p)==h,p
modules=list(load(W/'evidence/staging-map.json').values());records=[]
attempts=[(p,load(p)) for p in sorted((W/'evidence/attempts').glob('*.json'))]
for p,a in attempts:
 assert a['source_unchanged'] and sha(W/a['log'])==a['log_sha256']
 assert sha(p.with_suffix('.lean.txt'))==a['source_sha256']
for rel in modules:
 p,a=[(p,a) for p,a in attempts if a['source']==rel][-1]
 assert a['exit_code']==0 and a['source_sha256']==sha(W/rel)==sha(W/'after'/rel)==sha(W/(rel+'.txt'))
 log=W/a['log'];text=log.read_text();assert 'sorryAx' not in text and ' error:' not in text
 records.append({'source':'after/'+rel,'source_sha256':a['source_sha256'],'compiled_source':rel,'object':str(Path(rel).with_suffix('.olean')),'object_sha256':sha((W/rel).with_suffix('.olean')),'receipt':str(p.relative_to(W)),'receipt_sha256':sha(p),'log':str(log.relative_to(W)),'log_sha256':sha(log),'named_axiom_checks':text.count('axiom footprint'),'noncircular_checks':text.count('non-circular')})
report={'status':'passed','modules':records,'named_axiom_checks':sum(x['named_axiom_checks'] for x in records),'noncircular_checks':sum(x['noncircular_checks'] for x in records),'dependency_sources_unchanged':len(before['project']),'dependency_objects_unchanged':len(before['project'])+len(before['external_objects']),'all_failed_lean_attempts_preserved':True,'lean_attempts':len(attempts),'failed_lean_attempts':sum(a['exit_code']!=0 for _,a in attempts)}
save('final-compiles.json',report)
sys.path.insert(0,str(R/'bench'))
from adapter_common import declarations,imports
public='bench/tritonbench_g/triton_mul2/FiniteCorrect.lean';parsed=declarations(W/'after'/public)
contracts=[{'file':public,'declaration':k,'kind':v['kind'],'statement_sha256':v['statement_sha256']} for k,v in parsed.items() if v['kind']=='specification'];assert len(contracts)==1
save('public-contracts.json',contracts)
# Independently re-enumerate every direct external import, including imports of
# new candidate modules, to prevent a direct external object snapshot omission.
external=set()
for rel in modules+list(load(W/'evidence/dependency-closure.json')['project']):
 path=W/rel if rel in modules else R/rel
 for name in imports(path.read_text()):
  target=name.replace('.','/')+'.lean'
  if target not in modules and not (R/target).is_file():external.add(name)
assert external==set(before['external_objects']),(external-set(before['external_objects']),set(before['external_objects'])-external)
native=[];counts=Counter()
for path in sorted((W/'native-attempts').glob('*/receipt.json')):
 receipt=load(path)
 for rel,h in receipt['artifacts'].items():assert sha(path.parent/rel)==h
 for row in receipt['records']:counts[row['result']['status']]+=1
 native.append({'receipt':str(path.relative_to(W)),'sha256':sha(path),'records':len(receipt['records']),'gpu_launches':receipt['gpu_launches']})
assert counts=={'compiled':30,'rejected':42,'frontend_passed':6}
save('native-evidence-summary.json',{'receipts':native,'record_count':sum(counts.values()),'counts':dict(counts),'gpu_launches':0,'native_scalar_validation':False})
assert load(W/'evidence/proposal-validation.json')['status']=='passed' and load(W/'evidence/authority-check.json')['status']=='passed'
print(json.dumps({k:v for k,v in report.items() if k!='modules'}))
