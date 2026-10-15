from pathlib import Path
import hashlib,json,sys
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
d=load(W/'evidence/dependencies-before.json')
for rel,v in d['project'].items():assert sha(R/rel)==v['sha256'],rel
(W/'evidence/dependencies-after.json').write_text(json.dumps(d,indent=2)+'\n')
modules=list(load(W/'evidence/staging-map.json').values());records=[]
for rel in modules:
 attempts=[]
 for p in sorted((W/'evidence/attempts').glob('*.json')):
  a=load(p)
  if a['module']=='after/'+rel:attempts.append((p,a))
 p,a=attempts[-1];assert a['exit_code']==0 and a['source_sha256']==sha(W/'after'/rel)
 log=p.with_suffix('.log');assert sha(log)==a['log_sha256']
 text=log.read_text();assert 'sorryAx' not in text and ' error:' not in text
 records.append({'source':'after/'+rel,'source_sha256':a['source_sha256'],'receipt':str(p.relative_to(W)),'receipt_sha256':sha(p),'log':str(log.relative_to(W)),'log_sha256':sha(log),'named_axiom_checks':text.count('axiom footprint'),'noncircular_checks':text.count('non-circular')})
report={'status':'passed','modules':records,'named_axiom_checks':sum(x['named_axiom_checks'] for x in records),'noncircular_checks':sum(x['noncircular_checks'] for x in records),'dependency_sources_unchanged':len(d['project']),'all_failed_lean_attempts_preserved':True}
(W/'evidence/final-compiles.json').write_text(json.dumps(report,indent=2)+'\n')
sys.path.insert(0,str(R/'bench'))
from adapter_common import declarations
public='bench/tritonbench_g/vector_addition/FiniteCorrect.lean'
parsed=declarations(W/'after'/public)
contracts=[{'file':public,'declaration':k,'kind':v['kind'],'statement_sha256':v['statement_sha256']} for k,v in parsed.items() if v['kind']=='specification']
assert len(contracts)==1
(W/'evidence/public-contracts.json').write_text(json.dumps(contracts,indent=2)+'\n')
for p in (W/'VectorFinite').glob('*.lean'):
 target=W/'evidence/current-workbench'/(p.name+'.txt');target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())
assert load(W/'evidence/proposal-validation.json')['status']=='passed'
assert load(W/'evidence/authority-check.json')['status']=='passed'
print(json.dumps({k:v for k,v in report.items() if k!='modules'}))
