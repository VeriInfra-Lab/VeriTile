import sys;sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
W=Path(__file__).resolve().parent;C=Path('/tmp/veritile-standardization-20261015/quant-transpose-finite-acceptance-revision1-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(n,d):(W/n).write_text(json.dumps(d,indent=2)+'\n')
receipt=load(C/'evidence/receipt.json');pin=sha(C/'evidence/receipt.json')
assert receipt['frozen'] is True and receipt['revision']==1
for p,h in receipt['modules'].items():assert sha(C/p)==h and sha(C/receipt['checkpoint_module_source_paths'][p])==h
for p,h in receipt['module_objects'].items():assert sha(C/p)==h
for p,h in receipt['artifacts'].items():assert sha(C/p)==h,p
dependencies={};external={}
for f in [C/'evidence/dependencies-before.json']+sorted((C/'evidence').glob('dependencies-added-before-*.json')):
 snap=load(f)
 for mod,e in snap.get('project',{}).items():
  assert sha(e['source'])==e['sha256'] and sha(e['olean'])==e['olean_sha256'],mod
  dependencies[mod]=e
 for mod,e in snap.get('external_objects',{}).items():assert sha(e['olean'])==e['sha256'];external[mod]=e
 for p,h in snap.get('input_sha256',{}).items():assert sha(p)==h,p
save('frozen-input-verification.json',{'status':'passed','receipt_sha256':pin,'modules':receipt['modules'],'artifacts_verified':len(receipt['artifacts']),'module_objects_verified':len(receipt['module_objects']),'project_sources_and_objects':len(dependencies),'external_objects':len(external)})
modules=load(C/'evidence/module-layout.json')['modules'];overlay=W/'overlay';overlay.mkdir(exist_ok=True)
for module,entry in dependencies.items():
 relative=module.replace('.','/')+'.olean';link=overlay/relative
 if str(Path(relative).with_suffix('.lean')) in modules:continue
 link.parent.mkdir(parents=True,exist_ok=True)
 if not link.exists():link.symlink_to(entry['olean'])
for rel in modules:
 p=overlay/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C/rel,p)
base=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=R,text=True).strip()
env=dict(os.environ,LEAN_PATH=str(overlay)+':'+base)
attempts=W/'compile-attempts';attempts.mkdir(exist_ok=True);results=[]
for rel in modules:
 p=overlay/rel;number=len(list(attempts.glob('*.json')))+1;tag=f'{number:04d}-{p.stem}'
 log=attempts/(tag+'.log');source=sha(p);cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(overlay),str(p),'-o',str(p.with_suffix('.olean'))]
 started=time.monotonic()
 with log.open('w') as stream:r=subprocess.run(cmd,cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
 result={'module':rel,'receipt_sha256':pin,'source_sha256':source,'source_unchanged':sha(p)==source==receipt['modules'][rel],'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'exit_code':r.returncode,'seconds':round(time.monotonic()-started,3),'log':str(log.relative_to(W)),'log_sha256':sha(log)}
 if r.returncode==0:result['object_sha256']=sha(p.with_suffix('.olean'))
 (attempts/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n');results.append(result)
 print(json.dumps({'module':rel,'exit_code':r.returncode,'seconds':result['seconds']}),flush=True)
 if r.returncode:print(log.read_text(),flush=True);raise SystemExit(r.returncode)
 assert 'sorryAx' not in log.read_text() and 'DISALLOWED' not in log.read_text()
save('independent-compiles.json',{'status':'passed','receipt_sha256':pin,'modules':results,'named_axiom_checks':sum((W/r['log']).read_text().count('axiom footprint') for r in results),'noncircular_checks':sum((W/r['log']).read_text().count('non-circular') for r in results)})
sys.path.insert(0,str(R/'bench'))
from adapter_common import declarations,canonical
from check_original_program_scope import validate_case,declaration_scope,validate_deferrals
row=load(C/'evidence/scope-row-proposal.json');declaration=load(C/'evidence/declaration-proposal.json');delta=load(C/'evidence/numerical-authority-delta.json');cases=load(C/'evidence/scope-cases.json')
for case in cases:validate_case(case)
assert len({canonical(c) for c in cases})==len(cases)
declaration_scope(declaration,row['source_inventory']);validate_deferrals('quant_transpose_kernel',row['deferred_numerical_assumptions'],{'rules':delta['rules']})
actual=declarations(overlay/declaration['file'])[declaration['declaration']]
assert actual['statement_sha256']==declaration['statement_sha256']
statement=actual['statement'] if 'statement' in actual else ''
assert sha(C/'evidence/receipt.json')==pin
for p,h in receipt['modules'].items():assert sha(C/p)==h
save('independent-schema.json',{'status':'passed','receipt_sha256':pin,'case_count':len(cases),'scalar_families':len(delta['rules']),'scope_bindings':len(delta['provenance']),'public_statement_sha256':declaration['statement_sha256'],'checks':['unchanged validate_case','unique cases','unchanged declaration_scope','unchanged validate_deferrals','actual final specification statement hash','final candidate source and receipt unchanged']})
print(json.dumps({'status':'passed','receipt_sha256':pin,'modules':len(modules),'schema_cases':len(cases)}),flush=True)
