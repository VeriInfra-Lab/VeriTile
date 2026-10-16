"""Read-only candidate review plus helper-only tests; never run finalize.py."""
from pathlib import Path
import ast,collections,difflib,hashlib,importlib.util,json,subprocess,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent
P=W.parent/'next-three-finalizer-summary-correction-001'
O=W.parent/'next-three-source-integration-scripts-revision2'
V=W.parent/'next-three-source-integration-run-001/005-verify'
expected='0b84bf2e898e7dc81e65bf1df7306ce9048f13e9bc1a1791b2acb605692c9921'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
checks=[];pins={}
def check(label,cond):
    checks.append({'name':label,'passed':bool(cond)})
    assert cond,label
def pin(p,h=None):
    p=Path(p);d=sha(p)
    if h is not None:check('pin '+str(p),d==h)
    pins[str(p)]={'sha256':d,'size':p.stat().st_size}
def save(name,v):
    p=W/name;assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')
pin(P/'candidate-receipt.json',expected);receipt=load(P/'candidate-receipt.json')
for rel,h in receipt['artifacts'].items():pin(P/rel,h)
pin(O/'candidate-receipt.json',receipt['original_scripts_receipt_sha256'])
for rel,h in load(O/'candidate-receipt.json')['artifacts'].items():pin(O/rel,h)
original=(O/'finalize.py').read_text();candidate=(P/'finalize.py').read_text()
patch=''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='original/finalize.py',tofile='corrected/finalize.py'))
check('complete stored patch equals independent diff',patch==(P/'finalize.patch').read_text())
oldtree,newtree=ast.parse(original),ast.parse(candidate)
def calls(tree,name):
    return collections.Counter(ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name)
old_requires,new_requires=calls(oldtree,'require'),calls(newtree,'require')
removed=old_requires-new_requires
check('only removed require is old trailing-OK summary',sum(removed.values())==1 and 'Unclear unittest completion:' in next(iter(removed)) and "attr='endswith'" in next(iter(removed)))
for name in ['exact','state','preservation','verification_inputs','explicit_base']:
    check('all old '+name+' calls retained',not(calls(oldtree,name)-calls(newtree,name)))
check('original common source path precedes import',candidate.index('_sys.path.insert(0, str(_ORIGINAL_SCRIPTS))')<candidate.index('from common import *'))
check('original common has no summary shadow',(O/'unittest_summary.py').exists() is False)
check('correction hash mandatory and checked',"p.add_argument('--finalizer-correction-sha256',required=True)" in candidate and 'correction=verify_scripts(_OVERRIDE_ROOT,a.finalizer_correction_sha256)' in candidate)
check('correction provenance archive precedes seal',candidate.index("b/'evidence/finalizer-summary-correction'")<candidate.index("write(b/'validation.json',v)"))
check('all existing complete-run conditions retained',"observed['phase']=='verify' and observed['state']=='completed' and observed.get('inputs_unchanged') is True" in candidate)
check('all existing run log hashes retained',"exact(folder/command['log'],command['log_sha256'])" in candidate)
check('global exit one and exact scope failure retained',"commands['global']['exit_code']==1" in candidate and "scope['errors']==['172 original-program scopes remain unaccepted']" in candidate)
before={c['name']:c for c in load(V/'run.json')['commands'] if c['name'] in ['bench-tests','shard-tests']}
observed=load(P/'observed-completed-tests.json')
for name,c in before.items():
    check('completed observed command '+name,c['state']=='completed' and type(c['exit_code']) is int and c['exit_code']==0)
    pin(V/c['log'],c['log_sha256']);check('candidate observed log exact '+name,c['log_sha256']==observed[name]['log_sha256'])
cmd=[sys.executable,'-B',str(P/'test_summary.py')]
run=subprocess.run(cmd,cwd=W,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(W/'tests.log').write_text(run.stdout)
save('test-run.json',{'command':cmd,'cwd':str(W),'exit_code':run.returncode,'log':'tests.log','log_sha256':sha(W/'tests.log'),'finalizer_executed':False})
check('five helper tests pass independently',run.returncode==0 and 'Ran 5 tests in ' in run.stdout and run.stdout.rstrip().endswith('OK'))
spec=importlib.util.spec_from_file_location('independent_summary',P/'unittest_summary.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
parse=mod.observed_unittest_count
base='Ran 1 test in 0.001s\n\nOK\n'
mutations=[('false exit',base,False),('float exit',base,0.0),('negative exit',base,-1),('prior failure','FAILED (errors=1)\n'+base,0),('prior duplicate status','OK\n'+base,0),('later failure',base+'FAILED\n',0),('later duplicate status',base+'OK\n',0),('extra count',base+'Ran 2 tests in 0.1s\n',0),('negative count',base.replace('Ran 1','Ran -1'),0),('zero count',base.replace('Ran 1','Ran 0'),0),('intervening data',base.replace('\n\nOK','\nmessage\nOK'),0),('skipped success',base.replace('OK','OK (skipped=1)'),0),('unexpected success',base.replace('OK','FAILED (unexpected successes=1)'),0)]
for name,data,code in mutations:
    try:parse(data,code);reject=False
    except ValueError:reject=True
    check('independent negative '+name,reject)
check('single test plus buffered JSON accepted',parse(base+'{"inventory":184}\n',0)==1)
counts={name:parse((V/c['log']).read_text(),c['exit_code']) for name,c in before.items()}
check('actual bench count retained',counts['bench-tests']==283)
after={c['name']:c for c in load(V/'run.json')['commands'] if c['name'] in before}
check('completed test command records unchanged',before==after)
for path,row in pins.items():check('final unchanged '+path,sha(path)==row['sha256'] and Path(path).stat().st_size==row['size'])
save('checks.json',{'checks':checks,'count':len(checks),'all_passed':True})
save('input-pins.json',pins)
save('observed-test-snapshot.json',{'completed_test_commands':before,'counts':counts,'whole_live_run_manifest_pinned':False,'reason':'Global run may still be active; only completed test command records and immutable logs were checked.'})
print(json.dumps({'checks':len(checks),'files':len(pins),'tests':5,'independent_negative_cases':len(mutations),'actual_counts':counts}))
