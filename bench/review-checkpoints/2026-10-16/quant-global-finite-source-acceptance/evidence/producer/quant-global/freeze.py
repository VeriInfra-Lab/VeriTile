from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys,time
WORK=Path(__file__).resolve().parent
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
FROZEN=WORK/'frozen-v2'
LEAN='/tmp/veritile-lean-toolchain/bin/lean'
ORDER=['FiniteScalar','FiniteGrid','FiniteHost','FiniteAdmission','FiniteCorrect','FiniteOutcomes','FinitePhysical','FinitePhysicalExecution','FullSource','RetainedBridge','SupportInventory']
NAMES={'FiniteHost':'FiniteHostCorrect','FiniteAdmission':'FiniteDTypeCorrect','FiniteCorrect':'FiniteCorrect','FiniteOutcomes':'FiniteOutcomes','FullSource':'FullSourceCorrect'}
MAPPING={n:'bench.tritonbench_g.quantize_global.'+NAMES.get(n,'Proofs.'+('FiniteRetainedBridge' if n=='RetainedBridge' else 'FiniteSupportInventory' if n=='SupportInventory' else n)) for n in ORDER}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def entry(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
if FROZEN.exists():raise RuntimeError('Frozen destination exists; preserve it.')
(FROZEN/'evidence/build-attempts').mkdir(parents=True)
shutil.copytree(WORK/'evidence',FROZEN/'evidence/producer-attempts')
for folder in ['dtype-probes','torch-source','host-finite-probes','autotune-probes','native-attempts']:
 shutil.copytree(WORK/folder,FROZEN/'evidence'/folder,ignore=shutil.ignore_patterns('cache','__pycache__'))
for p in WORK.glob('*.py'):shutil.copy2(p,FROZEN/p.name)
for p in WORK.glob('*.log'):shutil.copy2(p,FROZEN/'evidence'/p.name)
base=json.loads((WORK/'evidence/FiniteOutcomes-latest.json').read_text())['LEAN_PATH'].split(':')[1:]
env=dict(os.environ,LEAN_PATH=str(FROZEN/'overlay')+':'+':'.join(base))
modules={};transforms=[]
for name in ORDER:
 source=WORK/(name+'.lean');relative=Path(*MAPPING[name].split('.')).with_suffix('.lean')
 target=FROZEN/'overlay'/relative;target.parent.mkdir(parents=True,exist_ok=True)
 mapped=source.read_text()
 for old,new in MAPPING.items():mapped=re.sub('^import '+old+'$', 'import '+new,mapped,flags=re.M)
 target.write_text(mapped);modules[str(relative)]=sha(target)
 frozenSource=FROZEN/(str(relative)+'.txt');frozenSource.parent.mkdir(parents=True,exist_ok=True);frozenSource.write_bytes(target.read_bytes())
 transforms.append({'source':str(source),'source_sha256':sha(source),'target':str(relative),'target_sha256':sha(target),'transformation':'Exact top-level local import names only, according to import_mapping; namespace/declaration/statement/proof bytes unchanged.'})
def support(src,dst):
 dst.mkdir(parents=True,exist_ok=True)
 for p in src.iterdir():
  q=dst/p.name
  if q.exists():
   if p.is_dir() and q.is_dir():support(p,q)
   continue
  q.symlink_to(p,target_is_directory=p.is_dir())
support(ROOT/'.lake/build/lib/lean/bench',FROZEN/'overlay/bench')
builds=[]
for idx,name in enumerate(ORDER,1):
 p=FROZEN/'overlay'/Path(*MAPPING[name].split('.')).with_suffix('.lean')
 tag=f'{idx:04d}-{name}';saved=FROZEN/'evidence/build-attempts'/(tag+'.lean.txt');saved.write_bytes(p.read_bytes())
 log=FROZEN/'evidence/build-attempts'/(tag+'.log');cmd=[LEAN,'--root='+str(FROZEN/'overlay'),str(p),'-o',str(p.with_suffix('.olean'))]
 start=time.monotonic()
 with log.open('w') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 record={'module':MAPPING[name],'source_sha256':sha(p),'saved_source':str(saved.relative_to(FROZEN)),'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'exit_code':r.returncode,'seconds':time.monotonic()-start,'log':str(log.relative_to(FROZEN)),'log_sha256':sha(log),'source_unchanged':sha(p)==sha(saved),'compiled_artifacts':[entry(q) for q in [p.with_suffix('.olean'),Path(str(p.with_suffix('.olean'))+'.private'),Path(str(p.with_suffix('.olean'))+'.server')] if q.is_file()]}
 put(FROZEN/'evidence/build-attempts'/(tag+'.json'),record);builds.append(record);print(json.dumps(record),flush=True)
 if r.returncode:put(FROZEN/'FAILED.json',builds);sys.exit(r.returncode)
moduleNames=sorted({l.removeprefix('dependency ') for l in log.read_text().splitlines() if l.startswith('dependency ')})
binaryRoots=[FROZEN/'overlay',*[Path(p) for p in base]]
sourceRoots=[FROZEN/'overlay',ROOT,*[p for p in Path('/tmp/veritile-tritonbench-lake/packages').iterdir() if p.is_dir()],Path('/tmp/veritile-lean-toolchain/src/lean'),Path('/tmp/veritile-lean-toolchain/src/lean/lake')]
deps=[];project={}
for name in moduleNames:
 rel=Path(*name.split('.'));binary=next((p/rel.with_suffix('.olean') for p in binaryRoots if (p/rel.with_suffix('.olean')).is_file()),None)
 source=next((p/rel.with_suffix('.lean') for p in sourceRoots if (p/rel.with_suffix('.lean')).is_file()),None)
 assert binary is not None and source is not None,name
 arts=[entry(binary)]
 for suffix in ['.private','.server']:
  p=Path(str(binary)+suffix)
  if p.is_file():arts.append(entry(p))
 deps.append({'module':name,'source':entry(source),'artifacts':arts})
 if source.is_relative_to(ROOT):project[str(source.relative_to(ROOT))]={'sha256':sha(source),'module':name}
put(FROZEN/'evidence/dependencies.json',deps)
put(FROZEN/'evidence/dependencies-project.json',{'kind':'Exact existing project dependencies reused by freshly compiled mapped candidate modules; recorded after compile. Not a trust-zero rebuild.','project':project})
source=ROOT/'bench/tritonbench_g/quantize_global/quantize_global.py'
(FROZEN/'original').mkdir();shutil.copy2(source,FROZEN/'original/quantize_global.py')
put(FROZEN/'evidence/import-mapping.json',{'mapping':MAPPING,'modules':modules,'transformations':transforms})
subprocess.run([sys.executable,str(WORK/'make_proposals.py'),str(FROZEN)],cwd=ROOT,check=True)
for helper in ['make_proposals.py','README.md']:
 shutil.copy2(WORK/helper,FROZEN/helper)
# Every owned source/evidence/helper artifact is bound. Symlinked dependency
# trees are deliberately excluded; their actual artifacts have separate hashes.
artifacts={}
for p in sorted(FROZEN.rglob('*')):
 if p.is_file() and not p.is_symlink() and 'overlay' not in p.relative_to(FROZEN).parts:
  artifacts[str(p.relative_to(FROZEN))]=sha(p)
receipt={'schema':1,'kind':'quantize_global_complete_finite_source_candidate','task':'quantize_global','frozen':True,
 'status':'producer_candidate_pending_independent_root_review_and_canonical_integration','source':entry(source),
 'modules':modules,'checkpoint_module_source_paths':{p:p+'.txt' for p in modules},'artifacts':artifacts,
 'import_mapping':MAPPING,'transformations':transforms,'builds':builds,'dependency_count':len(deps),
 'dependencies':{'path':'evidence/dependencies.json','sha256':sha(FROZEN/'evidence/dependencies.json'),'all_sources_resolved':True},
 'trusted_boundary':'bench/original_program_acceptance.md; actual ordinary Torch tensor metadata and primitive mechanics, fresh allocation, finite lifetime/address/launch/load/store semantics. Task-specific dtype dispatch, producer/consumer wiring, actual typed programs/trials, values, frames, finite indexes and source hazards are proved.',
 'runtime_profile':{'torch':'2.13.0+cu130','torch_commit':'cf30153c4c131c8164ee7798e5022d810682e2cb','triton':'3.7.1','torch_default_dtype':'float32','target':{'backend':'cuda','arch':90,'warp_size':32},'layout':'ordinary dense strided CPU/CUDA tensors, valid Torch size/storage metadata'},
 'native_gpu_executions':0,'whole_original_program_accepted':0,'numerical_authority_integrated':False,
 'axioms':'All candidate modules pass #auditModuleAxioms; only propext, Classical.choice, Quot.sound.',
 'primary_public_statement':'QuantizeGlobalFinite.quantize_global_full_source_outcomes',
 'preservation':'Frozen-v1 fp32 host/grid and prior rotary packages untouched. All 20+ local Lean attempts preserve exact source, raw log, command, and real exit code; all final mapped modules freshly compiled with separately preserved source/log/status. Existing imported dependencies reused and fingerprinted, not rebuilt from trust zero.',
 'toolchain':{'lean':entry(Path(LEAN)),'version':subprocess.check_output([LEAN,'--version'],text=True).strip(),'lean_toolchain':entry(ROOT/'lean-toolchain'),'lake_manifest':entry(ROOT/'lake-manifest.json')},
 'remaining_review':['Independent substantive root review of typed-source correspondence, runtime boundary, all source rejection/hazard classifications and the exact delta.','Canonical public catalog, schema2 coverage and exact numerical authority integration.','Numerical/native conformance remains explicitly unvalidated; no GPU execution available.','Other Torch default floating dtypes, other backends/architectures, sparse/quantized layouts and arbitrary Python objects are outside this explicit runtime API boundary, not claimed covered.']}
put(FROZEN/'evidence/receipt.json',receipt);put(FROZEN/'receipt.json',receipt)
print('FROZEN_RECEIPT_SHA256 '+sha(FROZEN/'evidence/receipt.json'),flush=True)
