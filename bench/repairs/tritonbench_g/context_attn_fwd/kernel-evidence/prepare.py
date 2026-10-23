"""Freeze source, build, dependency, and native evidence for this task only."""
from pathlib import Path
import hashlib,json,re,shutil,tarfile,io
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TASK=ROOT/'bench/repairs/tritonbench_g/context_attn_fwd'; E=TASK/'kernel-evidence'
CACHE=ROOT/'.veritile-cache/task-deliveries/context_attn_fwd'; OVER=ROOT/'.veritile-cache/kernel-build/lib/lean'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(name,data):(E/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
for name in ['entry.json','pre-revision-sources.tar.gz','baseline-build-order.json','initial-build.log','baseline-build.log','kernel-prelude-build.log','kernel-build.log','statement-audit.log','cpu-check.json','cpu-check.log','native-probe.json','native-probe.log','native_probe.py','native-static.json','native-static.log','native_static.py']:
 shutil.copy2(CACHE/name,E/name)
(E/'native').mkdir(exist_ok=True)
for p in sorted((CACHE/'native').glob('*')):shutil.copy2(p,E/'native'/p.name)
header=Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_occupancy.h')
write('resource-audit.json',{'profiles':[{k:c[k] for k in ['label','shared_bytes','SM90_233472_byte_SM_capacity_exceeded']} for c in json.loads((E/'native-static.json').read_text())['cases']], 'capacity_basis':pin(header),'capacity_excerpt_around_compute_90':header.read_text().splitlines()[889:941], 'runtime_launch_performed':False,'limitation':'BM128/D256 uses 262144 bytes, exceeding even the 228 KiB SM90 per-SM capacity. Successful runtime delivery remains an external applicability boundary; no guard is added.'})
lean=sorted(p for p in TASK.rglob('*.lean') if E not in p.parents);assert len(lean)==48
receipts=[];search=None;(E/'build-receipts').mkdir(exist_ok=True)
for p in lean:
 rp=OVER/p.relative_to(ROOT).with_suffix('.workspace-build.json');r=json.loads(rp.read_text())
 assert r['source_sha256']==sha(p) and r['exit_code']==0
 if search is None:search=r['lean_path']
 assert search==r['lean_path']
 for artifact,digest in r['objects'].items():assert sha(Path(artifact))==digest
 dest=E/'build-receipts'/p.relative_to(TASK).with_suffix('.json');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(rp,dest)
 receipts.append({'source':pin(p),'receipt':pin(dest),'original_receipt':pin(rp),'objects':[{'path':path,'sha256':digest} for path,digest in r['objects'].items()]})
write('task-builds.json',{'count':len(receipts),'receipts':receipts,'effective_search_order':search,'kind':'all current task modules freshly rebuilt; shared transitive caches reused, not a cold transitive build'})
def imports(p):return [s for line in p.read_text().splitlines() if line.startswith('import ') for s in line[7:].split() if not s.startswith('--')]
def source(m):return ROOT/(m.replace('.','/')+'.lean')
def effective(m,ext):
 rel=m.replace('.','/')+ext
 return next((Path(b)/rel for b in search if (Path(b)/rel).is_file()),None)
def close(roots,compiled=False):
 found={};todo=list(roots)
 while todo:
  p=todo.pop();m=str(p.relative_to(ROOT).with_suffix('')).replace('/','.')
  if m in found:continue
  direct=imports(p);obj=effective(m,'.olean');assert obj,m
  il=effective(m,'.ilean');actual=json.loads(il.read_text())['directImports'] if il else None
  found[m]={'module':m,'source':pin(p),'source_imports':direct,'effective_olean':pin(obj),'effective_olean_resolved_path':str(obj.resolve()),'effective_ilean':pin(il) if il else None,'compiled_direct_imports':actual,'status':'fresh_current_task' if p in lean else 'reused_dependency_object'}
  if compiled and actual:direct=direct+[x[0] for x in actual]
  todo.extend(source(x) for x in direct if source(x).is_file())
 return found
public=close([TASK/(n+'.lean') for n in ['Kernels','Contract','Correct','FPEquiv']]);all_source=close(lean);all_compiled=close(lean,True)
external=sorted({i for r in all_compiled.values() for i in r['source_imports']+([x[0] for x in r['compiled_direct_imports']] if r['compiled_direct_imports'] else []) if not source(i).is_file()})
write('import-closure.json',{'public_source_module_count':len(public),'all_task_source_module_count':len(all_source),'repository_source_plus_available_ilean_module_count':len(all_compiled),'available_ilean_count':sum(r['effective_ilean'] is not None for r in all_compiled.values()),'missing_reused_ilean_modules':[m for m,r in all_compiled.items() if r['effective_ilean'] is None],'public_source_modules':sorted(public),'all_task_source_modules':sorted(all_source),'records':[all_compiled[k] for k in sorted(all_compiled)],'external_direct_import_boundaries':external,'note':'Source imports and available compiled direct imports are pinned. Missing reused ilean files are disclosed. External toolchain/package caches are reused; their entire transitive object graph is not freshly rebuilt.'})
entry=json.loads((E/'entry.json').read_text());changed=[];unchanged=[]
with tarfile.open(E/'pre-revision-sources.tar.gz') as arc:
 members=arc.getnames()
 for row in entry['files']:
  p=ROOT/row['path']
  if sha(p)==row['sha256']:unchanged.append(pin(p))
  else:
   assert p.name in ('Correct.lean','FPEquiv.lean'),p
   member=next(n for n in members if n==row['path'] or n.endswith('/'+row['path']))
   old=arc.extractfile(member).read().decode();assert p.read_text().startswith(old.rstrip()+'\n'),p
   changed.append({'current':pin(p),'historical_sha256':row['sha256'],'historical_content_retained_as_prefix':True})
write('preservation.json',{'entry_files':len(entry['files']),'unchanged':unchanged,'additive_standard_entries':changed,'historical_theorem_bodies_preserved':True,'selected_python_unchanged':True,'guards_contract_unchanged':True,'core_lake_ledger_edited':False})
write('run-receipts.json',{'runs':[{'name':'initial Kernels rebuild','exit_code':0,'log':'initial-build.log'}, {'name':'42 unchanged task modules rebuild','exit_code':0,'session':6655,'log':'baseline-build.log'}, {'name':'KernelScale initial prelude build','exit_code':1,'log':'kernel-prelude-build.log','reason':'noncircularity audit named an FP module not imported by Host; changed audit avoidance to the actual Host.scale boundary'}, {'name':'3 new modules and 2 standard entry modules rebuild','exit_code':0,'session':32956,'log':'kernel-build.log'}, {'name':'printed statements, standard axioms, noncircular targets','exit_code':0,'session':33646,'log':'statement-audit.log'}, {'name':'actual source CPU contract checker plus launch-scale conversion table','exit_code':0,'session':73985,'log':'cpu-check.log','settings':{'seed':20261020,'torch_threads':1,'dont_write_bytecode':True},'report':'cpu-check.json'}, {'name':'10 actual wrapper/native-binder SM90 offline compiles','exit_code':0,'session':58515,'log':'native-probe.log','report':'native-probe.json'}, {'name':'static symbolic PTX topology audit','exit_code':0,'log':'native-static.log','report':'native-static.json','no_floating_numerical_experiment':True}], 'initial_sandbox_attempt':'failed to locate Lean application; reviewed workspace-only driver subsequently approved outside sandbox','GPU_execution':False})
delivery={'task':'context_attn_fwd','status':'implemented_pending_independent_review','task_directory':str(TASK),'public_entries':['Kernels.lean','Contract.lean','Correct.lean','FPEquiv.lean'],'proved_endpoints':['PublicKernel.shared_source','PublicKernel.correct_real','PublicKernel.correct_fp32','PublicKernel.input_tensor'],'scope':pin(TASK/'KernelScope.md'),'selected_source':pin(TASK/'context_attn_fwd_repaired.py'),'original_source':pin(ROOT/'bench/tritonbench_g/context_attn_fwd/context_attn_fwd.py'),'source_and_guards_unchanged':True,'historical_theorem_bodies_preserved':True,'source_defect_found':False,'task_Lean_count':len(lean),'new_Lean_modules':[str(TASK/'KernelOnly.lean'),str(TASK/'Proofs/KernelScale.lean'),str(TASK/'Proofs/KernelProfile.lean')],'changed_historical_sources':[str(TASK/'Correct.lean'),str(TASK/'FPEquiv.lean')],'task_sources':[pin(p) for p in lean],'build_driver':pin(ROOT/'.veritile-cache/build-tools/workspace_lean_local_first_v2.py'),'task_build_receipts':pin(E/'task-builds.json'),'repository_import_closure':pin(E/'import-closure.json'),'independent_targets':['KernelOnly.mathematical: initial-array finite-sentinel base-2 normalized weighted sum','KernelOnly.nativeTarget: fixed-profile evaluation of independent masked online recurrence'],'model_boundaries':['original Tensor/storage/metadata observations','exact incoming FP32 scalar ABI binding, separate from abstract HostArithmetic','fixed local scalar and K8 WGMMA instruction ports; no whole-result oracle','compiler realization and successful race-free launch delivery','SM90 BM128/D256 resource failure and existing launcher grid-product overflow restrict runtime applicability'],'validation':{'all_task_modules_rebuilt':48,'standard_base_axioms_only':True,'noncircular_targets':True,'actual_source_CPU_invocations':7,'actual_wrapper_native_binder_offline_profiles':10,'compiler_text_artifacts':40,'static_exact_reduction_trees_and_replicas':True,'static_PV_coordinate_producer_routing':True,'static_scale_FMA_rescale_division_operand_order':True,'static_clear_live_and_K8_coverage':True,'payload_and_literal_independent_selection_proved':True,'GPU_execution':False,'primitive_numerical_experiments':False,'native_Real_equality':False,'error_bound':False,'transitive_cold_rebuild':False},'integration':'Parent owns ledger/Lake integration and independent review; no self-acceptance.'}
# Manifest bundles all repo sources, receipts, scripts and text evidence; binary caches
# remain at their exact recorded effective locations, and are hash-pinned separately.
files={Path(r['source']['path']) for r in all_compiled.values()}|{TASK/'KernelScope.md',TASK/'context_attn_fwd_repaired.py',TASK/'check_contract.py',TASK/'README.md',ROOT/'bench/tritonbench_g/context_attn_fwd/context_attn_fwd.py'}
files|={p for p in E.rglob('*') if p.is_file() and p.name not in ('delivery.json','packet.tar.gz','freeze-result.json','verification.json')}
delivery['frozen_files']=[pin(p) for p in sorted(files)];write('delivery.json',delivery)
with tarfile.open(E/'packet.tar.gz','w:gz') as arc:
 for p in sorted(files|{E/'delivery.json'}):arc.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
write('freeze-result.json',{'delivery':pin(E/'delivery.json'),'archive':pin(E/'packet.tar.gz'),'frozen_files':len(files),'archive_members':len(files)+1,'task_builds':len(lean),'source_closure':len(all_source),'compiled_repository_closure':len(all_compiled)})
print((E/'freeze-result.json').read_text())
