from pathlib import Path
import subprocess,json,hashlib,gzip,re
r=Path('/tmp/veritile-pr18-review');d=Path('/tmp/veritile-hgrn54-clean');c=Path('/tmp/veritile-body77')
parent=Path('/tmp/matrixsampling110-receipt.txt').read_text().strip();code=Path('/tmp/persistent111-code.txt').read_text().strip()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pin=lambda p:{'path':p.relative_to(r).as_posix(),'sha256':sha(p)}
for root in [r,d,c]:assert git(root,'rev-parse','HEAD')==code and not git(root,'status','--porcelain'),root
mods=Path('/tmp/persistent111-all-modules.txt').read_text().split();assert len(mods)==5572
builds={k:json.loads(Path('/tmp/persistent111-'+k+'.json').read_text()) for k in ['warm','clean','audit']}
for k,b in builds.items():assert b['status']=='PASS' and b['code_commit']==code,(k,b)
for k in ['warm','clean']:assert builds[k]['build_exit_code']==builds[k]['joint_exit_code']==0 and builds[k]['explicit_modules']==len(mods)
clean=builds['clean'];assert not clean['project_build_initially_present'] and not clean['project_objects_imported_from_other_checkouts'] and not clean['veritile_cache_initially_present']
jobs={}
for k in ['warm','clean']:
 s=Path('/tmp/persistent111-'+k+'-build.log').read_text(errors='replace');match=re.search(r'Build completed successfully \((\d+) jobs\)',s);assert match and not re.search(r'^error:',s,re.M);jobs[k]=int(match.group(1))
 assert not Path('/tmp/persistent111-'+k+'-joint.log').read_text().strip()
audit=Path('/tmp/persistent111-audit.log').read_text(errors='replace');targets=json.loads(Path('/tmp/persistent111-audit-targets.json').read_text());assert len(targets)==builds['audit']['theorems'] and audit.count('axiom footprint ⊆ standard base')==len(targets)
assert not any(s in audit for s in ['sorryAx','DISALLOWED','unresolved FP proof'])
assert not re.search(r'^(?:.*?:\d+:\d+: )?error:', audit, re.M)
assert all('axiom footprint ⊆ standard base' in line for line in audit.splitlines() if 'error:' in line)
assert 'FP assumptions used by source_pair_equiv:\n  none' in audit
assert 'FP assumptions used by static_unroll_equiv:\n  none' in audit
assert 'Ran 347 tests' in Path('/tmp/persistent111-python-tests.log').read_text() and '\nOK\n' in Path('/tmp/persistent111-python-tests.log').read_text()
s=Path('/tmp/persistent111-all-python.log').read_text();assert 'Ran 592 tests' in s and 'FAILED (errors=1)' in s and 'bound source/hash changed: repo:VeriTile/Triton/Float/Structural.lean' in s
ins=json.loads(Path('/tmp/persistent111-inspections.json').read_text());assert ins['code_commit']==code and ins['cpu']['status']=='PASS' and ins['torch']['status']=='UNAVAILABLE' and all(x['status']=='PASS' for x in ins['raw_bodies'])
comps=json.loads(Path('/tmp/persistent111-comparator-invocations.json').read_text());assert len(comps)>=50 and all(x['code_commit']==code and x['status']=='UNAVAILABLE' and x['sha256']==sha(r/x['file']) for x in comps)
changed=set(git(r,'diff','--name-only',parent,code).splitlines());core=sorted(x for x in changed if x.startswith('VeriTile/') and x.endswith('.lean'))
seen=set();todo=[m.replace('.','/')+'.lean' for m in mods]
while todo:
 rel=todo.pop()
 if rel in seen or not rel.startswith(('VeriTile/','bench/')):continue
 p=c/rel;assert p.is_file(),rel;seen.add(rel)
 for line in re.findall(r'^import\s+([^\n]+)',p.read_text(),re.M):todo.extend(m.replace('.','/')+'.lean' for m in line.split() if re.fullmatch(r'[\w.]+',m))
assert seen<=set(git(r,'ls-tree','-r','--name-only',code).splitlines());hashes={}
# Check Git objects in one batch, with exact byte lengths; no project artifact reuse.
paths=sorted(seen);blobs=subprocess.check_output(['git','cat-file','--batch'],cwd=r,input=''.join(code+':'+p+'\n' for p in paths).encode());pos=0
for rel in paths:
 end=blobs.index(b'\n',pos);header=blobs[pos:end].split();assert header[1]==b'blob';size=int(header[2]);data=blobs[end+1:end+1+size];pos=end+size+2
 h=hashlib.sha256(data).hexdigest();assert sha(r/rel)==sha(d/rel)==sha(c/rel)==h,rel;hashes[rel]=h
assert pos==len(blobs)
corpus=json.loads((r/'bench/source-correspondence.json').read_text());pairs=[p for row in corpus['tasks']+corpus['examples'] for p in row['checked_pairs']];assert len(pairs)==413 and all(p['mode']=='pasted_body' for p in pairs) and corpus['counts']['task_status']['selected_body_copy_checked']==117
assert 'FP assumptions used by index_commuted_equiv:\n  none' in audit
import sys
sys.path.insert(0,str(r/'bench'))
from check_main_entry_roles import declarations
ap=r/'bench/review-checkpoints/2026-10-23/reference-separation/main-formulas.json'
lp=r/'bench/tritonbench_g/task_progress.json'
ip=r/'bench/review-checkpoints/2026-10-23/reference-separation/io-audit.json'
rolep=r/'bench/review-checkpoints/2026-10-23/reference-separation/reader-roles.json'
a=json.loads(ap.read_text());l=json.loads(lp.read_text());io=json.loads(ip.read_text());roles=json.loads(rolep.read_text())
rows={x['task']:x for x in a['tasks']};tasks={x['task']:x for x in l['tasks']}
selected=sorted(x['name'] for x in corpus['tasks'] if x['status']=='selected_body_copy_checked')
impacted={n for n,row in rows.items() if changed & set(row['source_sha256'])}
extra=impacted-set(selected);reviewed=sorted(set(selected)|extra)
regression='The shared frontend widens Boolean values to zero or one when combined with signed integer operands, preserving the original quotient-plus-remainder-test syntax. A symbolic integer regression verifies both operand orders, subtraction, multiplication and an inline predicate. This is integer arithmetic, not an FP numerical identity. Every preceding task/example/frontend root is replayed with this final frontend in warm and independent empty-project-clean builds and joint imports.'
notes={n:regression+(' Body migration remains pending; its previous source-correspondence status is unchanged.' if n in extra else '') for n in reviewed}
notes['matmul_persistent_triton']='The complete original and existing row-product-commuted Python bodies have four verbatim FP16/E4M3 mappings. All 16 top-level statements remain: int64 casts, simultaneous metadata assignments, Boolean ceiling arithmetic, persistent signed tile/K loops, clamped pointers, live FP32 dot and output-dtype branches. Full Python module AST restoration permits exactly the row-index product operand commutation. Original Python, public Real/FP readers and contracts remain unchanged. Exact signed prefix arithmetic derives all counts from source; complete grid execution, trace safety, outputs and exterior frames reconnect to independent full-K mathematics and the actual FP pair. Historical false AST preservation is removed and raw execution is proved directly. No new numerical assumption or public domain restriction is introduced.'
ev=r/'bench/review-checkpoints/2026-10-29/native-persistent-matmul-function-bodies';assert not ev.exists();ev.mkdir(parents=True)
def save(src,dst):
 data=src.read_bytes();compressed=src.suffix=='.log' and len(data)>20000
 (dst.with_name(dst.name+'.gz') if compressed else dst).write_bytes(gzip.compress(data,mtime=0) if compressed else data)
(ev/'source-closure.json').write_text(json.dumps({'code_commit':code,'all_tracked':True,'local_lean_files':len(hashes),'committed_source_matches_independent_empty_project_build':True,'scope':'5572 cumulative roots and complete local import closure; only pinned external dependencies reused.','sha256':hashes},indent=2)+'\n')
(ev/'modules.json').write_text(json.dumps(mods,indent=2)+'\n');(ev/'pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
# Preserve both final receipts and the failed intermediate development runs.
for p in sorted(Path('/tmp').glob('persistent111-*')):
 if p.is_file() and p.suffix in ['.json','.log','.lean','.txt'] and not p.name.endswith(('-receipt.txt','-receipt-driver.json')):save(p,ev/p.name.removeprefix('persistent111-'))
for tag in ['persistent111']:
 folder=ev/'development'/tag;folder.mkdir(parents=True)
 for p in sorted(Path('/tmp').glob(tag+'-*')):
  if p.is_file() and p.suffix in ['.json','.log','.lean','.txt']:save(p,folder/p.name.removeprefix(tag+'-'))
 for snapshot in sorted(Path('/tmp').glob(tag+'-*attempt*')):
  if snapshot.is_dir():
   for p in snapshot.rglob('*'):
    if p.is_file() and p.suffix in ['.json','.log','.lean','.py']:
     target=folder/snapshot.name/p.relative_to(snapshot);target.parent.mkdir(parents=True,exist_ok=True);save(p,target)
for x in comps:
 p=Path(x['log']);assert 'Official comparator isolation is unavailable' in p.read_text(errors='replace');save(p,ev/p.name.removeprefix('persistent111-'))
phase=json.loads(Path('/tmp/persistent111-stage-final.json').read_text());assert phase['status']=='PASS' and phase['explicit_modules']==48
assert phase['joint_exit_code']==phase['audit_exit_code']==0
for mod,h in phase['latest_source_sha256'].items():assert sha(r/(mod.replace('.','/')+'.lean'))==h,mod
focused={'compiled_modules':phase['compiled_modules']}
assert set(Path('/tmp/persistent111-focused-modules.txt').read_text().split())<=set(mods)
(ev/'previous-reviews.json.gz').write_bytes(gzip.compress(json.dumps({'tasks':[rows[n] for n in reviewed],'ledger_tasks':[tasks[n] for n in reviewed],'io':io,'reader_roles':roles},ensure_ascii=False,indent=2).encode(),mtime=0))
review={'schema':1,'status':'verified_task_batch_corpus_migration_incomplete','code_commit':code,'merge_policy':'Keep draft; do not merge before explicit human review.','new_task_migrations':['matmul_persistent_triton'],'regression_only_pending_body_tasks':sorted(extra),'semantic_review':notes,'compiler_change':{'files':core,'behavior':regression},'verification':{
 'focused':{'status':'PASS','explicit_modules':48,'compiled_modules':focused['compiled_modules'],'joint_import':'PASS','scope':'All 48 focused roots plus joint import and 716 theorem footprints pass. The 50 comparator targets also include both changed frontend files. Final frozen 5572-root warm/clean replays cover every cumulative root with the same frontend.'},
 'warm_and_joint':{'status':'PASS','explicit_modules':len(mods),'jobs':jobs['warm']},
 'independent_clean_checkout':{'status':'PASS','explicit_modules':len(mods),'jobs':jobs['clean'],'project_build_initially_present':False,'project_objects_imported_from_other_checkouts':False,'only_fixed_external_dependencies_imported':True,'setup':pin(ev/'clean.json')},
 'axiom_audit':{'status':'PASS','defining_modules':builds['audit']['modules'],'elaborated_theorems':len(targets),'private_helpers_included':True,'allowed':['propext','Classical.choice','Quot.sound'],'method':'Theorem roots discovered by defining module; accumulated dependency union checked after every root in one immutable Lean environment.'},
 'fp_assumptions':'Persistent source_pair_equiv prints none; its structural integer rewrite preserves the complete FP arithmetic graph. All cumulative theorem audit commands are replayed.',
 'python':ins['python'],
 'source':{n:notes[n] for n in ['matmul_persistent_triton']},
 'official_comparator':{'status':'UNAVAILABLE','actual_final_source_attempts':len(comps),'reason':'Pinned official binaries require Landlock ABI v5; host supplies v4 and no systemd user isolation. No bypass.'},
 'torch':'UNAVAILABLE: retained checker requires missing Torch/Triton.',
 'cpu':ins['cpu'],'original_python_and_contracts':ins['original_python_and_contracts'],
 'gpu_or_numerical_experiment':'Not run.',
  'earlier_failures':'Failed parser/proof drafts are retained. Fixes cover Boolean signed promotion, exact prefix register projections, signed loop simulation, output dtype trace safety and final IO frames. Failed or unavailable attempts do not count as passing.',
  'scope_limits':'Original FP16/E4M3 profiles, complete persistent grid, zero axes, tails, strides and public contracts remain. Optional native instruction and Tensor results retain explicit model boundaries. The full Python suite retains one pre-existing Structural numerical-receipt import error; no false repinning.'},
 'source_pairs':{'total':len(pairs),'pasted_body':len(pairs),'legacy_ast':0},'source_closure':pin(ev/'source-closure.json'),'corpus_complete':False,'corpus_counts':corpus['counts'],'example_coverage':{'total':19,'covered':19,'historical':2,'authored_counterpart':2,'spec_derived':15}}
review['evidence']=[pin(p) for p in sorted(ev.rglob('*')) if p.is_file()];(ev/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');rp=pin(ev/'review.json')
def elog(k):
 p=ev/(k+'.log');return pin(p if p.exists() else p.with_name(p.name+'.gz'))
bases={'matmul_persistent_triton': ('bench/repairs/tritonbench_g/matmul_persistent_triton_scalar_cast/', 'bench/tritonbench_g/matmul_persistent_triton/')}
bridge_names={'matmul_persistent_triton': {'raw_bodies': 'Kernels.lean', 'exact_prefix': 'Prelude.lean', 'signed_loops': 'Loop.lean', 'complete_execution': 'Execution.lean', 'trace_safety': 'Proofs/SafetyLoops.lean', 'complete_io': 'Proofs/PublicResult.lean', 'independent_real': 'Proofs/LogicalIO.lean', 'actual_fp_pair': 'Proofs/Pair.lean', 'pair_io_frames': 'Proofs/PairIO.lean'}}
for n,bridges in bridge_names.items():assert all((r/(bases[n][0]+rel)).is_file() for rel in bridges.values())
for n in reviewed:
 row=rows[n];rev=tasks[n]['kernel_revision']
 roots={str(Path(p).parent).split('/Proofs')[0].split('/Misc')[0]+'/' for p in row['source_sha256'] if p.startswith('bench/') and Path(p).name in ['Kernels.lean','Correct.lean','FPEquiv.lean','Contract.lean']}
 for rel in set(row['source_sha256'])|set(core)|{p for p in changed if p.endswith('.lean') and any(p.startswith(root) for root in roots)}:
  if rel in changed:row['source_sha256'][rel]=sha(r/rel)
 if n in bases:
  base,shared=bases[n]
  bridges={k:base+v for k,v in bridge_names[n].items()}
  for rel in changed:
   if rel.startswith((base,shared)):row['source_sha256'][rel]=sha(r/rel)
  row['canonical_connection']={'historical_connection':row['canonical_connection'],'current_actual_source_bridges':{k:pin(r/f) for k,f in bridges.items()},'source_migration':notes[n],'preserved_scope':notes[n]}
 row['canonical_connection']['current_body_source_review']=notes[n];row['review_evidence']=rp
 row['verification']={'status':'passed_with_recorded_limit','scope':notes[n],'evidence_files':[elog('warm-build'),elog('clean-build')],'limits':['Official comparator and Torch unavailable; actual final-source attempts retained.','Known unchanged numerical receipt hash failure.','66 tasks pending; human review required before merge.']}
 for reader in row['readers']:
  p=r/reader['path']
  if sha(p)!=reader['sha256']:
   assert n in bases
   reader['sha256']=sha(p);reader['specifications']=[d for d in declarations(p) if d['kind']=='specification']
 rev['main_formula_review']['review_evidence']=rp;rev['main_formula_review']['verification_status']='passed_with_recorded_limit'
 rev['validation']['current_source_replay']={'review':rp,'code_commit':code,'integration':elog('warm-build'),'independent_clean_checkout':elog('clean-build'),'source_body_mode':'pasted_body' if n in selected else next(x['status'] for x in corpus['tasks'] if x['name']==n),'notes':notes[n]}
 for rel in rev.get('accepted_sha256',{}):
  if rel in changed:rev['accepted_sha256'][rel]=sha(r/rel)
 if n in bases:rev['jit_correspondence']={'status':'selected_body_copy_checked','inventory':'bench/source-correspondence.json','review_evidence':rp,'scope':notes[n]}
for row in io['tasks']:
 dif=[]
 for item in row['readers']+row['io_sources']:
  if item['path'] in changed and item['sha256']!=sha(r/item['path']):dif.append(item['path']);item['sha256']=sha(r/item['path'])
 if dif:
  assert row['task'] in bases
  row['assessment']={'organization':'Task-level IO retains the exact existing channels, logical matrices, strides and complete persistent schedules. Signed execution, access safety, independent mathematical recurrence and source-pair proofs live in Proofs.','source_update':notes[row['task']],'verification':regression,'previous_assessment':pin(ev/'previous-reviews.json.gz')}
  row['changed_paths']=dif;row['review_evidence']=rp;row['preservation_and_build_passed']=True;row['main_formula_review']['io_data_unchanged']=False;row['current_body_source_review']=rp
  row['validation_limits']={'official_comparator':'Unavailable; actual final-source attempts retained.','torch':'Unavailable.','numerical_receipt':'Known unchanged Structural receipt hash error.','scope':'Original mathematical input domain and public premises preserved; no new numerical experiment. Original persistent FP16/E4M3 storage profiles with explicit optional native modeling boundaries.'}
for row in roles['tasks']:
 if row['task'] not in bases:continue
 for reader in row['readers']:
  p=r/reader['path'];old={d['name']:d for d in reader['declarations']};decls=[]
  for decl in declarations(p):
   assert decl['name'] in old,decl
   item=dict(old[decl['name']]);item.update(decl);decls.append(item)
  reader.update(sha256=sha(p),declarations=decls)
 row['review_evidence']=rp;row['current_body_source_review']=notes[row['task']]
headlines=sum(d['kind']=='specification' for row in roles['tasks'] for pin_ in row['readers'] for d in pin_['declarations']);assert headlines==450
ap.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');ip.write_text(json.dumps(io,ensure_ascii=False,indent=2)+'\n');rolep.write_text(json.dumps(roles,ensure_ascii=False,indent=2)+'\n')
l['main_formula_review']['audit_sha256']=sha(ap);l['io_organization_review']['audit_sha256']=sha(ip);l['main_entry_role_review']['audit_sha256']=sha(rolep)
l['jit_correspondence_review']={'date':'2026-10-29','review':rp,'inventory':pin(r/'bench/source-correspondence.json'),'pasted_body_entry_tasks':selected,'scope':'19/19 examples, 117/183 selected tasks migrated; 66 tasks remain. All 413 mappings use complete pasted bodies.','counts':corpus['counts'],'previous_review':l['jit_correspondence_review']};lp.write_text(json.dumps(l,ensure_ascii=False,indent=2)+'\n')
(ev/'README.md').write_text(f"""# Persistent matmul 原始函数体

四个 FP16/E4M3 原始/候选函数体保留全部 cast、元组赋值、Boolean ceil、persistent tile/K 循环、clamped 指针、live FP32 dot 和输出 dtype 分支。候选只交换 row-index 乘法操作数，完整 Python 模块 AST 已检查；原 Python、公开 Real/FP 规格和契约不变。

48 个任务/回归入口及联合导入、716 个定理的开发检查通过。实际执行、安全、全部 grid 输出和其他内存保持不变重新连接到独立矩阵乘法规格及完整 FP 源程序对。累计 5572 个模块与联合导入通过热构建和空项目缓存重建；{len(hashes)} 个本地 Lean 源文件与 Git 一致；{builds['audit']['modules']} 个模块中的 {len(targets)} 个定理依赖通过标准公理审计。

347 项针对性 Python 测试通过。全量 592 项运行有一条旧 Structural 数值凭据导入错误，未虚假更新 hash。{len(comps)} 次官方 comparator 因主机隔离不可用；Torch/Triton 缺失；未运行 GPU 或新数值实验。失败草稿及日志保留。

累计 tasks 117/183、examples 19/19、413 个原始函数体对应项，66 个任务待迁移。保持草稿，等待人工审查。
""")
print('Recorded Persistent111',code,len(hashes),'Git-exact files',len(targets),'theorems',len(reviewed),'review rows',flush=True)
