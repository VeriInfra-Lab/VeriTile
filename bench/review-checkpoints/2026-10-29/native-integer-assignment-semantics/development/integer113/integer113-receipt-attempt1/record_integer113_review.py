from pathlib import Path
import subprocess,json,hashlib,gzip,re
r=Path('/tmp/veritile-pr18-review');d=Path('/tmp/veritile-hgrn54-clean');c=Path('/tmp/veritile-body77')
parent=Path('/tmp/welfold112-receipt.txt').read_text().strip();code=Path('/tmp/integer113-code.txt').read_text().strip()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pin=lambda p:{'path':p.relative_to(r).as_posix(),'sha256':sha(p)}
for root in [r,d,c]:assert git(root,'rev-parse','HEAD')==code and not git(root,'status','--porcelain'),root
mods=Path('/tmp/integer113-all-modules.txt').read_text().split();assert len(mods)==5666
builds={k:json.loads(Path('/tmp/integer113-'+k+'.json').read_text()) for k in ['warm','clean','audit']}
for k,b in builds.items():assert b['status']=='PASS' and b['code_commit']==code,(k,b)
for k in ['warm','clean']:assert builds[k]['build_exit_code']==builds[k]['joint_exit_code']==0 and builds[k]['explicit_modules']==len(mods)
clean=builds['clean'];assert not clean['project_build_initially_present'] and not clean['project_objects_imported_from_other_checkouts'] and not clean['veritile_cache_initially_present']
jobs={}
for k in ['warm','clean']:
 s=Path('/tmp/integer113-'+k+'-build.log').read_text(errors='replace');match=re.search(r'Build completed successfully \((\d+) jobs\)',s);assert match and not re.search(r'^error:',s,re.M);jobs[k]=int(match.group(1))
 assert not Path('/tmp/integer113-'+k+'-joint.log').read_text().strip()
audit=Path('/tmp/integer113-audit.log').read_text(errors='replace');targets=json.loads(Path('/tmp/integer113-audit-targets.json').read_text());assert len(targets)==builds['audit']['theorems'] and audit.count('axiom footprint ⊆ standard base')==len(targets)
assert not any(s in audit for s in ['sorryAx','DISALLOWED','unresolved FP proof'])
assert not re.search(r'^(?:.*?:\d+:\d+: )?error:', audit, re.M)
assert all('axiom footprint ⊆ standard base' in line for line in audit.splitlines() if 'error:' in line)
assert 'FP assumptions used by row_offset_reused_equiv:\n  none' in audit
assert 'FP assumptions used by add_value_commuted_equiv:' in audit
assert 'integer-literals' in audit and 'IntegerLiteral' in audit
assert 'Ran 362 tests' in Path('/tmp/integer113-python-tests.log').read_text() and '\nOK\n' in Path('/tmp/integer113-python-tests.log').read_text()
s=Path('/tmp/integer113-all-python.log').read_text();assert 'Ran 607 tests' in s and 'FAILED (errors=1)' in s and 'bound source/hash changed: repo:VeriTile/Triton/Float/Structural.lean' in s
ins=json.loads(Path('/tmp/integer113-inspections.json').read_text());assert ins['code_commit']==code and ins['cpu']['status']=='PASS' and ins['torch']['status']=='UNAVAILABLE' and all(x['status']=='PASS' for x in ins['raw_bodies'])
comps=json.loads(Path('/tmp/integer113-comparator-invocations.json').read_text());assert len(comps)>=346 and all(x['code_commit']==code and x['status']=='UNAVAILABLE' and x['sha256']==sha(r/x['file']) for x in comps)
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
corpus=json.loads((r/'bench/source-correspondence.json').read_text());pairs=[p for row in corpus['tasks']+corpus['examples'] for p in row['checked_pairs']];assert len(pairs)==415 and all(p['mode']=='pasted_body' for p in pairs) and corpus['counts']['task_status']['selected_body_copy_checked']==118
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
regression='Ordinary integer literal assignments now create signed or unsigned 32/64-bit tensors at the same boundaries as the pinned Triton frontend; inline weak scalars remain distinct and out-of-range literals are rejected. The strict FP32 software adapter uses its actual integer rounding operation for integer-valued literals and has an explicitly revised integer-literals model identifier. Five selected task families and all cumulative task/example/frontend roots are rechecked. The AddValue proof uses a kernel-checked strict conversion certificate; LoRA and Fast RoPE follow their actual signed local and loop states. This is a model/source correction, not a new hardware guarantee or numerical experiment.'
notes={n:regression+(' Body migration remains pending; its previous source-correspondence status is unchanged.' if n in extra else '') for n in reviewed}
notes.update({'add_value': 'The original body retains y = 10 and the original cast/order. Its actual integer-to-FP32 conversion is connected to the existing commuted candidate using the strict software certificate and strict addition commutation. The public statement, input domain, complete memory output and frame remain unchanged; only its explanatory comment is updated.', 'lora_expand_gemv': 'The unit-split candidate retains split_n_length = 1 as an integer tensor. Actual execution and TraceSafe are proved for the existing public branch engine and ordered reduction plans, including option failures. Different private integer registers are forgotten only through the existing clear-register launch boundary; complete schedules, outputs and exterior memory frames are retained. No equality to the old natural-register AST or arbitrary floating-point interpretation is asserted.', 'fast_rope_embedding': 'The original and commuted bodies keep ROPE_GROUP_SIZE = 4. Their signed head indices, bounded rangeInt loop, original casts and masked loads/stores are proved directly. Shared independent head-value mathematics is reused with a new actual-source execution and safety connection. The actual program pair and full IO are retained; no false equality to the historical natural-register body remains.', 'fifth_order_sph_harmonics': 'The ordinary coord_stride = 3 assignments are re-elaborated as integer tensors in all four selected forward/backward bodies. Their complete source execution, safety, independent polynomial specification and existing FP transformation are replayed with the corrected shared frontend. Original Python, public readers and input domain are unchanged.', 'token_attn_reduceV': 'The original and affine candidate retain cur_batch_start_index = 0 as an integer tensor. Actual metadata gathers, loop accumulation, output stores, access safety and full selected grid interface are replayed with the corrected frontend. Original Python, public readers, independent mathematics and input domain are unchanged.'})
ev=r/'bench/review-checkpoints/2026-10-29/native-integer-assignment-semantics';assert not ev.exists();ev.mkdir(parents=True)
def save(src,dst):
 data=src.read_bytes();compressed=src.suffix=='.log' and len(data)>20000
 (dst.with_name(dst.name+'.gz') if compressed else dst).write_bytes(gzip.compress(data,mtime=0) if compressed else data)
(ev/'source-closure.json').write_text(json.dumps({'code_commit':code,'all_tracked':True,'local_lean_files':len(hashes),'committed_source_matches_independent_empty_project_build':True,'scope':'5666 cumulative roots and complete local import closure; only pinned external dependencies reused.','sha256':hashes},indent=2)+'\n')
(ev/'modules.json').write_text(json.dumps(mods,indent=2)+'\n');(ev/'pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
# Preserve both final receipts and the failed intermediate development runs.
for p in sorted(Path('/tmp').glob('integer113-*')):
 if p.is_file() and p.suffix in ['.json','.log','.lean','.txt'] and not p.name.endswith(('-receipt.txt','-receipt-driver.json')):save(p,ev/p.name.removeprefix('integer113-'))
for tag in ['integer113']:
 folder=ev/'development'/tag;folder.mkdir(parents=True)
 for p in sorted(Path('/tmp').glob(tag+'-*')):
  if p.is_file() and p.suffix in ['.json','.log','.lean','.txt']:save(p,folder/p.name.removeprefix(tag+'-'))
 for snapshot in sorted(Path('/tmp').glob(tag+'-*attempt*')):
  if snapshot.is_dir():
   for p in snapshot.rglob('*'):
    if p.is_file() and p.suffix in ['.json','.log','.lean','.py']:
     target=folder/snapshot.name/p.relative_to(snapshot);target.parent.mkdir(parents=True,exist_ok=True);save(p,target)
for x in comps:
 p=Path(x['log']);assert 'Official comparator isolation is unavailable' in p.read_text(errors='replace');save(p,ev/p.name.removeprefix('integer113-'))
phase=json.loads(Path('/tmp/integer113-stage-final.json').read_text());assert phase['status']=='PASS' and phase['explicit_modules']==346
assert phase['joint_exit_code']==phase['audit_exit_code']==0
for mod,h in phase['latest_source_sha256'].items():assert sha(r/(mod.replace('.','/')+'.lean'))==h,mod
focused={'compiled_modules':phase['compiled_modules']}
assert set(Path('/tmp/integer113-focused-modules.txt').read_text().split())<=set(mods)
(ev/'previous-reviews.json.gz').write_bytes(gzip.compress(json.dumps({'tasks':[rows[n] for n in reviewed],'ledger_tasks':[tasks[n] for n in reviewed],'io':io,'reader_roles':roles},ensure_ascii=False,indent=2).encode(),mtime=0))
review={'schema':1,'status':'verified_task_batch_corpus_migration_incomplete','code_commit':code,'merge_policy':'Keep draft; do not merge before explicit human review.','new_task_migrations':[],'reverified_integer_assignment_tasks':sorted(['add_value', 'lora_expand_gemv', 'fast_rope_embedding', 'fifth_order_sph_harmonics', 'token_attn_reduceV']),'regression_only_pending_body_tasks':sorted(extra),'semantic_review':notes,'compiler_change':{'files':core,'behavior':regression},'verification':{
 'focused':{'status':'PASS','explicit_modules':346,'compiled_modules':focused['compiled_modules'],'joint_import':'PASS','scope':'All 346 affected task/frontend roots and joint import pass; focused theorem definitions and private helpers are audited. Final 5666-root warm/clean replays cover every cumulative root with the same shared frontend.'},
 'warm_and_joint':{'status':'PASS','explicit_modules':len(mods),'jobs':jobs['warm']},
 'independent_clean_checkout':{'status':'PASS','explicit_modules':len(mods),'jobs':jobs['clean'],'project_build_initially_present':False,'project_objects_imported_from_other_checkouts':False,'only_fixed_external_dependencies_imported':True,'setup':pin(ev/'clean.json')},
 'axiom_audit':{'status':'PASS','defining_modules':builds['audit']['modules'],'elaborated_theorems':len(targets),'private_helpers_included':True,'allowed':['propext','Classical.choice','Quot.sound'],'method':'Theorem roots discovered by defining module; accumulated dependency union checked after every root in one immutable Lean environment.'},
 'fp_assumptions':'The AddValue reader reports the strict integer-literal conversion and strict addition-commutation certificates under the explicitly revised software model. Experimental-rule reporting is retained separately. Actual LoRA and Fast RoPE pairs preserve their existing ordered FP graphs; all cumulative theorem audit commands are replayed.',
 'python':ins['python'],
 'source':{n:notes[n] for n in ['add_value', 'lora_expand_gemv', 'fast_rope_embedding', 'fifth_order_sph_harmonics', 'token_attn_reduceV']},
 'official_comparator':{'status':'UNAVAILABLE','actual_final_source_attempts':len(comps),'reason':'Pinned official binaries require Landlock ABI v5; host supplies v4 and no systemd user isolation. No bypass.'},
 'torch':'UNAVAILABLE: retained checker requires missing Torch/Triton.',
 'cpu':ins['cpu'],'original_python_and_contracts':ins['original_python_and_contracts'],
 'gpu_or_numerical_experiment':'Not run.',
  'earlier_failures':'Failed frontend, integer-loop and local-state proof drafts are retained. Unavailable or failed checks do not count as passing.',
  'scope_limits':regression+' Public contracts, parameters and mathematical input domains remain unchanged. Full Python discovery retains one historical Structural numerical-receipt module import error; the old receipt is not repinned.'},
 'source_pairs':{'total':len(pairs),'pasted_body':len(pairs),'legacy_ast':0},'source_closure':pin(ev/'source-closure.json'),'corpus_complete':False,'corpus_counts':corpus['counts'],'example_coverage':{'total':19,'covered':19,'historical':2,'authored_counterpart':2,'spec_derived':15}}
review['evidence']=[pin(p) for p in sorted(ev.rglob('*')) if p.is_file()];(ev/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');rp=pin(ev/'review.json')
def elog(k):
 p=ev/(k+'.log');return pin(p if p.exists() else p.with_name(p.name+'.gz'))
bases={'add_value': ('bench/repairs/tritonbench_g/add_value/', 'bench/tritonbench_g/add_value/'), 'lora_expand_gemv': ('bench/repairs/tritonbench_g/lora_expand_gemv/', 'bench/tritonbench_g/lora_expand_gemv/'), 'fast_rope_embedding': ('bench/repairs/tritonbench_g/fast_rope_embedding/', 'bench/tritonbench_g/fast_rope_embedding/'), 'fifth_order_sph_harmonics': ('bench/repairs/tritonbench_g/fifth_order_sph_harmonics/', 'bench/tritonbench_g/fifth_order_sph_harmonics/'), 'token_attn_reduceV': ('bench/repairs/tritonbench_g/token_attn_reduceV_grid_y_checked/', 'bench/repairs/tritonbench_g/token_attn_reduceV/')}
bridge_names={'add_value': {'raw_bodies': 'Kernels.lean', 'actual_fp_conversion': 'Proofs/Rewrite.lean', 'source_scope': 'Proofs/SourcePreservation.lean', 'public_real': 'Correct.lean', 'public_pair': 'FPEquiv.lean'}, 'lora_expand_gemv': {'raw_bodies': 'ByteSeparated/Kernels.lean', 'candidate': 'ByteSeparated/CandidateKernels.lean', 'actual_execution': 'ByteSeparated/Proofs/IntegerCandidate.lean', 'trace_safety': 'ByteSeparated/Proofs/IntegerCandidateSafety.lean', 'clear_register_schedule': 'ByteSeparated/Proofs/LocalObservation.lean', 'complete_io_pair': 'ByteSeparated/Proofs/Pair.lean', 'public_pair': 'FPEquiv.lean'}, 'fast_rope_embedding': {'raw_bodies': 'Kernels.lean', 'actual_execution': 'Proofs/Source.lean', 'trace_safety': 'Proofs/AccessSafety.lean', 'actual_pair': 'Proofs/SourcePair.lean', 'public_real': 'Correct.lean', 'public_pair': 'FPEquiv.lean'}, 'fifth_order_sph_harmonics': {'raw_bodies': 'Kernels.lean', 'actual_execution': 'Proofs/SourceExecution.lean', 'memory_connection': 'Proofs/Execution.lean', 'trace_safety': 'Proofs/KernelSafety.lean', 'actual_pair': 'Proofs/SourcePair.lean', 'public_real': 'Correct.lean', 'public_pair': 'FPEquiv.lean'}, 'token_attn_reduceV': {'raw_bodies': 'Kernels.lean', 'source_binding': 'Proofs/Bindings.lean', 'trace_safety': 'Proofs/AccessSafety.lean', 'actual_pair': 'Proofs/AffineEquivalence.lean', 'public_real': 'Correct.lean', 'public_pair': 'FPEquiv.lean'}}
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
 row['verification']={'status':'passed_with_recorded_limit','scope':notes[n],'evidence_files':[elog('warm-build'),elog('clean-build')],'limits':['Official comparator and Torch unavailable; actual final-source attempts retained.','Pre-existing historical numerical-receipt import error; no repinning.','65 tasks pending; human review required before merge.']}
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
  row['assessment']={'organization':'Task-level IO retains the existing channels, layouts, strides, launch schedule and input domain. Actual source execution, safety and complete pair connections remain in Proofs.','source_update':notes[row['task']],'verification':regression,'previous_assessment':pin(ev/'previous-reviews.json.gz')}
  row['changed_paths']=dif;row['review_evidence']=rp;row['preservation_and_build_passed']=True;row['main_formula_review']['io_data_unchanged']=False;row['current_body_source_review']=rp
  row['validation_limits']={'official_comparator':'Unavailable; actual final-source attempts retained.','torch':'Unavailable.','numerical_receipt':'Pre-existing historical Structural receipt import error; the current integer-assignment/model correction is separately recorded.','scope':'Original mathematical input domain and public premises preserved; no new numerical experiment. Original per-task storage profiles and explicit optional native modeling boundaries remain.'}
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
l['jit_correspondence_review']={'date':'2026-10-29','review':rp,'inventory':pin(r/'bench/source-correspondence.json'),'pasted_body_entry_tasks':selected,'scope':'19/19 examples, 118/183 selected tasks migrated; 65 tasks remain. All 415 mappings use complete pasted bodies.','counts':corpus['counts'],'previous_review':l['jit_correspondence_review']};lp.write_text(json.dumps(l,ensure_ascii=False,indent=2)+'\n')
(ev/'README.md').write_text(f"""# 整数赋值语义及受影响程序重证

普通整数赋值按 Triton 的边界生成 int32、uint32、int64 或 uint64 张量，内联弱类型标量保持区别。严格 FP32 软件模型的整数常量使用实际整数舍入器，模型标识显式加入 integer-literals；AddValue 通过严格转换证书连接。LoRA 和 Fast RoPE 的实际有符号寄存器、循环、访存安全及完整 IO 重新证明，另外两组受影响题目完整重验。

346 个任务/前端入口及联合导入通过；开发阶段 {phase['axiom_theorems']} 个定理通过依赖审计。累计 5666 个入口及联合导入通过热构建和空项目缓存重建，{len(hashes)} 个本地 Lean 文件与 Git 精确一致；{builds['audit']['modules']} 个定义模块的 {len(targets)} 个定理通过标准公理依赖审计。

362 项针对性 Python 测试通过；全量 607 项仍有一条历史 Structural 数值凭据导致的模块导入错误，未重写旧凭据。{len(comps)} 次官方 comparator 因主机隔离不可用；Torch/Triton 缺失；未运行 GPU 或新数值实验。保留失败诊断，不计为通过。

本批修正既有覆盖项，不增加迁移计数：tasks 118/183、examples 19/19、415 个函数体映射，65 个任务待迁移。公开语句、前提和输入域保留；AddValue 仅修正文档注释。继续保持草稿，等待人工审查后再考虑合入。
""")
print('Recorded integer113',code,len(hashes),'Git-exact files',len(targets),'theorems',len(reviewed),'review rows',flush=True)
