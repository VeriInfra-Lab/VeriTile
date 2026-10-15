from pathlib import Path
import datetime,hashlib,json
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=W/'evidence/receipt.json';assert not receipt.exists(),'Package is already frozen.'
compiles=json.loads((W/'evidence/final-compiles.json').read_text());assert len(compiles['modules'])==9
for m in compiles['modules']:
 assert sha(W/m['source'])==m['source_sha256']
 assert sha(W/m['receipt'])==m['receipt_sha256']
 assert sha(W/m['log'])==m['log_sha256']
for dep in json.loads((W/'evidence/repo-dependencies.json').read_text())['project_import_closure']:
 p=(R if dep['origin']=='repository' else W/'after')/dep['path'];assert sha(p)==dep['source_sha256'],str(p)
files={}
for p in sorted(W.rglob('*')):
 if p.is_symlink() or not p.is_file():continue
 rel=p.relative_to(W)
 if rel.parts[0]=='build-overlay' or '__pycache__' in rel.parts or p.suffix in ['.olean','.ilean','.ir'] or p.name in ['task.md','receipt.json','receipt.sha256']:continue
 files[str(rel)]=sha(p)
result={'schema':1,'task':'softmax_flaggems','kind':'immutable_additive_host_grid_review_package','status':'frozen_pending_independent_root_review','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':'11f0c747af151465286ea9e8821932f80b71513fbb2d343a304ba54f4e8013f4','repository_and_git_modified':False,'previous_matrix_package_modified':False,'acceptance_increment':0,'native_functional_correctness':'not_established','numeric_assumptions_newly_registered':[],'final_additive_modules':9,'named_axiom_checks':compiles['named_axiom_checks'],'noncircular_checks':compiles['noncircular_checks'],'source_probes':{'selected_host_cases':960,'heuristic_cases':352,'boundary_cases':7,'frontend_profiles':80,'frontend_rejections':10,'autotune_routes':252,'autotune_rejections':28,'alias_cases':24,'layout_cases':24,'noncanonical_layout_outputs':3},'primary_public_specification':'VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid.softmax_autograd_source_admitted_real','public_file':'after/bench/tritonbench_g/softmax_flaggems/HostCorrect.lean','remaining_scope':'remaining-scope.json','frozen_files':files,'excluded_generated_build_artifacts':['build-overlay/**','*.olean','*.ilean','*.ir','__pycache__/**'],'mutable_instruction_file_not_part_of_receipt':'task.md'}
receipt.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');digest=sha(receipt);(W/'evidence/receipt.sha256').write_text(digest+'  receipt.json\n');print(json.dumps({'receipt_sha256':digest,'frozen_files':len(files),'status':result['status']}))
