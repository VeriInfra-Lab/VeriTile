import hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).parent;r=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');old=p.parent/'remaining-original-scope-audit';sha=lambda b:hashlib.sha256(b).hexdigest()
inv=json.loads((old/'inventory.json').read_text());cp=r/'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics/current-source-catalog.json'
base='fc183f5dae5e3f09f549b8151a9a1e4c366beb02';data=subprocess.check_output(['git','show',base+':bench/original_program_scope.json'],cwd=r);(p/'baseline-scope-ledger.json').write_bytes(data)
inv.update(reviewed_base_commit=base,reviewer='/root/scope_remaining_cohort',catalog_path=str(cp),catalog_sha256=sha(cp.read_bytes()),ledger_snapshot_sha256=sha(data));(p/'inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
src=(old/'finalize_batch.py').read_text().split('\nbatch=int(')[0];src=src.replace("'/root/full_scope_audit'","'/root/scope_remaining_cohort'")
src=src.replace("helpers={", "helpers={\n 'chunk_gla_simple':['Proofs/RealFlat.lean','Proofs/TypedExecution.lean'],\n 'softmax_triton3':['Proofs/Real.lean'],\n 'attention_fwd_triton1':['Proofs/RealIO.lean'],\n 'chunk_delta_fwd':['Proofs/RealIO.lean','Proofs/UndefinedTail.lean','Proofs/TypedExecution.lean'],")
src+='''
label=sys.argv[1];names=sys.argv[2:];results=[freeze(t) for t in names]
out={'schema':1,'kind':'original_scope_batch_review','batch':3,'part':label,'reviewer':'/root/scope_remaining_cohort','reviewed_base_commit':inv['reviewed_base_commit'],'catalog_sha256':inv['catalog_sha256'],'source_ledger_snapshot_sha256':inv['ledger_snapshot_sha256'],'task_count':len(results),'structural_gaps_confirmed':len(results),'whole_original_program_accepted':0,'completion_increment':0,'public_declarations_checked':sum(len(r['public_declarations_checked']) for r in results),'structural_gap_count':sum(len(r['structural_gaps']) for r in results),'catalog_declaration_metadata_discrepancies':[{'task':r['task'],**d} for r in results for d in r['catalog_declaration_metadata_discrepancies']],'integration_note':'Read-only source scope audit. Existing partial proofs remain credited. No numerical deferrals or original-program acceptance registered. Root substantive review is required before integration.','tasks':results}
p=root/f'batch-03-{label}-review.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\\n');print(json.dumps({'path':str(p),'sha256':sha(p),'tasks':len(results),'public_declarations':out['public_declarations_checked'],'gaps':out['structural_gap_count'],'metadata_discrepancies':len(out['catalog_declaration_metadata_discrepancies'])}))
''';(p/'freeze.py').write_text(src)
(p/'read_task.py').write_bytes((old/'read_task.py').read_bytes())
for t in ['chunk_gla_simple','softmax_triton3','attention_fwd_triton1','chunk_delta_fwd']:(p/'read-inputs'/f'{t}.json').write_bytes((old/'read-inputs'/f'{t}.json').read_bytes())
helper=(old/'notes_batch02_part1.py').read_text().split("\nt='chunk_cumsum_vector'")[0];(p/'note_helpers.py').write_text(helper)
