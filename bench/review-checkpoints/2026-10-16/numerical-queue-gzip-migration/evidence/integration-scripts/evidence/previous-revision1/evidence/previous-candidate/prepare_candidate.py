from pathlib import Path
import json,hashlib,ast
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');P=W.parent/'queue-gzip-migration-wip';I=W.parent/'queue-gzip-independent-review'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'receipt.json')=='2b5c5d20c5113565e82f28422c6300a4debd5705b798f196cac70fd7b2e578ba'
assert sha(I/'receipt.json')=='f7c52e0e79159a613dbb7ef6249ca118ccee454de99b14438e0a96a43593f855'
for package in [P,I]:
 r=json.loads((package/'receipt.json').read_text())
 for rel,h in r['artifacts'].items():assert sha(package/rel)==h,rel
inputs=json.loads((P/'input-sha256.json').read_text());changes={}
for rel,h in inputs.items():assert sha(R/rel)==h
for rel in ['bench/deferred_numerical_experiments.py','bench/check_original_program_scope.py','bench/tests/test_numerical_queue_storage.py']:
 text=(P/'after'/rel).read_text()
 if rel.endswith('deferred_numerical_experiments.py'):
  text=text.replace('if legacy.exists():','if legacy.exists() or legacy.is_symlink():')
 elif rel.endswith('check_original_program_scope.py'):
  old="    queue_ref = ledger.get('deferred_queue') or {}\n    if queue_ref.get('path') != QUEUE:\n        raise ValueError('scope ledger must bind the unified numerical queue')\n    queue = load(checked(root, QUEUE, queue_ref.get('sha256')))"
  assert text.count(old)==1
  text=text.replace(old,"    queue = load_active_queue(root, ledger.get('deferred_queue') or {})")
  pos=text.index('\ndef main():')
  helper='''\ndef load_active_queue(root, queue_ref):
    """The standalone scope gate must also reject a competing plain queue."""
    root = Path(root)
    plain = root / (BASE + 'deferred_numerical_experiments.json')
    if plain.exists() or plain.is_symlink():
        raise ValueError('Competing plain numerical queue; only .json.gz may be active')
    if queue_ref.get('path') != QUEUE:
        raise ValueError('scope ledger must bind the unified numerical queue')
    return load(checked(root, QUEUE, queue_ref.get('sha256')))

'''
  text=text[:pos]+helper+text[pos:]
 elif rel.endswith('test_numerical_queue_storage.py'):
  text=text.replace('import deferred_numerical_experiments as queue','import sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nimport deferred_numerical_experiments as queue')
 out=W/'after'/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text);changes[rel]={'before':inputs.get(rel),'after':sha(out)}
extra='bench/tests/test_numerical_queue_migration.py';changes[extra]={'before':None,'after':sha(W/'after'/extra)}
# The current development target is the approved three-version policy. Legacy
# wider-domain evidence and exact acceptance rows keep their previous meaning.
policy='review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/primary-version-policy.md'
mapping='review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/three-version-inventory/eight-task-mapping.md'
for rel in ['bench/current_source_acceptance.md','bench/original_program_scope.md']:
 text=(R/rel).read_text();updated=text.replace('[plain numerical queue]','[compressed numerical queue]').replace('deferred_numerical_experiments.json)', 'deferred_numerical_experiments.json.gz)').replace('[`deferred_numerical_experiments.json`]','[`deferred_numerical_experiments.json.gz`]')
 if rel.endswith('current_source_acceptance.md'):
  old='The requested scope is **183 original programs, covering all declared stages,\ndtypes and branches**. **8 source scopes are accepted and 175 remain pending.**'
  new=f'The current goal is **183 original programs with Real, FP32 and BF16 as the\nthree primary versions, plus source-required types**, under the approved\n[primary-version policy]({policy}). Each selected version retains its actual\nstages, branches, layouts, memory behavior and failure paths. The\n[eight-task evidence map]({mapping}) records reusable proofs and remaining\nversion-specific gaps.\n\nThe **legacy full-source scope ledger records 8 accepted and 175 pending**.\nThese counts retain the earlier wider-domain work; they do not mean that eight\ntasks have completed all three versions. This storage migration assigns no new\nversion-completion counts.'
  assert old in updated;updated=updated.replace(old,new)
 else:
  old='The requested standard covers **all declared stages, dtypes and branches of\n183 original programs**. `fp4_to_bf16_conversion` is excluded because its existing\ncontract covers a repaired candidate; the separate `fp4_to_bf16` original task\nremains in the requested set.'
  new=f'The current development target is **Real, FP32 and BF16 for 183 original\nprograms, plus source-required types**, as specified by the approved\n[primary-version policy]({policy}). The\n[eight-task evidence map]({mapping}) preserves existing proofs and records the\nwork still needed for each selected version. No new three-version completion\ncount is assigned by the queue storage migration.\n\nThis page describes the retained **legacy full-source scope checker**, whose\ncontract covers all declared stages, dtypes and branches. Its earlier wider\nproofs remain reusable evidence; its counts do not establish completion of the\nthree selected versions. `fp4_to_bf16_conversion` remains the separate repaired\ncandidate, and the `fp4_to_bf16` original task remains in the requested set.'
  assert old in updated;updated=updated.replace(old,new).replace('records **8 source scopes accepted and 175 pending**.','records **8 legacy full-source scopes accepted and 175 pending**.')
 updated+=f'\nThe [pre-migration summary](@MIGRATION_CHECKPOINT@/before/{rel}) is preserved byte-for-byte with the gzip migration evidence.\n'
 assert updated!=text;out=W/'after'/rel;out.write_text(updated);changes[rel]={'before':sha(R/rel),'after':sha(out)}
pins={p:sha(R/p) for p in ['.github/workflows/bench-audit.yml','bench/current_source_audit.py','bench/adapter_common.py','bench/exact_numerical_bindings.py','bench/exact_scope_gate.py','bench/current_source_catalog.py','bench/current_value_metadata.py']}
policy_inputs={rel:sha(R/rel) for rel in ['bench/'+policy,'bench/'+mapping]}
assert policy_inputs['bench/'+policy]=='10a59ed41d81aee0e5bb82b81a998116564a8dd7c378516a0a4edf894b07be85'
(W/'evidence/source-plan.json').write_text(json.dumps({'changes':changes,'unchanged_consumers':pins,'policy_inputs':policy_inputs,'original_candidate_receipt_sha256':sha(P/'receipt.json'),'independent_receipt_sha256':sha(I/'receipt.json'),'live_repository_modified':False,'expected_base_commit':None},indent=2)+'\n')
print(json.dumps({'candidate_changes':len(changes),'unchanged_consumers':len(pins),'base_commit':'must_be_explicit_after_current_batch_commit'}))
