"""Freeze only after complete Lean, dependency and staged whitespace checks."""
from datetime import datetime, timezone
import hashlib, json, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
previous=root.parent/'vector-addition-mixed-fp32-revision1-wip'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
assert not (root/'evidence/receipt.json').exists(), 'never overwrite a frozen receipt'
subprocess.run([sys.executable,str(root/'freeze.py'),'after'],check=True)
subprocess.run([sys.executable,str(root/'freeze.py'),'whitespace'],check=True)
prior=json.loads((previous/'evidence/scope-review.json').read_text())
assert sha(Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')/'bench/tritonbench_g/vector_addition/vector_addition.py')==prior['source_sha256']
profiles=json.loads((root/'evidence/source-profile/receipt.json').read_text())
selected={(r['x'],r['y']) for r in profiles['profiles']}
assert len(selected)==36
rows=prior['dtype_pairs']
assert sum((r['x'],r['y']) in selected for r in rows)==36
for r in rows:
    if (r['x'],r['y']) in selected:
        r['coverage']='new_mixed_fp16_bf16_concrete_software_value_and_ordered_grid_proved'
        r['remaining']=['native scalar numerical binding','native memory/address/error/masked-poison/scheduling refinement']
counts={
 'canonical_signatures':17,'canonical_pairs':289,'semantically_promoted_pairs':217,
 'observed_promotion_errors':72,'prior_floating_logical_pairs':25,
 'prior_integer_bool_logical_pairs':81,'prior_mixed_fp32_logical_pairs':18,
 'new_mixed_fp16_bf16_logical_pairs':36,'cumulative_logical_value_and_grid_pairs':160,
 'promoted_pairs_without_logical_value_execution':57,'remaining_mixed_fp64_pairs':18,
 'remaining_supported_frontend_fp64_float_pairs':11,
 'remaining_promoted_fnuz_frontend_rejections':28,
 'new_frontend_compilations':0,'native_backend_compilations':0,'gpu_executions':0,
 'whole_original_program_completion_increment':0}
public={
 'module':'bench.tritonbench_g.vector_addition.MixedSmallValueCorrect',
 'declarations':['vector_addition_mixed_small_total_correctness','vector_addition_mixed_small_defined_correctness'],
 'specification_commands':2,'bit_interpreter_closed_claim':False,
 'software_primitives':['Binary16.round directly from exact integer rational','Binary16.add',
  'FP.round .fp32 and FP.add for bf16 mixed profiles','exact BinaryConversion.widen for fp16→fp32',
  'bf16 field embedding into fp32','FP.cast .bf16 cfg for bf16 output',
  'IntegerConversion.toSigned/toUnsigned with explicit totalSigned/totalUnsigned policies'],
 'carrier':'FP.Value .fp32 carries low16 binary16/bfloat16 payloads and full32 binary32 values. Profile/dtype chooses decoding; carrier embedding is not numerical fp32 rounding.',
 'integer_conversion_ast':'Profile-selected fromInt/fromNat rounds directly to target computation dtype. Explicit .real→target cast is an identity carrier-tag bridge, not a numerical conversion or native .real provenance theorem.',
 'unused_floating_rest':'No Algebra callback occurs in the independent result formula; rest supplies unreachable operations.',
 'total_contract':'Every logical source bit pattern and masked lane has a totalized execution. Undefined active integer conversions retain the explicitly supplied arbitrary policy.',
 'defined_contract':'Only active integer-output addresses must have successful finite/range toward-zero conversion after the real source extension route. Under this domain every arbitrary policy gives the same specified memory.',
 'floating_output_contract':'No integer-conversion domain condition or finiteness condition is required for fp16/bf16 output. Concrete software arithmetic covers exceptional encodings.',
 'native_conformance_proved':False}
numerical=[
 {'id':'VA-SM-N1','status':'software_concrete_native_binding_deferred','primitive':'direct signed/unsigned integer→fp16 or fp32',
  'detail':'Own-width BitVec normalization precedes one exact-rational rounding. fp16 uses Binary16.round directly, never round-to-fp32 then narrow. bf16 mixed profiles round integer operands to fp32. Bind native rounding mode/format/representability.'},
 {'id':'VA-SM-N2','status':'software_concrete_native_binding_deferred','primitive':'binary16 and binary32 addition',
  'detail':'Binary16.add is used for fp16 mixed profiles; FP.add for bf16 mixed profiles. Ordered operands are retained. Bind cfg, flush behavior, rounding and NaN/signaling/payload behavior; no commutativity or real exactness assumption is substituted.'},
 {'id':'VA-SM-N3','status':'software_concrete_native_binding_deferred','primitive':'bf16→fp32, fp16→fp32, fp32→bf16',
  'detail':'bf16 input widens by exact sign/exponent/fraction embedding. Half results widen by BinaryConversion.widen before integer conversion; bf16 floating outputs narrow with FP.cast .bf16 cfg. Native widening signaling/NaN/flush behavior and narrowing must be bound. BinaryConversion.narrow is not used as a rounding operation.'},
 {'id':'VA-SM-N4','status':'software_partial_conversion_and_active_domain_independence_proved_native_binding_deferred','primitive':'binary32 toward-zero integer conversion',
  'detail':'toSigned/toUnsigned retains destination width, including signed8 for bool stores. Finite/range domain is explicit; undefined policy remains visible outside it. No finite/representable runtime guard was added to Python. Policy independence only constrains active integer stores.'}]
structural=[
 {'id':'VA-SM-S1','gap':'57 promoted dtype pairs have no logical value/grid proof',
  'detail':'18 mixed fp64,11 supported fp64 floating,28 semantically promoted fnuz frontend rejections. Errors and unsupported target behavior still require source-level contracts.'},
 {'id':'VA-SM-S2','gap':'Native source/compiler/error/poison/sanitizer refinement',
  'detail':'All36 exact-source TTIR graphs are copied with verified frozen-audit hashes. Every graph was audited through ordered loads/conversions/add/store, pointer roots, absent load other, signed i32 pid*1024+arange and mask slt n_elements. This is an SSA observation audit, not a formal extraction/compiler theorem. No new frontend/native backend/GPU runs; totalized software execution does not prove native poison propagation or absence of traps.'},
 {'id':'VA-SM-S3','gap':'Native byte/tag/storage and finite-width addresses',
  'detail':'Low16 payload carrier views, logical Int/Nat cells, own-width normalization, bool i8 reinterpretation and exact Nat offsets need native representability and allocation/layout refinement. The i32 native offset/mask expressions are audited but their overflow/extent behavior is not proved here. Missing shape/stride/device guards were not invented; physical overlapping views with different bases/widths/strides remain open.'},
 {'id':'VA-SM-S4','gap':'Native scheduling and host/harness composition',
  'detail':'This batch proves the complete ascending logical grid with current-memory reads and equal logical region aliases. Instruction interleaving, allocation/empty_like/device behavior, native wrapper dispatch and Python/Torch harness execution retain prior gaps.'}]
scope={
 'task':'vector_addition concrete mixed integer/bool and fp16/bf16 extension',
 'standard':prior['standard'],'source_sha256':prior['source_sha256'],
 'whole_original_program_accepted':False,'completed_original_program_increment':0,
 'counts':counts,'public_contract':public,'numerical_assumptions':numerical,
 'structural_remaining':structural,'dtype_pairs':rows,
 'inherited_obligations':'All previous scope obligations persist except the36 explicitly proved logical source/value/grid profiles.',
 'prior_mixed_fp32_receipt_sha256':sha(previous/'evidence/receipt.json'),
 'prior_mixed_fp32_scope_sha256':sha(previous/'evidence/scope-review.json'),
 'source_route_profile_sha256':sha(root/'evidence/source-profile/receipt.json'),
 'independent_review':'pending formal frozen-package review; source audit improved after separate static reviewer feedback'}
dump(root/'evidence/scope-review.json',scope)
results=json.loads((root/'evidence/verification-results.json').read_text())
dependencies=json.loads((root/'evidence/final-dependencies-after.json').read_text())
artifacts={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
 if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean','.pyc'}
 and not (p.name=='receipt.json' and p.parent==root/'evidence')}
receipt={
 'task':scope['task'],'package_root':str(root),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'whole_original_program_accepted':False,'original_program_completion_increment':0,
 'modules':{r['module']:sha(root/r['module']) for r in results},'public_contract':public,
 'verification':{'module_count':len(results),'actual_exit_codes':[r['exit_code'] for r in results],
  'axiom_checks':sum((root/r['log']).read_text().count('axiom footprint') for r in results),
  'noncircularity_checks':sum((root/r['log']).read_text().count('non-circular ✓') for r in results),
  'source_and_dependency_hashes_unchanged':True,'dependency_file_count':len(dependencies['files']),
  'imported_project_olean_hashes_unchanged':True,'external_library_build_rechecked':False,
  'staged_whitespace_check_exit_code':0,'warnings':0,'module_axiom_audits':6,'results':'evidence/verification-results.json'},
 'coverage_summary':counts,'acceptance_limits':structural,
 'failed_attempts':['evidence/failed-attempts/value-stack-overflow/verification.json'],
 'source_binding':{'python':'bench/tritonbench_g/vector_addition/vector_addition.py',
  'source_sha256':prior['source_sha256'],'route_profile_sha256':scope['source_route_profile_sha256'],
  'source_native_refinement_proved':False},
 'prior_receipts':dependencies['prior_receipts'],'artifacts':artifacts}
dump(root/'evidence/receipt.json',receipt)
print(json.dumps({'receipt_sha256':sha(root/'evidence/receipt.json'),'scope_sha256':sha(root/'evidence/scope-review.json'),'verification':receipt['verification'],'counts':counts}))
