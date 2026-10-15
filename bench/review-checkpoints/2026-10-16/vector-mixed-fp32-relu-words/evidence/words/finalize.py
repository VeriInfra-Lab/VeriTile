"""Freeze the word-refinement package after actual Lean verification."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
prior=root.parent/'relu-original-wip'
sys.path.insert(0,str(repo/'bench'))
from adapter_common import declarations
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
subprocess.run([sys.executable,str(root/'freeze.py'),'after'],check=True)
results=read(root/'evidence/verification-results.json')
dependencies=read(root/'evidence/final-dependencies-after.json')
old=read(prior/'evidence/receipt.json')
oldscope=read(prior/'evidence/scope-review.json')
modules={r['module']:sha(root/r['module']) for r in results}
publicfile='bench/tritonbench_g/relu_triton_kernel/WordGridCorrect.lean'
publicname='relu_word_grid_correctness'
publicdecl=declarations(root/publicfile)[publicname]
assert publicdecl['kind']=='specification'
logs=[(root/r['log']).read_text() for r in results]
checks={'module_count':len(results),'actual_exit_codes':[r['exit_code'] for r in results],
        'seconds':sum(r['seconds'] for r in results),
        'named_axiom_checks':sum(s.count('axiom footprint') for s in logs),
        'spec_noncircularity_checks':sum(s.count('non-circular ✓') for s in logs),
        'module_axiom_audits':sum((root/p).read_text().count('#auditModuleAxioms') for p in modules),
        'warnings':sum(s.count('warning:') for s in logs),
        'dependency_file_count':len(dependencies['files']),
        'dependency_breakdown':dict(Counter(v['source_kind'] for v in dependencies['files'].values())),
        'source_and_dependency_hashes_unchanged':True,'head_equality_required':False,
        'actual_lakefile_toml_bound':True,'external_library_build_rechecked':False,
        'compiler_results':'evidence/verification-results.json',
        'source_trailing_whitespace_check':'all three sources have a final newline and no trailing spaces/tabs'}
assert checks['actual_exit_codes']==[0,0,0]
assert checks['named_axiom_checks']==23 and checks['spec_noncircularity_checks']==2
assert checks['module_axiom_audits']==3 and checks['warnings']==0
public={'file':publicfile,
        'declaration':'VeriTile.Bench.TritonBenchG.ReluTritonKernel.WordGridCorrect.'+publicname,
        'parser_name':publicname,'kind':'specification','statement_sha256':publicdecl['statement_sha256'],
        'model_kind':'backend_value_contract','role':'parametric_value_contract',
        'dtype_scope':'selected_explicit_profiles_only',
        'profiles':['int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool'],
        'structural_premise':'InputWords equality of the complete tagged logical cell with explicit finite-word decoding at every offset below N; no successful-execution or result premise.',
        'unclosed_numeric_callbacks':['M.fromInt none on decoded signed source-width integers',
                                     'M.fromNat none on decoded unsigned words or normalized unsigned i1 bool',
                                     'M.literal none fp32 0 in the signed negative branch'],
        'unused_inherited_parameters':'cmp16/cmp32 occur in the existing backend parameterization but no integer/bool source operation or word result calls them.',
        'native_conformance_proved':False}
remaining=[
 {'id':'RELU-W-S1','gap':'Native loads to the explicit InputWords relation',
  'detail':'Actual storage bytes, endianness, source-word assembly, tag correspondence, dtype dispatch and load-validity remain unproved. Finite-word decoding and bool normalization are now formal; defining decoded cells is not a native memory theorem.'},
 {'id':'RELU-W-S2','gap':'Physical aliases and allocation/layout',
  'detail':'Logical equal region names are permitted; shifted, misaligned or mixed-width physical byte views are not represented. The y!=x construction lemma preserves a supplied output region but does not prove fresh or separate empty_like allocation.'},
 {'id':'RELU-W-S3','gap':'Native finite-width address, launch and error semantics',
  'detail':'N, B, program IDs and element offsets still use Nat. Invalid block domains, zero-grid native behavior, pointer overflow and debug/sanitizer behavior are not closed.'},
 {'id':'RELU-W-S4','gap':'Original dtype, host, compiler and execution scope',
  'detail':'The prior fp64 value gap, four observed fp8 errors, noncanonical dtype families, native host/harness/module stages, compiler refinement and instruction interleavings remain open.'}
]
scope={
 'task':'relu_triton_kernel finite input words to typed grid refinement',
 'standard':oldscope['standard'],
 'whole_original_program_accepted':False,'completed_original_program_increment':0,
 'source_sha256':sha(repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'),
 'prior_frozen_receipt':{'path':str(prior/'evidence/receipt.json'),'sha256':sha(prior/'evidence/receipt.json')},
 'prior_frozen_scope':{'path':str(prior/'evidence/scope-review.json'),'sha256':sha(prior/'evidence/scope-review.json')},
 'prior_modules_unchanged':old['modules'],
 'new_proved':[
  'Explicit signed/unsigned BitVec8/16/32/64 decoding with range bounds and source-word roundtrips.',
  'All256 raw bool i8 words normalize via !=0 to unsigned i1 with exact zero/nonzero characterizations.',
  'Signed extension and comparison agrees with decoded two\'s-complement nonnegativity; unsigned and normalized-bool comparisons are universally true in the audited routes.',
  'Word comparison/select/source-width conversion result equals the already-proved typed result on the fully tagged decoded cell.',
  'InputWords binds every active input offset; actual full ascending typed grid success is constructed with an independent initial-word result and complete frame.',
  'Every written result is tagged fp32; an initially fp32-tagged output remains tagged, including its untouched tail.',
  'A constructive logical input memory and a separate y!=x lemma make the logical alias/separate-output boundary explicit.'
 ],
 'counts':{'finite_word_input_profiles':9,'original_logical_profiles_unchanged':12,
           'new_logical_profile_names_added':0,'prior_ttir_hashes_rechecked':13,
           'annotated_integer_bool_routes':9,'raw_bool_byte_domain_size':256,
           'native_backend_compilations':0,'gpu_executions':0},
 'public_contract':public,
 'memory_model':{
  'input_relation':'For all o<N, initial.mem x o equals decodeCell encoding (words o), including the int/nat/bool tag. This rejects relying on mismatched-tag Cell.read defaults.',
  'output':'Only pid0 writes fp32-tagged cells with offsets below both B and N when the modeled grid is nonempty. All other memory is the prelaunch memory.',
  'alias_policy':'No x!=y hypothesis in the relation/grid theorem. withInputWords supplies decoded logical input memory; y!=x is explicitly required only to preserve a separately supplied output initialization. No physical alias correspondence is established.',
  'masked_lanes':'No representation, finiteness or value condition is imposed for offsets outside N; their loads remain unspecified and their stores are masked.'},
 'structural_remaining':remaining,
 'prior_original_structural_obligations_retained':oldscope['structural_remaining'],
 'reduced_gap':'RELU-S4 is refined by a proved finite-word decoding/comparison/normalization relation; native storage satisfying that relation remains a structural obligation.',
 'numerical_followups':[
  {'id':'RELU-W-N1','scope':'Nine finite-word integer/bool profiles after native load-to-InputWords mapping is established',
   'primitive':'M.fromInt/M.fromNat to fp32',
   'status':'descriptive_pending_exact_queue_scope_binding',
   'requirements':['Bind direct source-width signed/unsigned conversion rounding and exceptional behavior to fp32.','Do not reclassify native byte assembly, dtype/tag mapping, pointer validity or physical aliasing as numerical assumptions.']},
  {'id':'RELU-W-N2','scope':'Signed negative source-word branch only',
   'primitive':'M.literal none fp32 0',
   'status':'descriptive_pending_exact_queue_scope_binding',
   'requirements':['Bind the literal to native positive fp32 zero.','The word-level comparisons and bool normalization have no remaining numerical callback.']}
 ],
 'suggested_covered_stage':'nine_profile_finite_input_word_to_pid_zero_grid_refinement',
 'source_view_metadata_proposal':oldscope['source_view_metadata_proposal'],
 'source_view_constraint':'Preserve the prior original shared-DSL source view; no new Python/DSL/compiler refinement is claimed for the word definitions.',
 'route_evidence':{'path':'evidence/route-review.json','sha256':sha(root/'evidence/route-review.json')},
 'verification':checks,
 'independent_review':'pending root/independent integration review'
}
write(root/'evidence/scope-review.json',scope)
artifacts={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
           if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean','.pyc'}
           and p != root/'evidence/receipt.json'}
receipt={'task':scope['task'],'package_root':str(root),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
         'whole_original_program_accepted':False,'original_program_completion_increment':0,
         'modules':modules,'public_contract':public,'verification':checks,'coverage_summary':scope['counts'],
         'prior_frozen_receipt_sha256':sha(prior/'evidence/receipt.json'),
         'prior_modules_unchanged':old['modules'],'acceptance_limits':remaining,'artifacts':artifacts}
write(root/'evidence/receipt.json',receipt)
print(json.dumps({'receipt_sha256':sha(root/'evidence/receipt.json'),
                  'scope_sha256':sha(root/'evidence/scope-review.json'),
                  'public_statement_sha256':public['statement_sha256'],
                  'verification':checks,'module_sha256':modules},indent=2))
