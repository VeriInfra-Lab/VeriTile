"""Freeze the byte-memory refinement, actual proof checks and explicit limits."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
words=root.parent/'relu-integer-word-refinement-wip'
original=root.parent/'relu-original-wip'
address=root.parent/'relu-finite-address-wip'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
assert not (root/'evidence/receipt.json').exists(),'refuse frozen receipt overwrite'
subprocess.run([sys.executable,'-B',str(root/'freeze.py'),'after'],check=True)
sys.path.insert(0,str(repo/'bench'))
from adapter_common import declarations
rows=read(root/'evidence/verification-results.json')
history=read(root/'evidence/verification-attempts.json')
modules={r['module']:sha(root/r['module']) for r in rows}
logs=[(root/r['log']).read_text() for r in rows]
deps=read(root/'evidence/final-dependencies-after.json')
failed=[r for r in history if r['exit_code']!=0]
checks={'module_count':len(rows),'actual_exit_codes':[r['exit_code'] for r in rows],
    'seconds':sum(r['seconds'] for r in rows),
    'named_axiom_checks':sum(s.count('axiom footprint') for s in logs),
    'spec_noncircularity_checks':sum(s.count('non-circular ✓') for s in logs),
    'module_axiom_audits':sum((root/p).read_text().count('#auditModuleAxioms') for p in modules),
    'warnings':sum(s.count('warning:') for s in logs),
    'dependency_file_count':len(deps['files']),
    'dependency_breakdown':dict(Counter(v['source_kind'] for v in deps['files'].values())),
    'imported_project_olean_count':sum('olean_sha256' in row for row in deps['files'].values()),
    'inherited_ttir_count':len(deps['audited_ttir_sha256']),
    'source_dependency_and_olean_hashes_unchanged':True,
    'all_twelve_frozen_relu_source_modules_unchanged':True,
    'actual_lakefile_toml_bound':True,'external_library_build_rechecked':False,
    'source_whitespace_check':'All four sources have a final newline and no trailing spaces/tabs.',
    'compiler_results':'evidence/verification-results.json',
    'all_compile_attempts':'evidence/verification-attempts.json',
    'attempts_retained':len(history),'failed_attempts_retained':len(failed),
    'failure_exit_codes':[r['exit_code'] for r in failed],
    'failure_logs_and_source_snapshots_retained':True}
assert checks['actual_exit_codes']==[0,0,0,0]
assert checks['named_axiom_checks']==30 and checks['spec_noncircularity_checks']==3
assert checks['module_axiom_audits']==4 and checks['warnings']==0
assert checks['dependency_file_count']==180 and checks['imported_project_olean_count']==172
publicfile='bench/tritonbench_g/relu_triton_kernel/ByteGridCorrect.lean'
publicname='relu_byte_grid_correctness'
actual=declarations(root/publicfile)[publicname]
assert actual['kind']=='specification'
public={'file':publicfile,'parser_name':publicname,
    'declaration':'VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteGridCorrect.'+publicname,
    'kind':'specification','statement_sha256':actual['statement_sha256'],
    'model_kind':'backend_value_contract','role':'parametric_value_contract',
    'dtype_scope_relation':'selected_explicit_profiles_only',
    'profiles':['int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool'],
    'unclosed_numeric_callbacks':read(words/'evidence/receipt.json')['public_contract']['unclosed_numeric_callbacks'],
    'bit_interpreter_closed_command':False,'native_binding_status':'not_established',
    'carrier':'BitVec32 for every output Algebra value; full arbitrary32-bit payloads are serialized.',
    'premises':['Complete tagged LogicalView of arbitrary mathematical initial bytes.',
        'Distinct logical input/output names and explicit nonoverlapping full N-element byte intervals.',
        'B1024, N<2^31 and representable64-bit bases/exclusive input-output endpoints.'],
    'conclusion':'Actual old typed-grid success and wordSpec, final LogicalView of the independent byte-grid execution, full byteSpec/frame, finite-address input loads and every output-byte readback, plus the inherited address certificate.'}
remaining=[
 {'id':'RELU-B-S1','gap':'Native storage to mathematical little-endian byte memory',
  'detail':'Real8-bit assembly/splitting and read/write relations are proved for a chosen total Nat-addressed byte map. Native device bytes, little-endian representation, pointer provenance, alignment, allocation/admission and source dtype dispatch remain to be related to that map. This is not merely a renamed logical cell, but it is not a native memory theorem.'},
 {'id':'RELU-B-S2','gap':'Native source/compiler and byte execution correspondence',
  'detail':'The custom byteProgram is the source memory-effect projection with total byte reads, pre-store pid0 lane values and sequential4-byte stores. The nonzero-pid loaded/result functions can remain unevaluated; byte memory-effect success does not imply their native load success. The actual old logical source independently succeeds and its final tagged cells correspond to the byte result. This is an end-to-end representation proof, not a stepwise TTIR/extraction/native instruction theorem. Masked/unused values, access failures, poison, sanitizers and native addptr lowering remain open.'},
 {'id':'RELU-B-S3','gap':'Host-established layout and domain coverage',
  'detail':'Distinct logical names and nonoverlapping contiguous physical byte intervals are explicit subtheorem premises; no runtime guard, empty_like freshness, tensor layout/stride or native wrapper property is inferred. Shifted/overlapping/misaligned layouts, N>=2^31, other block sizes, launch limits and native zero-grid behavior remain open.'},
 {'id':'RELU-B-S4','gap':'Remaining dtype, host/module and concurrent scope',
  'detail':'Nine prior finite-word profiles are refined, with no additional dtype names. Prior fp64/fp8/error families, native host/harness/module stages, compiler targets and instruction interleavings remain required. Math-byte sequential grid success is not whole-original-program acceptance.'}
]
scope={'task':'relu_triton_kernel little-endian mathematical byte-memory refinement',
    'standard':read(words/'evidence/scope-review.json')['standard'],
    'whole_original_program_accepted':False,'completed_original_program_increment':0,
    'source_sha256':sha(repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'),
    'prior_frozen_receipts':deps['frozen_receipts'],'public_contract':public,
    'counts':{'finite_word_profiles':9,'new_dtype_pairs':0,'new_logical_profile_names':0,
        'original_logical_profiles_unchanged':12,'new_public_contracts':1,
        'byte_width_bits':8,'input_word_width_bits':[8,16,32,64],'output_word_width_bits':32,
        'raw_bool_input_bytes':256,'new_frontend_compilations':0,'native_backend_compilations':0,'gpu_executions':0},
    'new_proved':[
        'True little-endian assembly recursively concatenates contiguous8-bit bytes; splitByte extracts each8-bit slice. A per-bit theorem states the exact low-address/low-bit correspondence.',
        'Generic load/store and store/load roundtrips preserve every word and arbitrary initial byte map; explicit encoding casts specialize these to all nine8/16/32/64-bit source profiles.',
        'Sequential4-byte stores equal a non-circular whole-byte reference. Only the first min(N,1024) output words change; every byte outside that interval is preserved.',
        'The memory-effect byte program captures pid0-only stores with pre-store lane values, and the complete ascending byte grid equals the independent initial-byte specification.',
        'LogicalView relates full tagged input cells and fp32 output cells to real byte loads. Under explicit interval separation the view holds after actual old typed-grid execution and the independent byte-grid execution.',
        'Arbitrary initial bytes admit a constructed logical state for distinct logical names without changing the bytes or constraining their output tail.',
        'Generic BitVec64-address load/store equations compose with the existing finite-pointer proof. Public input-word loads and every active output byte, including the final byte, agree at the actual defined64-bit address expression.'
    ],
    'memory_model':{'byte_map':'Nat -> BitVec8, total mathematical memory',
        'load':'count-byte little-endian concatenation at successive addresses',
        'store':'individual8-bit slice writes; complete memory frame',
        'view':'Whole tagged int/nat/bool inputs and fp32 outputs for offsets<N, linked by loadInput/load32.',
        'output_tail':'Arbitrary initial bytes at and after outputBase+4*min(N,1024) remain unchanged, as do all bytes before outputBase.',
        'masked_lanes':'No source/native inactive-value, pointer admission, fault or poison claim. Only active result bytes and full memory effects are refined.',
        'numerical_carrier':'BitVec32 Algebra outputs serialize exactly; numerical callbacks remain unproved native correspondences.'},
    'domain':read(address/'evidence/scope-review.json')['domain'],
    'alias_policy':'Logical x!=y and full input/output numeric byte-interval separation are explicit selected-domain premises, not allocator freshness or added source guards.',
    'structural_remaining':remaining,
    'prior_original_structural_obligations_retained':read(original/'evidence/scope-review.json')['structural_remaining'],
    'numerical_followups':read(words/'evidence/scope-review.json')['numerical_followups'],
    'new_numerical_callbacks':[],
    'suggested_covered_stage':'little_endian_separated_byte_memory_to_finite_word_grid_refinement',
    'source_view_constraint':'Retain original source views. The new mathematical byte-memory execution is not an independently extracted native source or compiler refinement.',
    'inherited_route_review':deps['inherited_route_review'],
    'verification':checks,'independent_review':'pending root/independent integration review'}
write(root/'evidence/scope-review.json',scope)
artifacts={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
    if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean','.pyc'}
    and p!=root/'evidence/receipt.json'}
receipt={'task':scope['task'],'package_root':str(root),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
    'whole_original_program_accepted':False,'original_program_completion_increment':0,
    'modules':modules,'public_contract':public,'coverage_summary':scope['counts'],'verification':checks,
    'prior_frozen_receipts':deps['frozen_receipts'],'acceptance_limits':remaining,
    'failed_attempts':failed,'artifacts':artifacts}
write(root/'evidence/receipt.json',receipt)
print(json.dumps({'receipt_sha256':sha(root/'evidence/receipt.json'),'scope_sha256':sha(root/'evidence/scope-review.json'),
    'public_statement_sha256':public['statement_sha256'],'module_sha256':modules,'verification':checks},indent=2))
