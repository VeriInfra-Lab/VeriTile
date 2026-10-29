"""Freeze exact successful finite-address proof evidence and explicit limits."""
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
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
assert not (root/'evidence/receipt.json').exists(),'refuse receipt overwrite'
subprocess.run([sys.executable,'-B',str(root/'freeze.py'),'after'],check=True)
sys.path.insert(0,str(repo/'bench'))
from adapter_common import declarations
rows=read(root/'evidence/verification-results.json')
modules={r['module']:sha(root/r['module']) for r in rows}
logs=[(root/r['log']).read_text() for r in rows]
deps=read(root/'evidence/final-dependencies-after.json')
checks={'module_count':len(rows),'actual_exit_codes':[r['exit_code'] for r in rows],
        'seconds':sum(r['seconds'] for r in rows),
        'named_axiom_checks':sum(s.count('axiom footprint') for s in logs),
        'spec_noncircularity_checks':sum(s.count('non-circular ✓') for s in logs),
        'module_axiom_audits':sum((root/p).read_text().count('#auditModuleAxioms') for p in modules),
        'warnings':sum(s.count('warning:') for s in logs),
        'dependency_file_count':len(deps['files']),
        'dependency_breakdown':dict(Counter(v['source_kind'] for v in deps['files'].values())),
        'unchanged_audited_ttir_count':len(deps['audited_ttir_sha256']),
        'source_and_dependency_hashes_unchanged':True,
        'all_nine_frozen_relu_source_modules_unchanged':True,
        'actual_lakefile_toml_bound':True,'external_library_build_rechecked':False,
        'source_trailing_whitespace_check':'all three sources end with newline and have no trailing spaces/tabs',
        'compiler_results':'evidence/verification-results.json'}
assert checks['actual_exit_codes']==[0,0,0]
assert checks['named_axiom_checks']==21 and checks['spec_noncircularity_checks']==3
assert checks['module_axiom_audits']==3 and checks['warnings']==0
publicfile='bench/tritonbench_g/relu_triton_kernel/AddressGridCorrect.lean'
publicname='relu_word_grid_finite_address_correctness'
publicdecl=declarations(root/publicfile)[publicname]
assert publicdecl['kind']=='specification'
public={'file':publicfile,'declaration':'VeriTile.Bench.TritonBenchG.ReluTritonKernel.AddressGridCorrect.'+publicname,
    'parser_name':publicname,'kind':'specification','statement_sha256':publicdecl['statement_sha256'],
    'model_kind':'backend_value_contract','role':'parametric_value_contract',
    'dtype_scope':'selected_explicit_profiles_only',
    'profiles':['int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool'],
    'unclosed_numeric_callbacks':read(words/'evidence/receipt.json')['public_contract']['unclosed_numeric_callbacks'],
    'native_conformance_proved':False,
    'premises':['Complete tagged InputWords relation from the frozen word-grid package.',
        'B fixed to the original wrapper constant1024; N<2^31.',
        'Representable input/output bases and full N-element exclusive endpoints at most2^64, with1/2/4/8-byte input elements and4-byte output elements.'],
    'conclusion':'Existing logical grid success, exact word result and full frame, paired with an independent finite-index and byte-address arithmetic certificate.'}
remaining=[
 {'id':'RELU-A-S1','gap':'Native load/storage and physical view correspondence',
  'detail':'Logical x/y regions are not related to inputBase/outputBase by a memory representation relation. No actual byte memory, assembly, endianness, alignment, pointer provenance, contiguous/strided layout or allocation correctness is proved. Numeric intervals are not proof of allocated objects.'},
 {'id':'RELU-A-S2','gap':'Compiler and finite-address model correspondence',
  'detail':'Source-prefix registers are formally related to bounded BitVec index/mask expressions. The selected64-bit signExtend/scale/add formula is proved arithmetically, but TTIR addptr lowering, bool i1 pointer byte scaling, native pointer formation and execution against byte addresses remain unproved. No interpreter with native pointers is introduced.'},
 {'id':'RELU-A-S3','gap':'Domains outside the finite1024-tile subtheorem and native launch/errors',
  'detail':'N>=2^31, other block sizes, invalid block domains, native dtype/constant specialization, launch limits, zero-grid behavior, integer-overflow debug/sanitizer outcomes and masked-lane pointer/poison behavior remain open. The N=0 conclusion is only the inherited logical empty-grid result.'},
 {'id':'RELU-A-S4','gap':'Original dtype, host and concurrency scope',
  'detail':'Prior fp64/fp8/error dtype routes, native host/harness/module stages, allocation callbacks and instruction-level concurrent execution remain open. The grid remains the original ascending logical sequence with only pid0 storing.'}
]
scope={'task':'relu_triton_kernel finite address arithmetic composed with finite-word grid',
    'standard':read(words/'evidence/scope-review.json')['standard'],
    'whole_original_program_accepted':False,'completed_original_program_increment':0,
    'source_sha256':sha(repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'),
    'prior_frozen_receipts':deps['frozen_receipts'],'public_contract':public,
    'new_proved':[
        'For fixed1024 and every0<=N<2^31, the complete padded grid extent is at most2^31, including the N=2^31-1 boundary.',
        'All launched pid/lane i32 multiply/add values agree with Nat offsets before masking; signed<N and pid==0 agree with the original logical conditions.',
        'The first four statements extracted directly from frozen TypedValue.source execute successfully and their actual offsets/mask registers equal the bounded finite expressions.',
        'For active accesses, signExtend64/element-scale/base-add and every intra-element byte address are exact under explicit mathematical64-bit endpoint bounds.',
        'Every element byte, including byte width-1, lies inside its mathematical view interval and below2^64; byte coordinates are injective and separated intervals are disjoint.',
        'The frozen word-grid success/result/frame theorem composes directly with the independent address certificate, without assuming execution success or a desired result.'
    ],
    'counts':{'finite_word_profiles':9,'new_profile_names':0,'new_public_contracts':1,
        'original_logical_profiles_unchanged':12,'audited_ttir_profiles':13,
        'native_backend_compilations':0,'gpu_executions':0},
    'domain':{'N':'Nat with N<2^31; no original host input restriction or native dispatch theorem inferred',
        'B':1024,'base':'Nat less than2^64','input_end':'inputBase+N*(encoding.bits/8)<=2^64',
        'output_end':'outputBase+N*4<=2^64 (stronger than the first-tile-only write footprint)',
        'inactive_lanes':'All i32 indices/masks are exact, but inactive byte pointers have no range requirement or native validity claim.',
        'zero_N':'No launched lanes in the logical model; native zero-grid behavior is not proved.'},
    'memory_model':'Same logical whole tagged cells and full prelaunch-memory frame as the frozen word-grid proof. AddressCertificate is separate arithmetic; x/y and bases have no native memory mapping premise or conclusion.',
    'alias_policy':'Logical x=y remains allowed. Arithmetic disjointness is a separate conditional lemma and does not establish physical memory aliases, allocation separation or native mixed-width view behavior.',
    'structural_remaining':remaining,
    'prior_original_structural_obligations_retained':read(original/'evidence/scope-review.json')['structural_remaining'],
    'numerical_followups':read(words/'evidence/scope-review.json')['numerical_followups'],
    'new_numerical_callbacks':[],
    'suggested_covered_stage':'fixed1024_bounded_index_and_byte_address_arithmetic_with_word_grid',
    'source_view_constraint':'Preserve original DSL source views; TTIR observation at N1025 and mathematical64-bit address definitions do not establish native compiler correspondence.',
    'route_evidence':{'path':'evidence/address-route-review.json','sha256':sha(root/'evidence/address-route-review.json')},
    'verification':checks,'independent_review':'pending root/independent integration review'}
write(root/'evidence/scope-review.json',scope)
artifacts={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
    if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean','.pyc'}
    and p!=root/'evidence/receipt.json'}
receipt={'task':scope['task'],'package_root':str(root),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
    'whole_original_program_accepted':False,'original_program_completion_increment':0,
    'modules':modules,'public_contract':public,'verification':checks,'coverage_summary':scope['counts'],
    'prior_frozen_receipts':deps['frozen_receipts'],'acceptance_limits':remaining,'artifacts':artifacts}
write(root/'evidence/receipt.json',receipt)
print(json.dumps({'receipt_sha256':sha(root/'evidence/receipt.json'),'scope_sha256':sha(root/'evidence/scope-review.json'),
    'public_statement_sha256':public['statement_sha256'],'modules':modules,'checks':checks},indent=2))
