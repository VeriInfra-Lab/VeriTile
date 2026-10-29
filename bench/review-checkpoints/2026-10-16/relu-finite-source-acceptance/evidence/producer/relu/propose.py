"""Generate exact source-derived ReLU acceptance metadata; never mark reviewed."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0, str(REPO/'bench'))
from adapter_common import canonical, declarations, sha
from check_original_program_scope import declaration_scope, validate_case, validate_deferrals

TASK = 'relu_triton_kernel'
BASE = 'bench/tritonbench_g/'+TASK+'/'
MODULES = [BASE+p+'.lean' for p in ['Proofs/FiniteIndices','Proofs/FiniteBoundary','Proofs/FiniteHost','Proofs/FiniteOutcomes','FiniteCorrect']]
PYTHON = BASE+TASK+'.py'
PUBLIC = MODULES[-1]
NAME = 'relu_finite_source_outcomes'
NS = 'VeriTile.Bench.TritonBenchG.ReluTritonKernel.'
hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text())
def save(name, data):
    p=ROOT/'evidence'/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def check_source():
    text=(REPO/PYTHON).read_text();tree=ast.parse(text)
    fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    host=[ast.unparse(n) for n in fs['relu'].body]
    assert host==['out = torch.empty_like(x, dtype=torch.float32, device=x.device)',
      'N = out.numel()', 'BLOCK_SIZE = 1024', 'num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE',
      'grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)',
      'relu_kernel[grid](x, out, N, BLOCK_SIZE)', 'return out']
    kernel=[ast.unparse(n) for n in fs['relu_kernel'].body]
    assert kernel==['pid = tl.program_id(0)','block_start = pid * block_size',
      'offsets = block_start + tl.arange(0, block_size)','mask = offsets < N',
      'x = tl.load(x_ptr + offsets, mask=mask)','result = tl.where(x >= 0, x, 0.0)',
      'if pid == 0:\n    tl.store(out_ptr + offsets, result, mask=mask)']
    extracted=ast.parse((ROOT/'evidence/frontend/extracted_relu_kernel.py').read_text())
    fun=next(n for n in extracted.body if isinstance(n,ast.FunctionDef))
    assert ast.dump(fun,include_attributes=False)==ast.dump(fs['relu_kernel'],include_attributes=False)
    profiles=load(ROOT/'evidence/frontend/frontend-profiles.json')
    assert len(profiles)==85
    for n in [0,1,1024,1025,2**31-1]:
        rows=[r for r in profiles if r['N']==n]
        assert len(rows)==17 and sum(r['status']=='frontend_generated' for r in rows)==13
        for r in rows:
            if 'ttir' in r:
                raw=ROOT/'evidence/frontend'/r['ttir'];assert hash_file(raw)==r['ttir_sha256']
                ttir=raw.read_text()
                assert 'tt.get_program_id' in ttir and 'tt.load' in ttir and 'scf.if' in ttir and 'tt.store' in ttir
                assert ttir.index('tt.load') < ttir.index('arith.select') < ttir.index('scf.if') < ttir.index('tt.store')
            elif r['input'] in ['float8_e4m3fn','float8_e5m2']:
                assert 'unexpected type' in r['error']
            else:
                assert r['input'] in ['float8_e4m3fnuz','float8_e5m2fnuz']
                assert 'not supported in this architecture' in r['error']
    receipt=load(ROOT/'evidence/frontend/receipt.json')
    assert hash_file(REPO/PYTHON)==receipt['source_sha256']
    boundary=load(ROOT/'evidence/frontend-boundary/frontend-profiles.json')
    assert len(boundary)==68
    for n in [2**31,2**31+1,2**32-1,2**32]:
        rows=[r for r in boundary if r['N']==n]
        assert len(rows)==17 and sum('ttir' in r for r in rows)==13
        for r in rows:
            if 'ttir' in r:
                raw=ROOT/'evidence/frontend-boundary'/r['ttir']
                assert hash_file(raw)==r['ttir_sha256']
                ttir=raw.read_text()
                assert ('arith.cmpi ult' if n<2**32 else 'arith.cmpi slt') in ttir
                assert 'arith.muli %pid, %block_start_0 : i32' in ttir
                assert 'tt.addptr' in ttir and 'tensor<1024xi32>' in ttir
            elif r['input'] in ['float8_e4m3fn','float8_e5m2']:
                assert 'unexpected type' in r['error']
            else:assert 'not supported in this architecture' in r['error']
    save('source-check.json',{'source':PYTHON,'source_sha256':hash_file(REPO/PYTHON),
      'host_statements':host,'kernel_statements':kernel,'kernel_ast_identical':True,
      'frontend_pairs':85,'frontend_successes':65,'comparison_rejections':10,'signature_rejections':10,
      'tested_N':[0,1,1024,1025,2**31-1],'full_N_domain_is_not_exhaustively_probed':True,
      'boundary_frontend_pairs':68,'boundary_frontend_successes':52,'boundary_frontend_rejections':16,
      'included_N_endpoint':2**31,'excluded_unsafe_N_begin':2**31+1,
      'boundary_diagnosis':'N is constexpr: at2^31 mask uses unsigned i32 and all offsets remain signed-positive. The next block has signed multiply overflow before the gate; larger configurations still frontend-generate, so are not frontend rejections.',
      'generalization_basis':'N supplies a signed-i32 mask bound below2^31 and an unsigned-i32 bound at2^31; tile shape1024 and value dtype route are independent of N. Trusted frontend type primitive rule plus source review, not an inference from samples.',
      'native_backend_compiled':False,'gpu_executed':False,
      'test_harness_review':'test_relu is non-JIT, has no production caller, and calls the proved relu entrypoint four times. The import-time result_gold call is test initialization, retained in HostWrapper.initializeModule.'})
    return receipt

def generate():
    receipt=check_source()
    ledger=load(REPO/'bench/original_program_scope.json')
    row=deepcopy(next(r for r in ledger['tasks'] if r['task']==TASK))
    inventory=row['source_inventory']
    ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(ROOT/PUBLIC)[NAME]['statement_sha256']}
    names=['int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool',
      'float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz']
    dtype=lambda d:{'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64',
      'float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz'}.get(d,d)
    configs=[]
    for d in names:
        for size,lo,hi in [('empty',0,1),('one_block',1,1025),('multiple_blocks',1025,2**31),('unsigned_constexpr_boundary',2**31,2**31+1)]:
            final='returned' if d in names[:13] else 'comparison_rejected' if d in names[13:15] else 'signature_rejected'
            for outcome in ['device_error','allocation_error','numel_error','launch_setup_error',final]:
                compute={'index':'int32','mask_comparison':'uint32' if size=='unsigned_constexpr_boundary' else 'int32','result':'fp32' if d in names[:9] else dtype(d)}
                if d=='bfloat16':compute['comparison']='fp32'
                else:compute['comparison']='int32' if d in ['int8','int16','uint8','uint16','bool'] else dtype(d)
                c={'dtype_profile':{'inputs':{'x':dtype(d)},'outputs':{'out':'fp32'},'compute':compute},
                  'configuration':{'entrypoint':'relu','source_variant':'original','block_size':1024,
                    'N':{'lower_inclusive':lo,'upper_exclusive':hi},'size_class':size,
                    'outcome':outcome,'grid':'ceil_div_N_1024','target':{'backend':'cuda','arch':90,'warp_size':32},
                    'triton_version':receipt['environment']['triton'],'torch_version':receipt['environment']['torch'],
                    'options_sha256':sha(canonical(receipt['options'])),
                    'layout':'all_actual_strided_views_with_safe_flat_N_element_source_span',
                    'allocation':'trusted_empty_like_preserve_format_float32_fresh_view',
                    'input_storage_offset':'arbitrary_nonnegative_with_extent_and_uint64_no_wrap',
                    'numerical_values':'all_defined_source_values_including_floating_specials',
                    'gate':'no_programs' if size=='empty' else 'pid0_only' if size=='one_block' else 'pid0_stores_other_pids_load_and_select'}}
                validate_case(c);configs.append((d,c))
    cases=[c for _,c in configs]
    assert len(cases)==340 and len({canonical(c) for c in cases})==340
    stage_ids=[s['id'] for s in inventory['stages'] if not s['may_be_test_harness']]
    branch_ids=[s['id'] for s in inventory['branches']]
    dtype_ids=[s['id'] for s in inventory['dtypes']]
    scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stage_ids,'branches':branch_ids,'dtypes':dtype_ids,
      'cases':{ident:deepcopy(cases) for ident in stage_ids+branch_ids+dtype_ids}}
    # Case-specific callback authority. Each port is selected from source and TTIR,
    # not inferred from the presence of a user-written numerical binding.
    cb={'fromInt':'VeriTile.Triton.FP.Structural.Algebra.fromInt',
      'fromNat':'VeriTile.Triton.FP.Structural.Algebra.fromNat','literal':'VeriTile.Triton.FP.Structural.Algebra.literal',
      'cast':'VeriTile.Triton.FP.Structural.Algebra.cast',
      'cmp16':'FiniteCorrect.relu_finite_source_outcomes.cmp16',
      'cmp32':'FiniteCorrect.relu_finite_source_outcomes.cmp32',
      'cmp64':NS+'FP64Source.orderedGeZero64',
      'zero64':NS+'FP64Source.value [zero literal 0#64]',
      'narrow':NS+'FP64Source.Numerics.narrow64to32'}
    families={
      'SINT32':('fromInt','arith.sitofp','Selected nonnegative signed finite source integer converts directly to fp32; exact native rounding, sign, overflow and subnormal policy.'),
      'UINT32':('fromNat','arith.uitofp','Unsigned finite source integer or normalized bool i1 converts directly to fp32; exact native conversion policy.'),
      'ZERO32':('literal','fp32_positive_zero','Native fp32 zero literal and/or integer-zero conversion is represented by M.literal none fp32 0.'),
      'ZERO16':('literal','fp16_positive_zero','Native fp16 positive zero, including integer-zero comparison operand and selected false branch.'),
      'ZEROBF16':('literal','bf16_positive_zero','Native bf16 positive zero in selected false branch.'),
      'CMP16':('cmp16','arith.cmpf_oge_fp16','cmp16 zero x is native ordered x>=+0 in fp16, including NaN false and both signed zeros true.'),
      'CMP32':('cmp32','arith.cmpf_oge_fp32','cmp32 zero x is native ordered x>=+0 in fp32, including NaN false and both signed zeros true.'),
      'EXT16':('cast','arith.extf_fp16_fp32','M.cast none fp16 fp32 agrees with native extension, including specials and denormal behavior.'),
      'EXTBF16':('cast','arith.extf_bf16_fp32','M.cast none bf16 fp32 agrees with native extension at both comparison and output ports, including specials.'),
      'CMP64':('cmp64','arith.cmpf_oge_fp64','Concrete sign/exponent/fraction word comparison agrees with native ordered fp64 x>=+0 for every input word.'),
      'ZERO64':('zero64','fp64_positive_zero','Both source zero operands materialize as the exact fp64 positive-zero word.'),
      'NARROW64':('narrow','arith.truncf_fp64_fp32','narrow64to32 agrees with native fp64 to fp32 conversion, including ties, overflow, underflow, signed zero, infinities and NaNs.')}
    def ports(d):
        if d.startswith('int'):return [('SINT32','selected_nonnegative','where_true'),('ZERO32','selected_negative','where_false')]
        if d.startswith('uint') or d=='bool':return [('UINT32','all_unsigned_words' if d!='bool' else 'all_256_bool_bytes_normalized_to_i1','where_true')]
        if d=='float16':return [('ZERO16','all_fp16_words','comparison_zero'),('CMP16','all_fp16_words','comparison'),('ZERO16','ordered_comparison_false','where_false'),('EXT16','all_selected_fp16_words','store')]
        if d=='bfloat16':return [('EXTBF16','all_bf16_words','comparison_cast'),('ZERO32','all_bf16_words','comparison_zero'),('CMP32','all_extended_bf16_words','comparison'),('ZEROBF16','ordered_comparison_false','where_false'),('EXTBF16','all_selected_bf16_words','store')]
        if d=='float32':return [('ZERO32','all_fp32_words','comparison_zero'),('CMP32','all_fp32_words','comparison'),('ZERO32','ordered_comparison_false','where_false')]
        if d=='float64':return [('CMP64','all_fp64_words','comparison'),('ZERO64','all_fp64_words','zero_operands'),('NARROW64','all_selected_fp64_words','store')]
        return []
    def types(suffix,d):
        if suffix=='SINT32':return [d], 'fp32'
        if suffix=='UINT32':return ['uint1_after_bool_normalization' if d=='bool' else d], 'fp32'
        if suffix.startswith('ZERO'):return ['source_zero_literal'], {'ZERO32':'fp32','ZERO16':'fp16','ZEROBF16':'bf16','ZERO64':'fp64'}[suffix]
        if suffix.startswith('CMP'):
            ty={'CMP16':'fp16','CMP32':'fp32','CMP64':'fp64'}[suffix]
            return [ty,ty], 'bool'
        return [{'EXT16':'fp16','EXTBF16':'bf16','NARROW64':'fp64'}[suffix]], 'fp32'
    rules={}
    for suffix,(callback,op,validation) in families.items():
        ident='RELU-FINITE-'+suffix
        rules[ident]={'id':ident,'status':'pending_numeric_validation','queue_status':'awaiting_explicit_numeric_validation',
          'affected_tasks':[TASK],'required_validation':validation,'native_whole_kernel_validated':False,
          'validation_evidence':None,'scope_bindings':[]}
    provenance={};requirements=[];deferrals=[];links={}
    for d,c in configs:
        selected=ports(d) if c['configuration']['outcome']=='returned' and c['configuration']['size_class']!='empty' else []
        refs=[]
        for suffix,scalar_domain,port in selected:
            callback,operation,validation=families[suffix];callback=cb[callback];ident='RELU-FINITE-'+suffix
            body={'task':TASK,'declarations':[ref],'callbacks':[callback],**deepcopy(c)}
            bid=TASK+':finite:'+sha(canonical({'rule':ident,'binding':body,'port':port}))[:32]
            binding={'id':bid,**body};rules[ident]['scope_bindings'].append(binding)
            link={'assumption_id':ident,'scope_binding_id':bid};links.setdefault(callback,[]).append(link)
            refs.append({'callback':callback,**link})
            p={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,
              **link,'callback_declaration':callback,'correspondence_kind':'concrete_scalar_port' if suffix in ['CMP64','ZERO64'] else 'parametric_callback', 'port':{'name':port,'operation':operation,'storage_dtype':d,
                'input_types':types(suffix,d)[0],'output_type':types(suffix,d)[1]},
              'scalar_domain':scalar_domain,'scalar_path_predicate':'active_source_element; '+scalar_domain,
              'native_execution':False,'source_statement':'result = tl.where(x >= 0, x, 0.0)' if port!='store' else 'tl.store(out_ptr + offsets, result, mask=mask)',
              'original_source':{'path':PYTHON,'sha256':hash_file(REPO/PYTHON)},
              'public_statement':ref,'lean_source_sha256':{f:hash_file(ROOT/f) for f in MODULES},
              'typed_route_source_sha256':hash_file(REPO/(BASE+'Proofs/TypedValue.lean')),
              'fp64_route_source_sha256':hash_file(REPO/(BASE+'Proofs/FP64Source.lean')),
              'raw_ttir':{'path':('evidence/frontend-boundary/2147483648-' if c['configuration']['size_class']=='unsigned_constexpr_boundary' else 'evidence/frontend/1025-')+d+'.raw.ttir','sha256':hash_file(ROOT/(('evidence/frontend-boundary/2147483648-' if c['configuration']['size_class']=='unsigned_constexpr_boundary' else 'evidence/frontend/1025-')+d+'.raw.ttir'))},
              'frontend_receipt':{'path':('evidence/frontend-boundary/' if c['configuration']['size_class']=='unsigned_constexpr_boundary' else 'evidence/frontend/')+'receipt.json',
                'sha256':hash_file(ROOT/(('evidence/frontend-boundary/' if c['configuration']['size_class']=='unsigned_constexpr_boundary' else 'evidence/frontend/')+'receipt.json'))},
              'configuration':c['configuration'],
              'structural_prerequisites_not_waived':['SourceDomain and Ready','trusted metadata/allocator/frontend/memory primitive boundary','safe view extent and permissions','no full-tensor ReLU claim when N>1024','no numerical UB or false real=floating equality']}
            provenance[ident+'::'+bid]=p
        requirements.append({**deepcopy(c),'required_callbacks':sorted({r['callback'] for r in refs}),'required_binding_refs':refs})
    for r in rules.values():r['scope_bindings'].sort(key=lambda b:b['id'])
    for ident,r in rules.items():
        for b in r['scope_bindings']:
            deferrals.append({'kind':'numerical_rule_conformance','assumption_id':ident,'rule_sha256':sha(canonical(r)),
              'scope_binding_id':b['id'],'dtype_profile':b['dtype_profile'],'configuration':b['configuration'],
              'required_validation':r['required_validation']})
    union=sorted(links)
    declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract',
      'scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stage_ids,
      'dtype_profile':'All17 canonical source-target dtype profiles;13 successes,2 ordered-comparison rejects,2 signature rejects. Empty and all finite0<=N<=2^31; block1024; actual strided flat views and fresh float32 allocation.',
      'dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],
      'native_binding_status':'not_established',
      'review_basis':'PRODUCER PROPOSAL: independent root substantive review required. Source-ordered primitive errors, all dtype dispatch branches, actual grid execution, allocation metadata/freshness, finite valid views, masks/frames and untouched tail are connected by relu_finite_source_outcomes.',
      'original_scope':scope}
    declaration_scope(declaration,inventory)
    coverage={d:[] for d in ['stages','dtypes','branches']}
    for dim,items in inventory.items():
        for item in items:
            if dim=='stages' and item['may_be_test_harness']:
                coverage[dim].append({'id':item['id'],'status':'test_harness','explanation':'Non-JIT test_relu has no production caller; four fixed source examples invoke relu. The import-time assignment runs this test helper; existing HostWrapper models its calls and failure propagation. No RNG implementation obligation.'})
            else:
                coverage[dim].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[ref],
                  'explanation':'Complete finite source-target API with all17 canonical dtypes, four disjoint N ranges, each early error site and all terminal dispatch outcomes. Successful calls derive execution from existing typed/fp64 grids and connect physical input views, output metadata/freshness, addresses, masks, frames and pid0-only stores. Unsafe/out-of-domain states remain explicit in domain-review.json.'})
    row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,
      deferred_numerical_assumptions=deferrals,callback_assumptions=links,
      proposed_status_after_independent_root_review='verified_with_deferred_numerics')
    row['input_sha256']['lean'].update({f:hash_file(ROOT/f) for f in MODULES})
    row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review',
      'policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(REPO/'bench/original_program_acceptance.md'),
      'finite_domain':'SourceDomain x and Ready views x s on CUDA sm90/Triton3.7.1/Torch2.13; all17 canonical profiles retained.',
      'source_bug':'For N>1024 output tail remains allocation payload; this is as-written source acceptance only.',
      'native_correctness':False,'queue_integration':'Exact authority delta must be integrated; proposed JSON alone does not satisfy the unchanged global queue checker.'}
    authority={'schema':1,'kind':'relu_finite_exact_numeric_authority_delta','task':TASK,'rules':list(rules.values()),
      'provenance':provenance,'requirements':[{'task':TASK,'declaration':ref,
        'source_ref':{'path':PUBLIC,'sha256':hash_file(ROOT/PUBLIC)},
        'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),
        'callback_union':union,'cases':requirements,
        'support_boundary':'All340 explicit finite source outcome cases;84 numeric ports on39 positive success cases; errors/empty launches need no numerical conformance.'}],
      'summary':{'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),
        'registered_declarations':1,'native_executions':0,'original_program_completion_increment':0}}
    assert len(provenance)==84
    validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
    expected={(r['assumption_id'],r['scope_binding_id']) for c in requirements for r in c['required_binding_refs']}
    assert expected=={(d['assumption_id'],d['scope_binding_id']) for d in deferrals}
    assert union==sorted({c for r in requirements for c in r['required_callbacks']})
    for c in requirements:
        if c['configuration']['outcome']!='returned' or c['configuration']['size_class']=='empty':assert not c['required_callbacks']
    save('declaration-proposal.json',declaration)
    save('scope-row-proposal.json',row)
    save('numerical-authority-delta.json',authority)
    save('scope-cases.json',cases)
    save('proposal-validation.json',{'status':'passed','checks':['current public statement parser','unchanged schema2 declaration_scope','all340 validate_case checks','unchanged exact numerical validate_deferrals','complete per-case port/callback and deferral bijection'],
      'counts':authority['summary'],'independent_root_review':'pending','global_queue_check':'pending authority integration; not claimed passed','scope_acceptance_increment':0})
    print(json.dumps(authority['summary']))

if __name__=='__main__':generate()
