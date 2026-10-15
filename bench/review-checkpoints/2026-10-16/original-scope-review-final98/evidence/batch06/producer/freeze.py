"""Freeze manual scope audit with independently recomputed bindings; all output is /tmp."""
import ast,datetime,hashlib,json,re,sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
root=Path(__file__).resolve().parent
sys.path.insert(0,str(repo/'bench'))
import adapter_common as adapter
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inv=json.loads((root/'inventory.json').read_text())
assert sha(root/'baseline-scope-ledger.json')==inv['ledger_snapshot_sha256']
cp=Path(inv['catalog_path']); assert sha(cp)==inv['catalog_sha256']
cat={r['task']:r for r in json.loads(cp.read_text())['tasks']}
ledger={r['task']:r for r in json.loads((root/'baseline-scope-ledger.json').read_text())['tasks']}
rows={r['task']:r for r in inv['tasks']}
helpers={
 'quantize_kv_copy':['Proofs/TypedQuantize.lean','Proofs/FP32Profile.lean'],
 'int8_matmul_quantization':['Proofs/QuantizeSource.lean','Proofs/MatmulSource.lean','Proofs/QuantizeContract.lean','Proofs/MatmulContract.lean'],
 'rms_matmul_rbe':['Proofs/Contracts.lean'],
 'cross_entropy2':['Proofs/ForwardIO.lean'],
 'kldiv_triton':['Proofs/Contracts.lean','Proofs/SourceDomain.lean'],
 'rms_rbe_matmul':['Proofs/Contracts.lean'],
 'pow_scalar_tensor':['Proofs/IntegerKernels.lean'],
 'int8_dequant_matmul':['Proofs/FPTypedSource.lean'],
 'triton_argmax':['Proofs/TypedSource.lean','Proofs/DimSource.lean','Proofs/DimCorrect.lean','Proofs/BitCorrect.lean','Proofs/BitBackend.lean'],
 'kldiv_ops':['Proofs/SourceDomain.lean','Proofs/Contracts.lean'],
 'ksoftmax_triton':['Proofs/ForwardSource.lean','Proofs/ForwardConstructors.lean','Proofs/ForwardPublic.lean','Proofs/ForwardMath.lean','Proofs/ForwardTypedProfilesTotal.lean','Proofs/BackwardSource.lean','Proofs/BackwardProfiles.lean'],
 'isfinite_kernel':['Proofs/Parameters.lean'],
 'chunk_gated_attention':['Proofs/RealHIO.lean','Proofs/UndefinedTail.lean'],
 'fused_recurrent_hgrn':['Proofs/FPStages.lean'],
 'rope_transform':['Proofs/TypedSource.lean','Proofs/TypedProof.lean','Proofs/TypedContract.lean','Proofs/TypedShape.lean'],
 'context_attn_nopad':['Proofs/RealIO.lean'],
 'bgmv_shrink_kernel':['Proofs/RealIO.lean'],
 'layer_norm_welfold':['Proofs/FPSourceBlock.lean','Proofs/FPSourceSoftware.lean'],
 'chunk_bwd_dqkg':['Proofs/RealIO.lean'],
 'llama_ff_triton':['Proofs/Contracts.lean','Proofs/ExecutionValues.lean'],
 'rotary_transform':['Proofs/FPRow.lean'],
 'chunked_cumsum_fwd':['Proofs/RealIO.lean'],
 'bmm_chunk_bwd':['Proofs/RealContract.lean'],
 'bmm_chunk_fwd':['Proofs/RealContract.lean','Proofs/RealIO.lean'],
 'fused_layernorm_triton':['Proofs/FPNormalize.lean','Proofs/FPWelfordBlock.lean'],
 'chunk_gla_simple':['Proofs/RealFlat.lean','Proofs/TypedExecution.lean'],
 'softmax_triton3':['Proofs/Real.lean'],
 'attention_fwd_triton1':['Proofs/RealIO.lean'],
 'chunk_delta_fwd':['Proofs/RealIO.lean','Proofs/UndefinedTail.lean','Proofs/TypedExecution.lean'],
 'kldiv_compute':['Proofs/Contracts.lean'],
 'quantize_global':['Proofs/TypedScaled.lean','Proofs/TypedShape.lean'],
 'seeded_dropout':['Proofs/RandomDomain.lean','Proofs/RealSurface.lean','Proofs/RealIO.lean','Proofs/Formula.lean'],
 'chunk_cumsum_kernel':['Proofs/RealPadding.lean'],
 'reversed_cumsum_scalar':['Proofs/Source.lean','Proofs/RealExecution.lean','Proofs/RealFormula.lean'],
 'matrix_reduction':['Proofs/TypedSource.lean','Proofs/MachineMax.lean','Proofs/ConcreteCorrect.lean'],
 'rowwise_quantization_triton':['Proofs/TypedScaled.lean','Proofs/TypedSoftware.lean'],
 'nested_loops_processing':['Proofs/Contracts.lean','Proofs/PayloadBoth.lean']}

def check_generated(t):
 if t not in ('swiglu_fwd','kldiv_compute'):return None
 text=(repo/f'bench/tritonbench_g/{t}/Kernels.lean').read_text()
 prefix,macro=('typed','swigluForward') if t=='swiglu_fwd' else ('kernel','kldivSource')
 kinds={'fp32':'float32','fp16':'float16','bf16':'bfloat16','fp64':'float64','f8e4':'float8e4nv','f8e5':'float8e5'}
 spans=list(re.finditer(r'(?m)^def '+prefix+r'_(\w+)_(\w+)\b',text))
 assert len(spans)==36,(t,len(spans))
 covered=set()
 for pos,m in enumerate(spans):
  end=spans[pos+1].start() if pos+1<len(spans) else len(text)
  body=text[m.start():end];a,b=m.groups();assert (a,b) not in covered;covered.add((a,b))
  found=re.findall(r'\| \.(\w+), Bool\.(false|true) =>\s*'+macro+r'%\[tl\.(\w+), tl\.(\w+), tl\.(\w+), (original|commuted)\]',body)
  assert len(found)==12,(t,a,b,len(found))
  seen=set()
  for out,v,atype,btype,otype,version in found:
   assert atype==kinds[a] and btype==kinds[b] and otype==kinds[out]
   assert version==('original' if v=='false' else 'commuted')
   assert (out,v) not in seen;seen.add((out,v))
  assert seen=={(o,v) for o in kinds for v in ('false','true')}
 assert covered=={(a,b) for a in kinds for b in kinds}
 final=re.findall(r'\| \.(\w+), \.(\w+) => '+prefix+r'_(\w+)_(\w+) output v',text)
 assert len(final)==36 and {(a,b) for a,b,c,d in final}==covered
 assert all(a==c and b==d for a,b,c,d in final)
 return {'basis':'Syntax-level exhaustive check of generated selector rows against the manually read macro, not a new frontend/build run.', 'operand_pairs':36,'output_choices':6,'variants':2,'selector_rows_checked':432,'final_dispatch_rows_checked':36,'eligible_original_operand_pairs':32 if t=='kldiv_compute' else 36}

def freeze(t):
 note=json.loads((root/'notes'/f'{t}.json').read_text()); row=rows[t]
 observed=json.loads((root/'read-inputs'/f'{t}.json').read_text())['files']
 for rel,h in observed.items():assert sha(repo/rel)==h,(t,'changed since read',rel)
 for h in helpers.get(t,[]):
  rel=f'bench/tritonbench_g/{t}/{h}';observed[rel]=sha(repo/rel)
 if t=='rope_backward_transform':
  for f in ['TypedSource','TypedProof','TypedContract','TypedShape']:
   rel=f'bench/tritonbench_g/rope_transform/Proofs/{f}.lean';observed[rel]=sha(repo/rel)
 all_inputs=ledger[t]['input_sha256']
 for category,files in all_inputs.items():
  for rel,h in files.items():assert sha(repo/rel)==h,(t,'changed versus scope baseline',rel)
 assert Counter(m['function'] for m in note['production_stage_mapping'])==Counter(p['name'] for p in row['production'])
 assert len(note['production_stage_mapping'])==len(row['production'])
 stage_occurrences=Counter()
 for m in note['production_stage_mapping']:
  candidates=[p for p in row['production'] if p['name']==m['function']]
  src=candidates[stage_occurrences[m['function']]]
  stage_occurrences[m['function']]+=1
  m.update({'path':src['file'],'line':src['line'],'end_line':src['end_line'],'sha256':observed[src['file']], 'stage_id':f"stage:{src['file']}:{src['line']}:{src['name']}"})
 public=[]
 drift=[]
 for d in cat[t]['declarations']:
  rel=d['file'];p=repo/rel;data=p.read_text(); ds=adapter.declarations(p)
  name=d.get('parser_name') or d['declaration']; current=ds[name]
  assert current['kind']=='specification',(t,name,current['kind'])
  assert current['statement_sha256']==d['statement_sha256'],(t,name,'statement mismatch')
  actual_sha=sha(p)
  file_entry=next(f for f in cat[t]['files'] if f['path']==rel)
  assert actual_sha==file_entry['observed_sha256'],(t,name,'catalog current file mismatch')
  old_sha=d.get('sha256',d.get('statement_ref',{}).get('sha256'))
  if actual_sha!=old_sha: drift.append({'file':rel,'declaration':name,'field':'sha256' if 'sha256' in d else 'statement_ref.sha256','catalog_value':old_sha,'actual_value':actual_sha})
  found=re.findall(r'(?m)^\s*specification\s+'+re.escape(name)+r'\b',data)
  assert len(found)==1
  at=data.find('specification '+name)
  line=data[:at].count('\n')+1
  old_line=d.get('line',d.get('statement_ref',{}).get('line'))
  if line!=old_line: drift.append({'file':rel,'declaration':name,'field':'line' if 'line' in d else 'statement_ref.line','catalog_value':old_line,'actual_value':line})
  public.append({'file':rel,'declaration':d['declaration'],'parser_name':name,'line':line,'sha256':sha(p),'statement_sha256':current['statement_sha256'],'statement':adapter.split_statement(current['text'])[0]})
 # Verify no actual public declaration in any catalog-public file is silently dropped.
 actual={(rel,name) for rel in {d['file'] for d in cat[t]['declarations']} for name,d in adapter.declarations(repo/rel).items() if d['kind']=='specification'}
 assert actual=={(d['file'],d['parser_name']) for d in public}
 for gap in note['structural_gaps']:
  assert gap['can_defer_as_numeric_assumption'] is False
  for e in gap['evidence']:
   rel=e.pop('file',e.get('path'));e['path']=rel;p=repo/rel;lines=p.read_text().splitlines();first=e['line'];last=e['end_line']
   assert 1<=first<=last<=len(lines),(t,e)
   assert rel in observed,(t,'unreviewed evidence file',rel)
   e['sha256']=observed[rel];e['excerpt']='\n'.join(lines[first-1:last])
 result={'schema':1,'kind':'original_scope_gap_review','reviewer':'/root/scope_remaining_cohort','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewed_base_commit':inv['reviewed_base_commit'],'catalog':{'path':str(cp.relative_to(repo)),'sha256':inv['catalog_sha256']}, **note,
  'source_sha256':observed,'all_input_sha256':all_inputs,'public_declarations_checked':public,'catalog_declaration_metadata_discrepancies':drift,
  'dtype_scope_caveat':'No unsupported dtype is assumed to compile merely because a Python wrapper admits it. Missing-route findings require either the actual supported conversion route or a precise admission/rejection contract; they do not demand arbitrary compiler internals.',
  'verification':{'manual_original_production_functions_read':len(row['production']),'actual_public_statements_reparsed_and_hash_checked':len(public),'key_source_and_referenced_definition_review':True,'source_matches_baseline_scope_inputs':True,'generated_selector_check':check_generated(t),'new_lean_runs':0,'new_frontend_runs':0,'gpu_runs':0,'numerical_deferrals_registered_by_this_read_only_review':0}}
 return result

label=sys.argv[1];names=sys.argv[2:];results=[freeze(t) for t in names]
out={'schema':1,'kind':'original_scope_batch_review','batch':6,'part':label,'reviewer':'/root/scope_remaining_cohort','reviewed_base_commit':inv['reviewed_base_commit'],'catalog_sha256':inv['catalog_sha256'],'source_ledger_snapshot_sha256':inv['ledger_snapshot_sha256'],'task_count':len(results),'structural_gaps_confirmed':len(results),'whole_original_program_accepted':0,'completion_increment':0,'public_declarations_checked':sum(len(r['public_declarations_checked']) for r in results),'structural_gap_count':sum(len(r['structural_gaps']) for r in results),'catalog_declaration_metadata_discrepancies':[{'task':r['task'],**d} for r in results for d in r['catalog_declaration_metadata_discrepancies']],'integration_note':'Read-only source scope audit. Existing partial proofs remain credited. No numerical deferrals or original-program acceptance registered. Root substantive review is required before integration.','tasks':results}
p=root/f'batch-06-{label}-review.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'tasks':len(results),'public_declarations':out['public_declarations_checked'],'gaps':out['structural_gap_count'],'metadata_discrepancies':len(out['catalog_declaration_metadata_discrepancies'])}))
