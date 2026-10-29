from pathlib import Path
import json,re,hashlib,collections,sys
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile'); OUT=Path('/tmp/veritile-assumptions-20261014/nonfp');sys.path.insert(0,str(ROOT/'scripts'))
from spec_sheet import strip_comments
rows=json.loads((OUT/'inventory.json').read_text()); predicates=json.loads((OUT/'predicates.json').read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evidence(file,line=1,description=''):
 p=ROOT/file
 return dict(file=file,line=line,sha256=digest(p),description=description)
def affected(r):return dict(file=r['file'],declaration=r['declaration'],line=r['line'],sha256=digest(ROOT/r['file']))
def binders(statement):
 s=strip_comments(statement);depth=0;groups=[];start=0
 for j,c in enumerate(s):
  if c in '({[':
   if depth==0:start=j
   depth+=1
  elif c in ')}]':
   depth-=1
   if depth==0:groups.append(s[start:j+1])
  elif c==':' and depth==0:break
 return groups
round_rows=[r for r in rows if re.search(r'\(\w+ : RoundingModel\)',r['statement'])]
identity_rows=[r for r in rows if 'hfp16 : R.round .fp16 = id' in r['statement']]
bgmv=[r for r in rows if 'native_add' in r['statement']]
empty=[r for r in rows if r['numeric_model']=='abstract_rounding' and '≡[' in r['statement']]
base_ev=[evidence('VeriTile/Triton/Float/RoundingModel.lean',33,'RoundingModel structure fields'),evidence('VeriTile/Triton/Float/EvalOpR.lean',26,'Only casts are rounded in this expression evaluator; primitive arithmetic retains the mathematical interpretation.')]
family_identity=dict(id='ROUND-FP16-ID',kind='explicit_theorem_law',formula='R.round .fp16 = id; equivalently ∀ x : ℝ, R.round .fp16 x = x',dtypes=['fp16'],precision='all real inputs, not merely representable fp16 grid points',domain='∀ x : ℝ',status='refuted_for_binary16_round_to_nearest_even; valid only as an idealized-model premise',affected=[affected(r) for r in identity_rows],evidence=base_ev+[dict(file=str(OUT/'fp16-identity-counterexample.json'),sha256=digest(OUT/'fp16-identity-counterexample.json'),description='Host binary16 reference witness: 2049/2048 rounds to 1, a finite fp32-exact input. Not a GPU result.')],experiment_question='Do not try to confirm the global identity by sampling. Measure actual intermediate cast operands and show representability only on a precisely stated domain, or change the theorem to retain real rounding. Existing global identity cannot be discharged by a successful finite sample.',repair_needed='Replace the global identity premise by a rounding-aware conclusion/proof or prove an explicitly restricted operand representability domain; no such Lean repair is claimed by this audit.')
family_native=dict(id='BGMV-NATIVE-ADD-COMMUTE',kind='explicit_theorem_law',formula='∀ site ∈ NativeFP.requirementsFor kernel, ∀ a b, D .finite a → D .finite b → (FP.ProfiledRewrite.engine M).binary site.profile.1 site.profile.2 .add a b = (FP.ProfiledRewrite.engine M).binary site.profile.1 site.profile.2 .add b a',dtypes='Actual source-selected native dtype; see profiles.json for 72 configurations',precision='Actual source-selected Option ComputeDType; output fp32 is proved (some .fp32, .real)',domain='Finite actual reduction result and loaded old output at each reached ADD_INPUTS branch, through the original dynamic output loop; evaluated masked lanes are included. No condition on unreachable branches.',status='explicit_pending_native_binding; existing fp32 Rules can discharge the fp32 profile conditionally',affected=[affected(r) for r in bgmv],evidence=[evidence('bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean',20,'ScalarAdditionCommutes exact formula'),evidence('bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean',44,'Site.Holds follows reached branches/ranges and evaluated operands'),evidence('bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean',110,'NativeAdditionCommutes requests only source-selected addition profiles'),evidence('bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean',428,'fp32_profiles and fp32_native_of_rules'),evidence('bench/tritonbench_g/bgmv_expand_slice/Proofs/AdditionProfiles.lean',1,'bf16 experiment uses widened fp32 and explicit rounding, not native bf16; fp16/fp64 unsupported'),evidence('bench/tritonbench_g/bgmv_expand_slice/Proofs/GeneralCountermodel.lean',1,'Logical countermodel shows fp32-only rules cannot establish native fp16 addition')],experiment_question='For each selected dtype/precision profile, run actual compiled native add(a,b) and add(b,a), capture the actual reached reduction/output operands and check finiteness, compare the observation used by the backend model (including signed-zero policy), and retain compiler/architecture/options and binary hashes. bf16 widened-fp32 emulation does not alone establish native bf16 correspondence; fp16 needs its own evidence.')
family_round=dict(id='ROUNDING-MODEL-STRUCTURE',kind='shared_model_requirement',formula=['R.round_real : R.round .real = id','R.round_idem : ∀ dtype x, R.round dtype (R.round dtype x) = R.round dtype x'],dtypes=['real','fp32','fp16','bf16','f8e4','f8e5'],precision='per FloatDType cast/store function',domain='RoundingModel round : FloatDType → ℝ → ℝ, all x : ℝ; NaN/Inf are not a separate part of this codomain',status='required_by_structure; no concrete hardware model supplied here; not 33 missing per-kernel experiments',affected=[affected(r) for r in round_rows],evidence=base_ev,experiment_question='A concrete rounding model must provide the idempotence proof/correspondence for each selected format and its handling of overflow/subnormal modes. An empirical check can recast all representable finite values, with boundary/underflow/overflow probes; finite sampling is not a proof of the quantified real-valued field.',notes='round_real concerns the mathematical channel only, not actual fp32/fp64 hardware. MonoRounding, OddRounding and GridNested exist as optional mixins but none occurs in these 474 public theorem signatures or inherited numerical variable blocks.')
declarations=[]
for r in rows:
 gs=binders(r['statement']);ins=[];stages=[];laws=[];models=[];oracle=[]
 for g in gs:
  if ' : ' not in g and ':' not in g:continue
  names=g[1:].split(':',1)[0].strip().split()
  ish=any(n.startswith('h') or n.startswith('_h') or n in ['native_add','finite_operands'] for n in names)
  if not ish:continue
  if 'R.round .fp16 = id' in g:laws.append('ROUND-FP16-ID')
  elif 'NativeFP.NativeAdditionCommutes' in g:laws.append('BGMV-NATIVE-ADD-COMMUTE')
  elif any(n in ['hPrev','hNext','hK','hV'] for n in names) and 's.readMem' in g and r['port'] in ['fused_recurrent_delta','fused_recurrent_hgrn','lightning_attention']:stages.append(g)
  elif '.undef' in g:oracle.append(g)
  else:ins.append(g)
 if r in round_rows:models=['ROUNDING-MODEL-STRUCTURE']
 source_ev=[evidence(r['file'],r['line'],'Actual public statement inspected; full text retained in inventory.json and statement_sha256 binds it.')]
 ev=source_ev[:]
 if r in empty:
  if r['declaration']=='ksoftmax_forward_index_equiv': ef='bench/tritonbench_g/ksoftmax_triton/Proofs/ForwardFP.lean';el=20
  else:
   ef=r['file'];lines=(ROOT/ef).read_text().splitlines();el=next(i+1 for i,l in enumerate(lines) if re.search(r'def assumptions.*:= \[\]',l))
  ev.append(evidence(ef,el,'Exact numerical atom list is []; no extra numerical law.'))
 # Referenced public definitions are retained as actual source locations, not promoted into hypotheses.
 for ref in r['references']:
  ev.append(evidence(r['file'],ref['line'],'Referenced definition: '+ref['name']))
 extra='explicit_numerical_law' if laws else 'shared_rounding_model_only' if models else 'empty_numerical_assumptions' if r in empty else 'no_extra_numerical_law_in_reviewed_contract'
 declarations.append(dict(**affected(r),port=r['port'],numeric_model=r['numeric_model'],statement_sha256=hashlib.sha256(r['statement'].encode()).hexdigest(),explicit_law_ids=laws,model_requirement_ids=models,accepted_input_conditions=ins,accepted_stage_conditions=stages,oracle_model_conditions=oracle,review_status=extra,evidence=ev,native_semantics_boundary=('mathematical_model_not_hardware_refinement' if r['numeric_model']=='mathematical' else 'cast_store_rounding_model_not_IEEE_arithmetic' if models else 'arbitrary_backend_equality_conditioned_on_displayed_native_law' if laws else 'structural_rewrite_with_empty_numerical_atoms'),notes='Normal size/layout/alias/bounds/finite operands and declared stage input contracts are retained; they are not experiment debt. This source-level audit does not claim a new elaborated-environment dependency replay.'))
# Record actual precomputed/carry predicates and known nonnumerical wrappers, to make negative classification checkable.
wrapper_files={
 'bench/tritonbench_g/softmax_optimize/Proofs/Geometry.lean':{'NumericalDomain'},
 'bench/tritonbench_g/matmul_persistent_triton/Proofs/RealIO.lean':{'Admissible','Implements'},
 'bench/tritonbench_g/softmax_flaggems/Proofs/RealForwardNonInnerAll.lean':{'ExecutionDomain'},
 'bench/tritonbench_g/softmax_flaggems/Proofs/RealForwardInnerAll.lean':{'ExecutionDomain'},
 'bench/tritonbench_g/softmax_flaggems/Proofs/RealBackwardNonInnerAll.lean':{'ExecutionDomain'},
 'bench/tritonbench_g/softmax_flaggems/Proofs/RealBackwardInnerAll.lean':{'ExecutionDomain'},
 'bench/tritonbench_g/llama_ff_triton/Proofs/FP8Arrays.lean':{'Inputs'},
 'bench/tritonbench_g/llama_ff_triton/Proofs/Real.lean':{'Inputs'},
 'bench/tritonbench_g/rms_matmul_rbe/Proofs/FP8Arrays.lean':{'Inputs'},
 'bench/tritonbench_g/rms_rbe_matmul/Proofs/FP8Arrays.lean':{'Inputs'},
}
wrapper_reviews=[]
for p in predicates:
 if p['name'] in wrapper_files.get(p['file'],set()):wrapper_reviews.append(dict(**evidence(p['file'],p['line']),name=p['name'],definition=p['text'],classification='normal_input_or_representation_condition',explanation='Size/alias/allocation constraints or explicit matching of independent arrays with initial input memory; no equality between a computed output and its desired result is assumed.'))
boundaries=[
 dict(id='REAL-ARITHMETIC-BOUNDARY',kind='native_refinement_boundary_not_added_theorem_hypothesis',formula='NumericDType add/sub/mul/div on all floating tags use Option.map₂ of exact ℝ operations; FloatDType.cast preserves WithBot ℝ through dtype tags.',evidence=[evidence('VeriTile/Triton/Semantics/Scalar.lean',46),evidence('VeriTile/Triton/Semantics/Scalar.lean',91),evidence('VeriTile/Triton/Semantics/Scalar.lean',341)],effect='The 429 mathematical statements are formal real-model results, not exact GPU floating results. Do not turn every such result into a new missing experiment; retain a separate semantics/refinement axis.'),
 dict(id='TRANSCENDENTAL-BOUNDARY',kind='native_refinement_boundary_not_added_theorem_hypothesis',formula=['realExp (some x) = some (Real.exp x)','realLog (some x) = some (Real.log x)','realSqrt (some x) = some (Real.sqrt x)','realRsqrt (some x) = some (1 / Real.sqrt x)','realPow (some a) (some b) = some (Real.rpow a b)'],evidence=[evidence('VeriTile/Triton/Semantics/TileOps.lean',154),evidence('VeriTile/Triton/Semantics/TileOps.lean',226)],effect='These are definitions of the mathematical model, not unproved exp/log/sqrt laws in the reviewed signatures. Approximation errors, exceptional IEEE inputs, fast-math and instruction correspondence need a separate hardware refinement study.'),
 dict(id='INT8-QUANTIZER-BOUNDARY',kind='native_refinement_boundary_not_added_theorem_hypothesis',formula='realToInt8 (some x) = if 0 ≤ x then Int.floor x else -Int.floor (-x); tritonInt8CastInRange x := -128 ≤ trunc(x) ∧ trunc(x) ≤ 127',evidence=[evidence('VeriTile/Triton/Semantics/Scalar.lean',57)],effect='The value model uses unbounded Int after truncation. Width overflow/saturation/wraparound must be scoped separately; do not call a pre-rounding/scaled-store theorem a completed int8 conversion proof.'),
 dict(id='PRECOMPUTED-STAGE-CONTRACTS',kind='accepted_stage_scope_not_numeric_law',affected=[affected(r) for r in rows if r['port'] in ['fused_recurrent_delta','fused_recurrent_hgrn','lightning_attention','fused_layernorm_triton','layer_norm_welfold']],evidence=[evidence(r['file'],r['line']) for r in rows if r['port'] in ['fused_recurrent_delta','fused_recurrent_hgrn','lightning_attention'] and ('hPrev' in r['statement'])],effect='hPrev/hNext and prepared KTrans/VTile refer to a declared local stage input. User accepts stage scope. In particular fused_recurrent_delta hNext is a supplied post-update carry and not discharged merely by an adjacent state-update clause; do not promote the stage theorem to end-to-end.'),
 dict(id='UNSPECIFIED-PADDING-STATE',kind='input_model_boundary_not_algebraic_numeric_law',affected=[affected(r) for r in rows if '.undef' in r['statement']],formula='s.undef = (fun _ _ => 0), or ∀ rg off, s.undef rg off = 0',effect='50 public statement texts mention an undef condition. Some are the finite-flat bridge convention, some select a zero oracle for masked/boundary loads. This is not an addition/rounding experiment. For hardware use preserve the zero-oracle profile or separately show undefined lanes are unobserved; no blanket arbitrary-undef claim is made. This textual count excludes conditions hidden only inside named IO definitions.'),
]
summary=dict(public_non_admitted_fp_declarations=len(rows),tasks=len({r['port'] for r in rows}),numeric_models=dict(collections.Counter(r['numeric_model'] for r in rows)),explicit_law_declarations=len(identity_rows)+len(bgmv),identity_round_declarations=len(identity_rows),native_add_declarations=len(bgmv),shared_rounding_model_declarations=len(round_rows),empty_numerical_atom_declarations=len(empty),per_declaration_status=dict(collections.Counter(r['review_status'] for r in declarations)),predicate_candidates_scanned=len(predicates),unknown_additional_numeric_laws_found=0)
obj=dict(schema=1,basis='Actual 629-row accepted contract inventory filtered to 474 mathematical/abstract_rounding declarations; all explicit binders reviewed, referenced numerical structures traced. Current source hashes retained. Not an elaborated environment comparator replay.',counts=summary,hypothesis_families=[family_identity,family_native,family_round],declarations=declarations,wrapper_reviews=wrapper_reviews,native_semantics_boundaries=boundaries,coverage_evidence=dict(inventory_sha256=digest(OUT/'inventory.json'),predicate_candidates_sha256=digest(OUT/'predicates.json'),hypotheses_sha256=digest(OUT/'hypotheses.txt')),unknowns=['Native operator/refinement evidence remains outside this source-contract audit.','Zero-undef hidden behind every IO wrapper was not exhaustively counted; the 50 count is explicitly direct statement occurrence only.'])
(OUT/'audit.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
