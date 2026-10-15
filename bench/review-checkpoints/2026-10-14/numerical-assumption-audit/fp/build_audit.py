#!/usr/bin/env python3
from pathlib import Path
import ast,collections,hashlib,json,re,subprocess
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path('/tmp/veritile-assumptions-20261014/fp')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(rel, needle=None):
 p=ROOT/rel; d={'file':rel,'sha256':sha(p)}
 if needle:
  s=p.read_text(); pos=s.index(needle);d['line']=s[:pos].count('\n')+1
 return d
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
entries=json.loads((OUT/'initial.json').read_text())
base='report:8a3a9e065c6200713a9aba7138912d9fac7e69af9f5593b90f08372df77e1734:fp32:'
supp='report:9b01836a349a09e938f62e813c56a99aed2d0e39bf801fb5b050d57000b5604c:fp32:'
keymap={base+'ADD-COMMUTE':'fp32_add_commute_real',base+'MUL-COMMUTE':'fp32_mul_commute_real',supp+'DIV-MUL-RCP':'fp32_div_mul_rcp_real',base+'MUL-COMMUTE:typed-fp32':'fp32_mul_commute_typed'}
probe=(OUT/'probe.log').read_text()
assert 'FP assumptions used by seeded_dropout_equiv:\n  none' in probe and 'error:' not in probe
for k in keymap: assert json.dumps(k) in probe
identities={}
for e in entries:
    short=e['printed_atoms']
    if short is None:
        assert e['declaration']=='seeded_dropout_equiv'; short=['none']
    atoms=[]
    for index,atom in enumerate(short):
        if atom=='none': continue
        rid={'add_commute':'ADD-COMMUTE','mul_commute':'MUL-COMMUTE','div_mul_rcp':'DIV-MUL-RCP'}[atom]
        key=(supp if atom=='div_mul_rcp' else base)+rid
        if len(short)==2 and index==1:
            assert e['file'].split('/')[2] in ['context_attn_fwd','context_attn_llama'] and short==['mul_commute','mul_commute']
            key+=':typed-fp32'
        atoms.append({'rule_id':rid,'instance_key':key})
    identities[e['qualified_declaration']]=atoms
assert len(identities)==155
old=json.loads((ROOT/'bench/review-checkpoints/2026-10-14/acceptance-after-parallel/receipt.json').read_text())
assert all(sha(ROOT/f)==h for f,h in old['input_sha256'].items() if f.endswith('.lean'))
rows=[]
review=json.loads((ROOT/'bench/tritonbench_g/coverage_review.json').read_text())
ports={p['file']:p for p in review['ports']}
for e in entries:
 atoms=identities[e['qualified_declaration']]; p=ROOT/e['file']; src=p.read_text()
 start=re.search(r'^specification '+re.escape(e['declaration'])+r'\b',src,re.M).start(); end=src.index(':=',start)
 r={'file':e['file'],'declaration':e['declaration'],'qualified_declaration':e['qualified_declaration'],
    'task':p.parent.name,'numeric_model':'admitted_fp','coverage_level':e['coverage_level'],
    'classification':'numerical_rule_used' if atoms else 'no_extra_rule_within_model',
    'atoms':[dict(a,obligation_id=keymap[a['instance_key']]) for a in atoms],
    'obligation_ids':[keymap[a['instance_key']] for a in atoms],
    'statement_ref':dict(ref(e['file']),line=src[:start].count('\n')+1),
    'statement_source':src[start:end].rstrip(),'reviewed_scope':e['scope'],
    'python_functions':e['python_functions'],
    'proof_used_evidence':{'method':'Existing complete acceptance StatementAudit instantiated proof traversal; exact entry identities checked by Probe.lean and source branch inspection',
                          'query_source':'Probe.lean','log':'probe.log' if e['declaration']=='seeded_dropout_equiv' else 'bench/review-checkpoints/2026-10-14/acceptance-after-parallel/build.log',
                          'qualified_declaration':e['qualified_declaration'],'unresolved':False},
    'support_sha256':ports[e['file']]['support_sha256']}
 if atoms:
  # Preserve exact in-file Site definitions and domain selection, without claiming
  # that every definition in a shared file belongs to this one declaration.
  spans=[]; defs=list(re.finditer(r'^def ([\w?]+)\b',src,re.M))
  for idx,m in enumerate(defs):
   stop=defs[idx+1].start() if idx+1<len(defs) else len(src)
   stop=min([x for x in [stop,src.find('\nspecification ',m.start()),src.find('\nend',m.start())] if x>=0])
   text=src[m.start():stop].rstrip()
   if re.search(r': (?:List )?Site\b|\bdomain :=',text):
    spans.append({'name':m[1],'line':src[:m.start()].count('\n')+1,'source':text})
  r['in_file_domain_definition_refs']=spans
  r['domain_interpretation']='Finite/nonzero predicates apply to actual operands after the original execution prefix, at every reached loop iteration and every evaluated lane, including eager masked/where lanes. Full per-declaration scope and source definitions are retained above.'
 if r['task'] in {'context_attn_fwd','context_attn_llama'}:
  r['storage_binding_selection']={'fp16':'fp32_mul_commute_typed','bf16':'fp32_mul_commute_typed','fp32':'fp32_mul_commute_real'}
 rows.append(r)
assert sum(not r['atoms'] for r in rows)==92
assert sum(bool(r['atoms']) for r in rows)==63
usedcounts=collections.Counter(o for r in rows for o in r['obligation_ids'])
assert dict(usedcounts)=={'fp32_add_commute_real':34,'fp32_div_mul_rcp_real':10,'fp32_mul_commute_real':19,'fp32_mul_commute_typed':2},usedcounts

# Read-only validation through the repository exporters rejects WARN/status/budget errors.
from scripts import export_numerical_rules as expbase,export_supplemental_rules as expsupp
reports={}
for name,mod in [('base',expbase),('supplement',expsupp)]:
 settings,profile,accepted,hashes,snapshot=mod.load_report(mod.REPORT)
 assert mod.render()==mod.OUTPUT.read_text()
 rel=str(mod.REPORT.relative_to(ROOT)); summary=json.loads((mod.REPORT/'summary.json').read_text())
 source_checks=[]
 for file,expected in settings['sources'].items():
  p=ROOT/file; actual=sha(p) if p.is_file() else None
  item={'file':file,'report_sha256':expected,'current_sha256':actual,'current_matches':actual==expected}
  frozen=OUT/'frozen-sources'/file
  if actual!=expected:
   assert frozen.is_file() and sha(frozen)==expected
   item['frozen_copy']='frozen-sources/'+file;item['frozen_sha256']=sha(frozen)
  source_checks.append(item)
 reports[name]={'directory':rel,'experiment':ref(rel+'/experiment.json'),'summary':ref(rel+'/summary.json'),
   'snapshot':snapshot,'profile':profile,'backend':settings['backend'],'hardware':settings.get('hardware',settings.get('execution',{}).get('hardware_model')),
   'report_validation':'PASS','generated_lean_matches_report':True,'accepted_rows':len(accepted),
   'source_identity':source_checks,'row_by_rule':{r['rule']:r for r in summary['rows'] if r['format']=='fp32'},
   'raw_gpu_samples_available_in_this_repo':False,'independent_gpu_or_sample_replay_this_audit':False,
   'published_replay_flag_is_report_data_not_new_replay':True,
   'report_trust_boundary':'Existing user-trusted published aggregate; raw observations/PTX remain outside Git. This audit verifies report integrity, status, budgets, provenance and exact Lean export, not original GPU execution.'}

standard_task={
 'input_format':'fp32','compute_format':'fp32','accumulator_format':'fp32','output_format':'fp32',
 'domain':'finite a and finite b','preserve':['actual arithmetic operation and operand order','fp32 output before surrounding casts','no implicit FMA contraction','actual reached-site operands, including eager masked/where lanes'],
 'original_report_profile':{'shape':[4096,4096],'block':1024,'num_warps':4,'replicates':4096,'distribution':'independent Normal(mean=1.0,std=1.0)','enable_fp_fusion':False,'compiler_triton':'3.7.1','compiler_cuda':'13.0','runtime_device':'NVIDIA L20X','capability':[9,0],'hardware_label':'H200'},
 'suggested_additional_cases':['signed zero','smallest and largest subnormal','smallest normal','large finite magnitudes','overflowing finite-input products or sums','values adjacent to rounding ties','representative actual kernel operand distributions'],
 'record':['source/AST/PTX hash','GPU architecture and driver','Triton/CUDA versions','launch and FP-fusion options','input/output/compute dtype','raw samples and input seeds','finite/nonzero domain violations','bitwise differences separately from ULP/error statistics','bias and magnitude gates plus independent replay'],
 'formal_limit':'Finite experiments can support scoped empirical admission or expose counterexamples; they do not prove universal IEEE equality or statistical-gate composition. A hardware-refinement proof is separate from these rule-model theorems.'}
obligations=[]
specs=[
 ('fp32_add_commute_real','ADD-COMMUTE','base','addCommute','Q32(a+b)','Q32(b+a)', 'real', 'ScalarArithmetic.Rules.validated .addCommute', 'Existing PASS row, explicit real-channel/fp32 binding. No new typed-channel rule is used.'),
 ('fp32_mul_commute_real','MUL-COMMUTE','base','mulCommute','Q32(a*b)','Q32(b*a)', 'real','ScalarArithmetic.Rules.validated .mulCommute','Existing PASS row, explicit real-channel/fp32 binding. The old scalar rule does not imply typed-fp32 multiplication.'),
 ('fp32_div_mul_rcp_real','DIV-MUL-RCP','supplement','divMulRcp','Q32(a/b)','Q32(a*Q32(1/b))','real','ScalarArithmetic.Rules.validated .divMulRcp','Existing PASS row for ordinary Triton /; it is a separate report and operation from tl.div_rn DIV-RCP.'),
 ('fp32_mul_commute_typed','MUL-COMMUTE','base',None,'Q32(a*b)','Q32(b*a)','fp32','TypedScalarArithmetic.Rules.validatedTyped','Published row reused as provenance, but a distinct :typed-fp32 evidence binding remains an explicit new premise.')]
for oid,rid,rep,atom,lhs,rhs,channel,premise,note in specs:
 report=reports[rep]; row=report['row_by_rule'][rid]; task=json.loads(json.dumps(standard_task));
 if rid=='DIV-MUL-RCP':
  task['domain']='finite a and finite b and b != 0; no implicit requirement that reciprocal/output are finite'
  task['preserve'].append('ordinary Triton /, not tl.div_rn; retain separately evaluated reciprocal')
  task['suggested_additional_cases']+=['positive/negative subnormal denominator','reciprocal overflow/underflow','finite numerator and denominator with nonfinite quotient']
 if channel=='fp32':
  task['native_context']='context_attn_fwd/context_attn_llama fp16 or bf16 storage -> dot produces actual typed fp32 qk; multiply typed fp32 qk by typed fp32 scale before real-carrier cast; guard every eager where lane.'
  task['required_followup']='Validate and record the exact :typed-fp32 binding using actual fp32 operands/output; old real-channel validation is insufficient. Existing offline compilation is type/source evidence only, on Triton 3.8.0 rather than the 3.7.1 numerical report.'
 else:
  task['required_followup']='No absent aggregate row. If applying to a new target compiler/device or claiming native whole-kernel behavior, replay the exact probe/configuration and validate the real-carrier/fp32 interpretation and actual kernel site lowering; retain the model premise explicitly.'
 oidkey=next(k for k,v in keymap.items() if v==oid)
 srcs=[ref('VeriTile/Triton/Float/ScalarArithmetic.lean','def guards'),ref('VeriTile/Triton/Float/ScalarArithmetic.lean','def lhsCode'),ref('VeriTile/Triton/Float/ScalarArithmetic.lean','structure Rules'),ref('VeriTile/Triton/Float/ProfiledRewrite.lean','def Site.Holds'),ref('VeriTile/Triton/Float/ProfiledRewrite.lean','def Equivalent'),ref('VeriTile/Spec.lean','opaque EvidenceValidated')]
 if channel=='fp32':srcs=[ref('VeriTile/Triton/Float/TypedScalarArithmetic.lean','def guards'),ref('VeriTile/Triton/Float/TypedScalarArithmetic.lean','structure Rules'),ref('bench/tests/TypedFP32Mul.lean','theorem old_rules_insufficient')]+srcs[-3:]
 if rid=='DIV-MUL-RCP':srcs.append(ref('VeriTile/Triton/Float/Reciprocal.lean','def lhs'))
 obligations.append({'id':oid,'rule_id':rid,'instance_key':oidkey,'used_by_declarations':usedcounts[oid],
  'used_by_tasks':sorted({r['task'] for r in rows if oid in r['obligation_ids']}),
  'lhs':lhs,'rhs':rhs,'lean_carrier_channel':channel,'lean_compute_precision':'some .fp32',
  'lean_output_channel':channel,'fragment_default_precision':'some .fp32' if channel=='fp32' else 'none (each arithmetic operation explicitly carries .fp32)',
  'guards':[{'operand':'a','kind':'finite'},{'operand':'b','kind':'finite'}]+([{'operand':'b','kind':'nonzero'}] if rid=='DIV-MUL-RCP' else []),
  'premise':premise,'exact_entry':('VeriTile.Triton.FP.TypedScalarArithmetic.entry' if channel=='fp32' else f'VeriTile.Triton.FP.ScalarArithmetic.entry .{atom} (by decide)'),
  'existing_report':{'report_id':rep,'summary_path':report['directory']+'/summary.json','row':row,'integrity_status':'PASS','row_status':'PASS','new_replay_performed':False},
  'binding_status':'explicit_typed_binding_validation_pending' if channel=='fp32' else 'published_row_bound_under_explicit_scoped_model_premise',
  'native_kernel_equality_proved':False,'note':note,'source_refs':srcs,'experiment_task':task})

# Exact original and current ordinary-division branch ASTs, helpers and specialization.
def specialize_rule(tree,rule):
 class S(ast.NodeTransformer):
  def visit_If(self,node):
   t=node.test
   if isinstance(t,ast.Compare) and isinstance(t.left,ast.Name) and t.left.id=='RULE' and len(t.ops)==1 and isinstance(t.ops[0],ast.Eq) and isinstance(t.comparators[0],ast.Constant):
    chosen=node.body if t.comparators[0].value==rule else node.orelse
    result=[]
    for c in chosen:
     v=self.visit(c);result.extend(v if isinstance(v,list) else [v])
    return result
   return self.generic_visit(node)
 return S().visit(tree)
krel='experiments/floating_point/supplement/kernels.py'
def projected_kernel(path):
 t=specialize_rule(ast.parse(path.read_text()),'DIV-MUL-RCP')
 funcs=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in {'rnd','elementwise'}]
 return ast.dump(ast.Module(body=funcs,type_ignores=[]),include_attributes=False)
a=projected_kernel(ROOT/krel);b=projected_kernel(OUT/'frozen-sources'/krel)
assert a==b
projection={'rule':'DIV-MUL-RCP','projection':'constant RULE selection, exact AST of rnd + elementwise including prefix, branch, output store, signature and decorators','current_source_sha256':sha(ROOT/krel),'frozen_source_sha256':sha(OUT/'frozen-sources'/krel),'projected_ast_sha256':hashlib.sha256(a.encode()).hexdigest(),'match':True}

native_rel='bench/review-checkpoints/2026-10-14/typed-fp32-native';native=ROOT/native_rel
nr=json.loads((native/'receipt.json').read_text()); native_checks=[]
for f,h in nr['artifact_sha256'].items():
 p=native/f; actual=sha(p) if p.is_file() else None
 native_checks.append({'file':f,'expected_sha256':h,'actual_sha256':actual,'status':'MATCH' if actual==h else ('MISSING_RUNTIME_CACHE' if '__pycache__/' in f and actual is None else 'FAIL')})
assert not any(x['status']=='FAIL' for x in native_checks)
native_info={'receipt':ref(native_rel+'/receipt.json'),'artifact_checks':native_checks,'source_type_mapping_passed':True,'gpu_executed':False,'samples_replayed':False,'compiler_version':'3.8.0','target':'cuda sm90','numerical_report_compiler_version':'3.7.1','scope':'Two original MUL-COMMUTE source sides compile to fp32 arith.mulf with reversed operands; does not discharge EvidenceValidated or whole-native-kernel equality.'}

report_catalog=[]
for p in sorted((ROOT/'experiments/floating_point').rglob('experiment.json')):
 rel=str(p.relative_to(ROOT)); d=json.loads(p.read_text());summary=p.parent/'summary.json'
 report_catalog.append({'experiment':ref(rel),'summary':ref(str(summary.relative_to(ROOT))) if summary.is_file() else None,
  'selected_by_this_155_declaration_audit':rel in [reports['base']['directory']+'/experiment.json',reports['supplement']['directory']+'/experiment.json'],
  'scope_note':'Only actual proof-used atoms determine selection; exp/log/count and other Rules fields are not charged to these headlines.'})
excluded={'not_used_by_these_headlines':['ADD-ASSOC','MUL-ASSOC','MUL-DISTRIB','CANCEL','ADD-ZERO','MUL-ONE','DIV-ONE','MUL-RCP-CANCEL','FMA-CONTRACT','EXP-*','LOG-*','COUNT-*','MAX-*','ROUND-IDEM','CAST-*','ACC-WIDEN'],
 'do_not_substitute_div_rn_for_ordinary_division':{'row':reports['base']['row_by_rule']['DIV-RCP'],'status':'WARN_NOT_ACCEPTED','source':ref('experiments/floating_point/kernels.py','elif RULE == "DIV-RCP"')},
 'existing_counterexamples_not_used_atoms':{'source':ref('experiments/floating_point/report/exact_counterexamples.json'),'rule_ids':['MUL-DISTRIB','FMA-CONTRACT','CAST-REMOVE','ACC-WIDEN'],'scope':'Counterexamples refute unconditional exact equality for those rules; none is a used atom in this 155-headline set.'},
 'typed_countermodel':{'source':ref('bench/tests/TypedFP32Mul.lean','theorem old_rules_insufficient'),'scope':'Proves old real-channel rules do not entail typed-fp32 commutativity; this is an abstract-model separation, not a GPU counterexample to fp32 MUL-COMMUTE.'}}
summary={'public_admitted_fp_declarations':155,'tasks':len({r['task'] for r in rows}),'no_extra_rule_within_model':92,'numerical_rule_used':63,'distinct_exact_bindings':4,'binding_occurrences':dict(usedcounts),'tasks_with_used_numerical_rule':len({r['task'] for r in rows if r['atoms']}),'unresolved_declarations':0,'existing_report_rows_selected':3,'exact_typed_binding_pending_declarations':2,'all_lean_inputs_match_prior_complete_acceptance':3379,'fp_assumption_unit_tests':{'tests':12,'status':'PASS'},'gpu_experiments_run_by_this_audit':0}
source_refs=[ref('VeriTile/Meta/StatementAudit.lean'),ref('scripts/test_fp_assumptions.py'),ref('bench/review-checkpoints/2026-10-14/acceptance-after-parallel/build.log'),ref('bench/review-checkpoints/2026-10-14/acceptance-after-parallel/receipt.json')]
result={'schema':1,'summary':summary,'rows':rows,'obligations':obligations,'reports':reports,'report_catalog':report_catalog,'source_refs':source_refs,'source_projection':projection,'typed_native_evidence':native_info,'excluded_or_nonpromotable_evidence':excluded,
 'method':{'numerical_dependency_basis':'Actual instantiated Derivation.atom nodes; all reachable proof branches retained, contract+evidence deduplicated exactly as StatementAudit. No use inferred from Rules package fields.','fresh_query':'Probe.lean passed for missing seeded_dropout output and four exact entry keys; optional large fresh queries retained separately with their actual status','proof_walker_changes':'Only output changed to rule_id+instance_key; Boolean constructors qualified to avoid imported DSL keywords.','existing_build_crosscheck':'154 of 155 concise per-declaration outputs matched complete acceptance log. seeded_dropout omitted the print command, so queried directly; exact instance keys are identified by fixed entry definitions, fresh Lean evaluation, and context storage branches; a redundant all-155 fresh query was stopped to avoid repeating expensive traversals.','not_claimed':['minimal semantic dependency','GPU replay','universal IEEE equality','whole-kernel statistical gate result','proof of native compiler/backend refinement']}}
save('audit.json',result);save('rows.json',rows);save('obligations.json',obligations);save('summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
