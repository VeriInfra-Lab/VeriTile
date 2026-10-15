"""Positive scopes and adversarial exact-binding/legacy compatibility checks."""
from copy import deepcopy
import importlib.util,json,shutil,sys,tempfile,unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'bench'),str(REPO/'scripts')]
from exact_numerical_bindings import (PACKAGE_PATHS,collect,derive_authority,
    validate_exact_queue,canonical,sha)
from deferred_numerical_experiments import collect_legacy
from reviewed_numerical_extensions import (RELU_FINITE_PACKAGE, RELU_FINITE_REVIEW,
    VECTOR_FINITE_PACKAGE, VECTOR_FINITE_REVIEW)
import check_original_program_scope as gate
legacy,authority,queue=collect(REPO)


def clone_inputs(destination):
 root=Path(destination)
 bindings=dict(authority['input_sha256'])
 bindings.update({'repo:'+p:h for p,h in legacy['input_sha256'].items()})
 for key in bindings:
  package,path=key.split(':',1);relative=Path(PACKAGE_PATHS[package])/path
  target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(REPO/relative,target)
 return root


def usage(task,selected=None):
 tables=[r for r in authority['requirements'] if r['task']==task]
 public={(r['declaration']['file'],r['declaration']['declaration']):{**r['declaration'],'unclosed_numeric_callbacks':r['callback_union']} for r in tables}
 proof_cases={};needed=set();callback_links={}
 for t in tables:
  key=(t['declaration']['file'],t['declaration']['declaration'])
  cases=[c for c in t['cases'] if selected is None or selected(t,c)]
  if not cases:continue
  proof_cases[key]={canonical({k:c[k] for k in ('dtype_profile','configuration')}) for c in cases}
  for c in cases:
   for ref in c['required_binding_refs']:
    ident=(ref['assumption_id'],ref['scope_binding_id']);needed.add(ident)
    callback_links.setdefault(ref['callback'],set()).add(ident)
 deferrals=[]
 for rule in authority['rules']:
  for b in rule['scope_bindings']:
   if (rule['id'],b['id']) in needed:
    deferrals.append({'assumption_id':rule['id'],'scope_binding_id':b['id'],'kind':'numerical_rule_conformance',
      'rule_sha256':sha(canonical(rule)),**{k:b[k] for k in ('dtype_profile','configuration')},'required_validation':rule['required_validation']})
 row={'deferred_numerical_assumptions':deferrals,'callback_assumptions':{cb:[{'assumption_id':i,'scope_binding_id':s} for i,s in sorted(refs)] for cb,refs in callback_links.items()}}
 return row,public,proof_cases


class ExactBindings(unittest.TestCase):
 def assert_reject(self,data):
  with self.assertRaises(ValueError):validate_exact_queue(data,authority,legacy)
 def test_complete_authority_and_legacy_unchanged(self):
  self.assertEqual(queue['rules'][:11],legacy['rules']);self.assertEqual(84,len(queue['rules']))
  self.assertEqual([29,20,9,340,2601],[len(r['cases']) for r in authority['requirements']])
  self.assertEqual([15,13,3,9,26],[len(r['callback_union']) for r in authority['requirements']])
  self.assertEqual(833,sum(len(b['declarations']) for r in authority['rules'] for b in r['scope_bindings']))
  result=validate_exact_queue(queue,authority,legacy)
  self.assertEqual(786,result['exact_bindings']);self.assertEqual(2999,result['declaration_cases'])
 def test_all_exact_usage_cases(self):
  for task in ('vector_addition','relu_triton_kernel'):
   row,p,c=usage(task);gate.validate_used_numerics(task,row,p,c,queue,root=REPO)
 def test_single_fp64_case_only_add64_required(self):
  row,p,c=usage('vector_addition',lambda t,c:t['declaration'] in [r['declaration'] for r in authority['requirements'][:2]] and c['dtype_profile']['inputs']=={'x':'fp64','y':'fp64'})
  self.assertEqual(1,len(row['callback_assumptions']));gate.validate_used_numerics('vector_addition',row,p,c,queue,root=REPO)
 def test_unsigned_and_bool_zero_do_not_require_literal(self):
  row,p,c=usage('relu_triton_kernel',lambda t,c:t['declaration']==authority['requirements'][2]['declaration'] and c['dtype_profile']['inputs']['x'] in {'uint8','bool'})
  self.assertEqual(1,len(row['callback_assumptions']));self.assertIn('fromNat',next(iter(row['callback_assumptions'])))
  gate.validate_used_numerics('relu_triton_kernel',row,p,c,queue,root=REPO)
 def test_signed_case_requires_both_guarded_paths(self):
  row,p,c=usage('relu_triton_kernel',lambda t,c:t['declaration']==authority['requirements'][2]['declaration'] and c['dtype_profile']['inputs']['x']=='int8')
  self.assertEqual(2,len(row['callback_assumptions']));gate.validate_used_numerics('relu_triton_kernel',row,p,c,queue,root=REPO)
 def test_missing_case_in_any_contract_rejected(self):
  for idx in range(len(authority['requirements'])):
   d=deepcopy(queue);d['exact_callback_requirements'][idx]['cases'].pop();self.assert_reject(d)
 def test_missing_callback_even_with_self_consistent_rehash_rejected(self):
  d=deepcopy(queue);r=d['exact_callback_requirements'][0]['cases'][0];r['required_callbacks'].pop();r['required_binding_refs'].pop()
  d['exact_callback_requirements'][0]['supported_cases_sha256']=sha(canonical(r));self.assert_reject(d)
 def test_extra_callback_rejected(self):
  d=deepcopy(queue);d['exact_callback_requirements'][2]['cases'][0]['required_callbacks'].append('native_allocator_freshness');self.assert_reject(d)
 def test_wrong_statement_and_qualified_name_rejected(self):
  for field,value in [('statement_sha256','0'*64),('declaration','wrong_theorem')]:
   d=deepcopy(queue);d['exact_callback_requirements'][0]['declaration'][field]=value;self.assert_reject(d)
 def test_wrong_dtype_rejected(self):
  d=deepcopy(queue);d['exact_callback_requirements'][0]['cases'][0]['dtype_profile']['inputs']['x']='fp8e4m3fnuz';self.assert_reject(d)
 def test_wrong_N_target_and_configuration_hash_rejected(self):
  for key,value in [('N',2048),('architecture','sm80'),('frontend_configuration_sha256','f'*64)]:
   d=deepcopy(queue);d['exact_callback_requirements'][2]['cases'][0]['configuration'][key]=value;self.assert_reject(d)
 def test_wrong_callback_source_hash_rejected(self):
  d=deepcopy(queue);next(iter(d['exact_binding_provenance'].values()))['callback_source']['sha256']='a'*64;self.assert_reject(d)
 def test_structural_rule_cannot_be_relabelled_numeric(self):
  d=deepcopy(queue);extra=deepcopy(d['rules'][-1]);extra['id']='native_byte_layout';extra['required_validation']='Numerically validate native allocation/layout';d['rules'].append(extra);self.assert_reject(d)
 def test_unsigned_zero_cannot_be_rebound_to_signed_literal(self):
  d=deepcopy(queue)
  row=next(x for x in d['exact_callback_requirements'][2]['cases'] if x['dtype_profile']['inputs']['x']=='uint8')
  row['required_callbacks']=[authority['requirements'][2]['callback_union'][2]];self.assert_reject(d)
 def test_path_predicate_cannot_be_widened_to_all_words(self):
  d=deepcopy(queue)
  row=next(v for v in d['exact_binding_provenance'].values() if v['scalar_path_predicate']=='signed_negative')
  row['scalar_path_predicate']='all_signed_words';self.assert_reject(d)
 def test_binding_declaration_or_port_omission_rejected(self):
  d=deepcopy(queue);d['rules'][11]['scope_bindings'].pop();self.assert_reject(d)
 def test_legacy_rule_mutation_rejected(self):
  d=deepcopy(queue);d['rules'][0]['status']='pending_numeric_validation';self.assert_reject(d)
 def test_unsupported_case_fails_scope_consumer(self):
  row,p,c=usage('relu_triton_kernel');key=next(iter(c));wrong=json.loads(next(iter(c[key])));wrong['configuration']['N']=2048;c[key]={canonical(wrong)}
  with self.assertRaisesRegex(ValueError,'unsupported dtype/configuration'):gate.validate_used_numerics('relu_triton_kernel',row,p,c,queue,root=REPO)
 def test_wrong_public_statement_fails_scope_consumer(self):
  row,p,c=usage('vector_addition');next(iter(p.values()))['statement_sha256']='a'*64
  with self.assertRaisesRegex(ValueError,'wrong exact numeric statement'):gate.validate_used_numerics('vector_addition',row,p,c,queue,root=REPO)
 def test_missing_used_callback_reference_fails_scope_consumer(self):
  row,p,c=usage('relu_triton_kernel');row['callback_assumptions'].pop(next(iter(row['callback_assumptions'])))
  with self.assertRaisesRegex(ValueError,'unclosed numerical callbacks'):gate.validate_used_numerics('relu_triton_kernel',row,p,c,queue,root=REPO)
 def test_missing_side_tables_do_not_fall_back_for_new_rules(self):
  d=deepcopy(queue)
  for k in ('exact_callback_requirements','exact_binding_provenance','exact_numerical_inputs','exact_numerical_summary'):d.pop(k)
  row,p,c=usage('vector_addition')
  with self.assertRaisesRegex(ValueError,'incomplete exact numeric'):gate.validate_used_numerics('vector_addition',row,p,c,d,root=REPO)
 def test_structural_gaps_still_block_before_any_numeric_claim(self):
  for gap in ('source_graph_correspondence','native_byte_layout','host_allocation'):
   with self.assertRaisesRegex(ValueError,'unresolved structural/scope gaps'):
    gate.validate_task(REPO,{}, {'task':'vector_addition'}, {'status':'verified_with_deferred_numerics','scope_schema':2,'structural_gaps':[gap]},queue)
 def test_actual_frozen_callback_source_change_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=clone_inputs(tmp)
   p=root/'bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean'
   source=p.read_text();needle='add64 : Word F64 → Word F64 → Word F64';self.assertIn(needle,source)
   p.write_text(source.replace(needle,'add64 : Word F64 → Word F64 → Word F32',1))
   with self.assertRaisesRegex(ValueError,'bound source/hash changed'):
    derive_authority(root)
 def test_actual_public_statement_source_change_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=clone_inputs(tmp)
   row=authority['requirements'][0]['source_ref'];p=root/row['path']
   source=p.read_text();needle='(separated : Separated p)';self.assertIn(needle,source)
   p.write_text(source.replace(needle,'(separated : True)',1))
   with self.assertRaisesRegex(ValueError,'bound source/hash changed: repo:bench/'):
    derive_authority(root)
 def test_changed_pinned_receipt_cannot_authorize_source_mutation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=clone_inputs(tmp)
   relative='bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean';p=root/relative
   source=p.read_text();needle='add64 : Word F64 → Word F64 → Word F64';self.assertIn(needle,source)
   p.write_text(source.replace(needle,'add64 : Word F64 → Word F64 → Word F32',1))
   receipt=root/PACKAGE_PATHS['fp64_core']/'evidence/receipt.json';data=json.loads(receipt.read_text());data['modules'][relative]=sha(p.read_bytes());receipt.write_text(json.dumps(data))
   with self.assertRaisesRegex(ValueError,'bound source/hash changed: fp64_core:evidence/receipt.json'):
    derive_authority(root)
 def test_actual_omitted_and_extra_candidate_ports_are_rejected(self):
  for extra in (False,True):
   with tempfile.TemporaryDirectory() as tmp:
    root=clone_inputs(tmp)
    p=root/PACKAGE_PATHS['fp64_grid']/'evidence/numerical-bindings.json';data=json.loads(p.read_text())
    if extra:data['instances'].append(deepcopy(data['instances'][0]))
    else:data['instances'].pop()
    p.write_text(json.dumps(data))
    with self.assertRaisesRegex(ValueError,'bound source/hash changed: fp64_grid:evidence/numerical-bindings.json'):
     derive_authority(root)
 def test_actual_hash_bound_legacy_input_change_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=clone_inputs(tmp)
   ext=json.loads((root/'bench/tritonbench_g/fp_value_extensions.json').read_text())
   p=root/ext['native_plan']['path'];p.write_text(p.read_text()+' ')
   with self.assertRaisesRegex(ValueError,'Changed bound input'):
    collect_legacy(root)
 def test_finite_empty_and_error_cases_require_no_numeric_assumptions(self):
  row,p,c=usage('relu_triton_kernel',lambda t,c:t['declaration']==authority['requirements'][3]['declaration'] and
    (c['configuration']['outcome']!='returned' or c['configuration']['size_class']=='empty'))
  self.assertEqual({},row['callback_assumptions']);self.assertEqual([],row['deferred_numerical_assumptions'])
  gate.validate_used_numerics('relu_triton_kernel',row,p,c,queue,root=REPO)
 def test_finite_repeated_scalar_port_cannot_be_omitted(self):
  row,p,c=usage('relu_triton_kernel',lambda t,c:t['declaration']==authority['requirements'][3]['declaration'] and
    c['dtype_profile']['inputs']['x']=='bf16' and c['configuration']['size_class']=='unsigned_constexpr_boundary' and
    c['configuration']['outcome']=='returned')
  gate.validate_used_numerics('relu_triton_kernel',row,p,c,queue,root=REPO)
  callback=next(k for k,v in row['callback_assumptions'].items() if len(v)==2)
  row['callback_assumptions'][callback].pop()
  with self.assertRaisesRegex(ValueError,'exact scalar port omitted'):
   gate.validate_used_numerics('relu_triton_kernel',row,p,c,queue,root=REPO)
 def test_changed_finite_helper_receipt_delta_or_review_rejected(self):
  for relative in [
    'bench/tritonbench_g/relu_triton_kernel/Proofs/FiniteIndices.lean',
    RELU_FINITE_PACKAGE+'/evidence/receipt.json',
    RELU_FINITE_PACKAGE+'/evidence/numerical-authority-delta.json',
    RELU_FINITE_REVIEW,
  ]:
   with tempfile.TemporaryDirectory() as tmp:
    root=clone_inputs(tmp);path=root/relative;path.write_bytes(path.read_bytes()+b' ')
    with self.assertRaisesRegex(ValueError,'changed reviewed numerical extension input'):
     derive_authority(root)
 def test_vector_empty_and_rejected_cases_need_no_numeric_assumptions(self):
  row,p,c=usage('vector_addition',lambda t,c:t['declaration']==authority['requirements'][4]['declaration'] and
    (c['configuration']['outcome']!='returned' or c['configuration']['size_class']=='empty'))
  self.assertEqual({},row['callback_assumptions']);self.assertEqual([],row['deferred_numerical_assumptions'])
  gate.validate_used_numerics('vector_addition',row,p,c,queue,root=REPO)
 def test_vector_each_repeated_callback_port_is_required(self):
  table=authority['requirements'][4]
  case=next(c for c in table['cases'] if len(c['required_binding_refs'])>len(c['required_callbacks']))
  row,p,c=usage('vector_addition',lambda t,x:t['declaration']==table['declaration'] and x==case)
  gate.validate_used_numerics('vector_addition',row,p,c,queue,root=REPO)
  callback=next(k for k,v in row['callback_assumptions'].items() if len(v)>1)
  row['callback_assumptions'][callback].pop()
  with self.assertRaisesRegex(ValueError,'exact scalar port omitted'):
   gate.validate_used_numerics('vector_addition',row,p,c,queue,root=REPO)
 def test_changed_vector_source_receipts_ports_or_review_rejected(self):
  for relative in [
    'bench/tritonbench_g/vector_addition/Proofs/FiniteIndices.lean',
    VECTOR_FINITE_PACKAGE+'/evidence/receipt.json',
    VECTOR_FINITE_PACKAGE+'/evidence/frontend/receipt.json',
    VECTOR_FINITE_PACKAGE+'/evidence/numerical-authority-delta.json',
    VECTOR_FINITE_REVIEW,
  ]:
   with tempfile.TemporaryDirectory() as tmp:
    root=clone_inputs(tmp);path=root/relative;path.write_bytes(path.read_bytes()+b' ')
    with self.assertRaisesRegex(ValueError,'changed reviewed vector input'):
     derive_authority(root)
 def test_exact_consumer_requires_every_port_ref(self):
  # Same-callback ports need independent references; the source-derived current
  # profiles happen to use one occurrence per callback, so exercise the generic
  # consumer directly without pretending to register a new production profile.
  key=('file','spec');case=b'case';refs=[{'callback':'add','assumption_id':'r','scope_binding_id':x} for x in ('x','y')]
  exact={key:{case:{'required_binding_refs':refs}}};applicable={('r','x'),('r','y')}
  with self.assertRaisesRegex(ValueError,'exact scalar port omitted'):
   gate.validate_exact_case_links(exact,key,case,applicable,{'add':{('r','x')}})
  gate.validate_exact_case_links(exact,key,case,applicable,{'add':applicable})

if __name__=='__main__':
 unittest.main(verbosity=2)
