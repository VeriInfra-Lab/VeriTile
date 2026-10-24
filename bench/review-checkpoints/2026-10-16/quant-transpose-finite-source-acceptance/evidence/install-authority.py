from pathlib import Path
import ast,json,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'quant-transpose-finite-acceptance-revision1-wip';V=T/'quant-transpose-revision1-root-validation';B=R/'bench/review-checkpoints/2026-10-16/quant-transpose-finite-source-acceptance'
sys.path.insert(0,str(R/'bench'));from adapter_common import hash_file,load
assert hash_file(V/'review.json')=='9e0c4b360363f94670bc2477b8783477d4a93aa89ac93a2e48b050b066e87f77';assert (B/'evidence/root-validation/review.json').read_bytes()==(V/'review.json').read_bytes()
p=R/'bench/quant_transpose_numerical_extension.py';assert not p.exists();s=(P/'authority_adapter.py').read_text();old="    for relative, expected in receipt['artifacts'].items():\n        bind(package_relative / relative, expected)";new="    for relative, expected in receipt['artifacts'].items():\n        # Historical Lean source is archived with exact bytes, never registered as active code.\n        archived = relative + '.txt' if relative.endswith('.lean') else relative\n        bind(package_relative / archived, expected)";assert old in s;s=s.replace(old,new);p.write_text(s)
f=R/'bench/reviewed_numerical_extensions.py';s=f.read_text();old="    return _extend_quant_global_authority(root, result)";assert s.count(old)==1
new="""    result = _extend_quant_global_authority(root, result)
    from quant_transpose_numerical_extension import extend_authority
    return extend_authority(
        root, result, QUANT_TRANSPOSE_PACKAGE, QUANT_TRANSPOSE_RECEIPT,
        QUANT_TRANSPOSE_REVIEW, QUANT_TRANSPOSE_REVIEW_SHA256,
    )"""
s=s.replace(old,new)
constants="""QUANT_TRANSPOSE_PACKAGE = (
    'bench/review-checkpoints/2026-10-16/quant-transpose-finite-source-acceptance/'
    'evidence/producer/quant-transpose'
)
QUANT_TRANSPOSE_RECEIPT = '63392bac5342812d467c2bc499e7870e2609d49fa74faaf01731c1391b1dcf83'
QUANT_TRANSPOSE_REVIEW = (
    'bench/review-checkpoints/2026-10-16/quant-transpose-finite-source-acceptance/'
    'evidence/root-validation/review.json'
)
QUANT_TRANSPOSE_REVIEW_SHA256 = '9e0c4b360363f94670bc2477b8783477d4a93aa89ac93a2e48b050b066e87f77'


"""
s=s.replace('def extend_reviewed_authority(root, previous):',constants+'def extend_reviewed_authority(root, previous):');f.write_text(s)
f=R/'bench/tests/test_exact_numerical_bindings.py';s=f.read_text();s=s.replace('QUANT_GLOBAL_PACKAGE, QUANT_GLOBAL_REVIEW)', 'QUANT_GLOBAL_PACKAGE, QUANT_GLOBAL_REVIEW,\n    QUANT_TRANSPOSE_PACKAGE, QUANT_TRANSPOSE_REVIEW)')
s=s.replace('self.assertEqual(100,len(queue[\'rules\']))','self.assertEqual(116,len(queue[\'rules\']))').replace('[29,20,9,340,2601,2400]','[29,20,9,340,2601,2400,863]')
delta=load(P/'evidence/numerical-authority-delta.json');n=len(delta['requirements'][0]['callback_union']);s=s.replace('[15,13,3,9,26,14]',str([15,13,3,9,26,14,n]).replace(' ',''))
s=s.replace('self.assertEqual(1253,','self.assertEqual(1393,').replace('self.assertEqual(1206,','self.assertEqual(1346,').replace('self.assertEqual(5399,','self.assertEqual(6262,')
# Existing predicates/earlier tests stay unchanged; new tests exercise actual trust boundaries.
extra='''
 def test_transpose_observations_require_no_numerical_ports(self):
  table=next(t for t in authority['requirements'] if t['task']=='quant_transpose_kernel')
  empty=[c for c in table['cases'] if not c['required_binding_refs']]
  self.assertEqual(847,len(empty))
  row,p,c=usage('quant_transpose_kernel',lambda t,c:not c['required_binding_refs'])
  self.assertEqual([],row['deferred_numerical_assumptions'])
  gate.validate_used_numerics('quant_transpose_kernel',row,p,c,queue,root=REPO)
 def test_transpose_every_required_port_is_enforced(self):
  table=next(t for t in authority['requirements'] if t['task']=='quant_transpose_kernel')
  case=next(c for c in table['cases'] if c['required_binding_refs'])
  row,p,c=usage('quant_transpose_kernel',lambda t,x:t['declaration']==table['declaration'] and x==case)
  gate.validate_used_numerics('quant_transpose_kernel',row,p,c,queue,root=REPO)
  for omitted in range(len(row['deferred_numerical_assumptions'])):
   changed=deepcopy(row);changed['deferred_numerical_assumptions'].pop(omitted)
   with self.assertRaises(ValueError):gate.validate_used_numerics('quant_transpose_kernel',changed,p,c,queue,root=REPO)
 def test_transpose_archived_source_and_independent_review_are_pinned(self):
  targets=[QUANT_TRANSPOSE_REVIEW,
   QUANT_TRANSPOSE_PACKAGE+'/inherited-v1/bench/tritonbench_g/quant_transpose_kernel/FiniteAcceptance.lean.txt',
   QUANT_TRANSPOSE_PACKAGE+'/peer-review/native-review-receipt.json',
   'bench/tritonbench_g/quant_transpose_kernel/Proofs/FiniteNativeArithmetic.lean']
  for target in targets:
   with self.subTest(target=target),tempfile.TemporaryDirectory() as temp:
    root=clone_inputs(temp);path=root/target
    path.write_bytes(path.read_bytes()+b'\\nchanged by adversarial test\\n')
    with self.assertRaises(ValueError):derive_authority(root)
'''
assert 'class ExactBindings' in s;s=s.replace("if __name__=='__main__':",extra+"\nif __name__=='__main__':")
# Added methods must remain in the test class before its __main__ footer.
parsed=ast.parse(s);cls=next(x for x in parsed.body if isinstance(x,ast.ClassDef) and x.name=='ExactBindings');assert all(any(isinstance(x,ast.FunctionDef) and x.name==n for x in cls.body) for n in ['test_transpose_observations_require_no_numerical_ports','test_transpose_every_required_port_is_enforced','test_transpose_archived_source_and_independent_review_are_pinned'])
f.write_text(s)
print(json.dumps({'adapter':'bench/quant_transpose_numerical_extension.py','families':16,'ports':140,'cases':863,'new_tests':3,'callback_union':n}))
