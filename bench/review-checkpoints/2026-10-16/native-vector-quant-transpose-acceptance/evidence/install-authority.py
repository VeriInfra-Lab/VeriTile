from pathlib import Path
import ast,json,sys
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
B=R/'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance'
sys.path.insert(0,str(R/'bench'));from adapter_common import hash_file,load,write_json
state=load(B/'evidence/preparation-state.json')
assert len(state['new_source_sha256'])==25
for p,h in state['new_source_sha256'].items():assert hash_file(R/p)==h,p
Q=T/'quant-transpose-finite-acceptance-revision2-wip';V=T/'vector-finite-acceptance-revision2-wip'
q=(Q/'authority_adapter.py').read_text()
old="    for relative, expected in receipt['artifacts'].items():\n        bind(package_relative / relative, expected)"
new="    for relative, expected in receipt['artifacts'].items():\n        # Archive exact historical source/object bytes outside active Lean discovery.\n        archived = relative + '.txt' if relative.endswith('.lean') else (\n            relative + '.bin' if relative.endswith(('.olean', '.olean.private', '.olean.server', '.ilean', '.ir')) else relative)\n        bind(package_relative / archived, expected)"
assert q.count(old)==1;q=q.replace(old,new);(R/'bench/quant_transpose_numerical_extension.py').write_text(q)
v=(V/'authority_adapter.py').read_text();old=' for rel,h in receipt[\'artifacts\'].items():bind(package/rel,h)'
new=" for rel,h in receipt['artifacts'].items():\n  # Preserve exact evidence bytes without registering historical Lean modules.\n  archived=rel+'.txt' if rel.endswith('.lean') else (rel+'.bin' if rel.endswith(('.olean','.olean.private','.olean.server','.ilean','.ir')) else rel)\n  bind(package/archived,h)"
assert v.count(old)==1;v=v.replace(old,new);assert not (R/'bench/vector_native_numerical_extension.py').exists();(R/'bench/vector_native_numerical_extension.py').write_text(v)
f=R/'bench/reviewed_numerical_extensions.py';s=f.read_text()
replacements={
 'bench/review-checkpoints/2026-10-16/quant-transpose-finite-source-acceptance/':'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance/',
 "QUANT_TRANSPOSE_RECEIPT = '63392bac5342812d467c2bc499e7870e2609d49fa74faaf01731c1391b1dcf83'":"QUANT_TRANSPOSE_RECEIPT = '883f7e8e086acb7b20ac7a3f20ba16dfdf697b425ef123d0af7509c877bf188a'",
 "QUANT_TRANSPOSE_REVIEW_SHA256 = '9e0c4b360363f94670bc2477b8783477d4a93aa89ac93a2e48b050b066e87f77'":"QUANT_TRANSPOSE_REVIEW_SHA256 = '54c7e23fb5975ec8fddd7f7bb19304616f9895607fc9f4af313a60e29aacf918'"}
for old,new in replacements.items():assert old in s,old;s=s.replace(old,new)
start=s.index('QUANT_TRANSPOSE_PACKAGE');s=s[:start]+s[start:].replace("'evidence/root-validation/review.json'","'evidence/root-validation/quant-transpose/review.json'")
constants="""VECTOR_NATIVE_PACKAGE = (
    'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance/'
    'evidence/producer/vector'
)
VECTOR_NATIVE_RECEIPT = '8679242cdbdbe9d8972fe59c011fc89f51ba723de0b233bbe8af07ec32312cc4'
VECTOR_NATIVE_REVIEW = (
    'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance/'
    'evidence/root-validation/vector/review.json'
)
VECTOR_NATIVE_REVIEW_SHA256 = '2d5529a7a7114a7e846026d44f3d1b8a7b26455294580b7e9a48326074c3f362'


"""
s=s.replace('def extend_reviewed_authority(root, previous):',constants+'def extend_reviewed_authority(root, previous):')
old='''    return extend_authority(
        root, result, QUANT_TRANSPOSE_PACKAGE, QUANT_TRANSPOSE_RECEIPT,
        QUANT_TRANSPOSE_REVIEW, QUANT_TRANSPOSE_REVIEW_SHA256,
    )'''
new='''    result = extend_authority(
        root, result, QUANT_TRANSPOSE_PACKAGE, QUANT_TRANSPOSE_RECEIPT,
        QUANT_TRANSPOSE_REVIEW, QUANT_TRANSPOSE_REVIEW_SHA256,
    )
    from vector_native_numerical_extension import extend_vector_native_authority
    return extend_vector_native_authority(
        root, result, VECTOR_NATIVE_PACKAGE, VECTOR_NATIVE_RECEIPT,
        VECTOR_NATIVE_REVIEW, VECTOR_NATIVE_REVIEW_SHA256,
    )'''
assert s.count(old)==1;s=s.replace(old,new);f.write_text(s)
f=R/'bench/tests/test_exact_numerical_bindings.py';s=f.read_text()
replacements={
 'QUANT_TRANSPOSE_PACKAGE, QUANT_TRANSPOSE_REVIEW)':'QUANT_TRANSPOSE_PACKAGE, QUANT_TRANSPOSE_REVIEW,\n    VECTOR_NATIVE_PACKAGE, VECTOR_NATIVE_REVIEW)',
 "self.assertEqual(116,len(queue['rules']))":"self.assertEqual(153,len(queue['rules']))",
 '[29,20,9,340,2601,2400,863]':'[29,20,9,340,2601,2400,1144,4498]',
 '[15,13,3,9,26,14,14]':'[15,13,3,9,26,14,14,22]',
 'self.assertEqual(1393,':'self.assertEqual(1689,',
 'self.assertEqual(1346,':'self.assertEqual(1642,',
 'self.assertEqual(6262,':'self.assertEqual(11041,',
 'self.assertEqual(847,len(empty))':'self.assertEqual(1128,len(empty))'}
for old,new in replacements.items():assert old in s,old;s=s.replace(old,new)
extra='''
 def test_native_vector_no_access_return_needs_no_numerical_ports(self):
  row,p,c=usage('vector_addition',lambda t,c:t['declaration']['file'].endswith('/NativeCorrect.lean') and c['configuration']['outcome']=='two31_no_access_return')
  self.assertTrue(c)
  self.assertEqual([],row['deferred_numerical_assumptions'])
  gate.validate_used_numerics('vector_addition',row,p,c,queue,root=REPO)
 def test_native_vector_required_ports_cannot_use_old_bounded_bindings(self):
  table=next(t for t in authority['requirements'] if t['declaration']['file'].endswith('/NativeCorrect.lean'))
  case=next(c for c in table['cases'] if c['required_binding_refs'])
  row,p,c=usage('vector_addition',lambda t,x:t['declaration']==table['declaration'] and x==case)
  gate.validate_used_numerics('vector_addition',row,p,c,queue,root=REPO)
  for omitted in range(len(row['deferred_numerical_assumptions'])):
   changed=deepcopy(row);changed['deferred_numerical_assumptions'].pop(omitted)
   with self.assertRaises(ValueError):gate.validate_used_numerics('vector_addition',changed,p,c,queue,root=REPO)
  previous,_,_=usage('vector_addition',lambda t,x:t['declaration']['file'].endswith('/FiniteCorrect.lean'))
  with self.assertRaises(ValueError):gate.validate_used_numerics('vector_addition',previous,p,c,queue,root=REPO)
 def test_native_vector_archived_objects_and_review_are_pinned(self):
  targets=[VECTOR_NATIVE_REVIEW,
   VECTOR_NATIVE_PACKAGE+'/evidence/final-objects/bench/tritonbench_g/vector_addition/NativeCorrect.olean.bin',
   VECTOR_NATIVE_PACKAGE+'/evidence/inherited/annotated-i32-scope-review/Narrowing.lean.txt',
   'bench/tritonbench_g/vector_addition/Proofs/NativeMemory.lean']
  for target in targets:
   with self.subTest(target=target),tempfile.TemporaryDirectory() as temp:
    root=clone_inputs(temp);path=root/target
    path.write_bytes(path.read_bytes()+b'\\nchanged by adversarial test\\n')
    with self.assertRaises(ValueError):derive_authority(root)
'''
s=s.replace("if __name__=='__main__':",extra+"\nif __name__=='__main__':");ast.parse(s);f.write_text(s)
import shutil
shutil.copyfile(Path(__file__),B/'evidence/install-authority.py')
write_json(B/'evidence/archive-adapter-review.json',{'reviewer':'root','status':'reviewed','change':'Only deterministic .lean to .lean.txt and generated Lean object to .bin archival mapping is added to each frozen adapter; all source/review/receipt/collision/whole-queue validation remains unchanged.','quant_producer_adapter_sha256':hash_file(Q/'authority_adapter.py'),'quant_current_adapter_sha256':hash_file(R/'bench/quant_transpose_numerical_extension.py'),'vector_producer_adapter_sha256':hash_file(V/'authority_adapter.py'),'vector_current_adapter_sha256':hash_file(R/'bench/vector_native_numerical_extension.py'),'adversarial_object_and_source_tests_added':True})
print(json.dumps({'new_families':53,'new_ports':436,'new_cases':5642,'total_rules':153,'exact_tests':43},indent=2))
