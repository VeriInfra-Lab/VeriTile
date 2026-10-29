"""Proposed exact reviewed cosine extension. Root supplies both literal pins.

This producer module cannot authorize itself. Integration must call it only
with independently reviewed receipt/review hashes; whole-queue equality stays
unchanged. Every input is registered under a repo: key and bound fail-closed.
"""
from copy import deepcopy
from pathlib import Path
from adapter_common import declarations,hash_file,load

def extend_cosine_compute_native_authority(root,previous,package_path,receipt_sha256,review_path,review_sha256):
 root=Path(root).resolve();result=deepcopy(previous)
 def bind(relative,expected):
  rel=Path(relative);p=(root/rel).resolve()
  if rel.is_absolute() or '..' in rel.parts or not p.is_relative_to(root) or not p.is_file() or hash_file(p)!=expected:
   raise ValueError('changed reviewed cosine input: '+str(rel))
  key='repo:'+rel.as_posix()
  if result['input_sha256'].get(key,expected)!=expected:raise ValueError('conflicting cosine input')
  result['input_sha256'][key]=expected;return p
 package=Path(package_path)
 receipt=load(bind(package/'evidence/receipt.json',receipt_sha256))
 review=load(bind(review_path,review_sha256))
 if not receipt['frozen'] or review['reviewer']!='root' or review['status']!='substantively_reviewed_for_finite_scope_integration' or review['producer_receipt_sha256']!=receipt_sha256:
  raise ValueError('unreviewed native cosine revision1 authority')
 mapping=receipt['checkpoint_module_source_paths']
 if mapping!={p:p+'.txt' for p in receipt['modules']}:raise ValueError('invalid frozen cosine source mapping')
 for rel,h in receipt['modules'].items():bind(package/mapping[rel],h);bind(rel,h)
 for rel,h in receipt['artifacts'].items():bind(package/rel,h)
 deps=load(root/package/'evidence/dependencies-before.json')
 for rel,entry in deps['project'].items():bind(rel,entry['sha256'])
 delta=load(root/package/'evidence/numerical-authority-delta.json')
 expected={'scalar_families':len(delta['rules']),'scope_bindings':len(delta['provenance']),'declaration_cases':sum(len(t['cases']) for t in delta['requirements']),'registered_declarations':len(delta['requirements'])}
 if any(delta['summary'].get(k)!=v for k,v in expected.items()):raise ValueError('cosine authority count mismatch')
 if delta['task']!='cosine_compute':raise ValueError('wrong task in cosine delta')
 for port in delta['provenance'].values():
  for field in ['original_source','scalar_source']:
   ref=port[field];bind(ref['path'],ref['sha256'])
  for field in ['execution_source','host_source']:
   ref=port[field];bind(ref['module'],ref['sha256'])
  for rel,h in port['lean_source_sha256'].items():bind(rel,h)
  for field in ['raw_ttir','host_math_probe','producer_scalar_inventory']:
   ref=port[field]
   if ref is not None:bind(package/ref['path'],ref['sha256'])
  for ref in port['native_receipts']:bind(package/ref['path'],ref['sha256'])
 for table in delta['requirements']:
  ref=table['declaration'];decl=declarations(root/ref['file']).get(ref['declaration'])
  if decl is None or decl['kind']!='specification' or decl['statement_sha256']!=ref['statement_sha256']:
   raise ValueError('changed cosine public statement')
 oldids={r['id'] for r in result['rules']};newids=[r['id'] for r in delta['rules']]
 oldrefs={(r['declaration']['file'],r['declaration']['declaration']) for r in result['requirements']}
 newrefs=[(r['declaration']['file'],r['declaration']['declaration']) for r in delta['requirements']]
 if oldids.intersection(newids) or len(newids)!=len(set(newids)) or set(result['provenance']).intersection(delta['provenance']) or oldrefs.intersection(newrefs) or len(newrefs)!=len(set(newrefs)):
  raise ValueError('reviewed cosine authority collision')
 result['rules']+=deepcopy(delta['rules']);result['provenance'].update(deepcopy(delta['provenance']));result['requirements']+=deepcopy(delta['requirements'])
 result['independent_source_reviews'].append({'package':'repo','path':str(review_path),'sha256':review_sha256})
 result['summary'].update(new_scalar_families=len(result['rules']),scope_bindings=len(result['provenance']),
  declaration_cases=sum(len(r['cases']) for r in result['requirements']),registered_declarations=len(result['requirements']),
  unvalidated_numeric_instances=len(result['provenance']),cosine_compute_native_delta=deepcopy(delta['summary']))
 return result
