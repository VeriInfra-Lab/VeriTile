"""Fail-closed ReLU extension. Integration requires two independently pinned receipts."""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from pathlib import Path
from copy import deepcopy
import hashlib,json,sys
sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def safe(root,rel):
 rel=Path(rel);p=(root/rel).resolve()
 if rel.is_absolute() or '..' in rel.parts or not p.is_relative_to(root):raise ValueError('escaping ReLU path')
 return p

def extend_authority(root,previous,package_path,receipt_sha256,review_path,review_sha256):
 from scripts.bench.adapter_common import declarations
 root=Path(root).resolve();out=deepcopy(previous)
 def bind(rel,h):
  rel = _archive_evidence_input(rel, package_path)
  p=safe(root,rel)
  if not p.is_file() or sha(p)!=h:raise ValueError('changed reviewed ReLU input: '+str(rel))
  key='repo:'+Path(rel).as_posix()
  if out['input_sha256'].get(key,h)!=h:raise ValueError('conflicting reviewed ReLU input')
  out['input_sha256'][key]=h;return p
 package=safe(root,package_path);base=Path(package_path)
 receipt=load(bind(base/'evidence/receipt.json',receipt_sha256));review=load(bind(review_path,review_sha256))
 if receipt.get('frozen') is not True or review.get('reviewer')!='root' or review.get('status')!='substantively_reviewed_for_finite_scope_integration' or review.get('producer_receipt_sha256')!=receipt_sha256:raise ValueError('unreviewed ReLU authority')
 if receipt['checkpoint_module_source_paths']!={p:p+'.txt' for p in receipt['modules']}:raise ValueError('invalid source mapping')
 for p,h in receipt['modules'].items():bind(base/(p+'.txt'),h);bind(p,h)
 for p,h in receipt['artifacts'].items():bind(base/p,h)
 delta=load(package/'evidence/numerical-authority-delta.json')
 if delta['task']!='relu_triton_kernel':raise ValueError('wrong ReLU authority task')
 src=delta['original_source'];bind(src['path'],src['sha256'])
 expected={'scalar_families':len(delta['rules']),'scope_bindings':len(delta['provenance']),'declaration_cases':sum(len(t['cases']) for t in delta['requirements']),'registered_declarations':len(delta['requirements'])}
 if any(delta['summary'].get(k)!=v for k,v in expected.items()):raise ValueError('ReLU authority count mismatch')
 captures=delta['dependency_snapshots']
 expected_captures={p for p in receipt['artifacts'] if p=='evidence/dependencies-before.json' or (Path(p).parent==Path('evidence') and Path(p).name.startswith('dependencies-added-before-'))}
 if {r['path'] for r in captures}!=expected_captures or len(captures)!=len(expected_captures):raise ValueError('incomplete dependency captures')
 for ref in captures:
  if receipt['artifacts'].get(ref['path'])!=ref['sha256']:raise ValueError('unbound dependency capture')
  captured=load(package/ref['path'])
  for module,row in captured['project'].items():bind(module.replace('.','/')+'.lean',row['sha256'])
 after=load(package/'evidence/dependencies-after.json')
 if after.get('all_unchanged') is not True or after.get('captures')!={r['path']:{'sha256':r['sha256'],'verified_unchanged':True} for r in captures}:raise ValueError('incomplete dependency verification')
 evidence=delta['native_evidence']
 if len(evidence)!=1 or evidence[0]['role']!='full22_and_constexpr_boundaries':raise ValueError('incomplete native provenance')
 native_refs=[]
 for entry in evidence:
  ref=entry['receipt'];native_refs.append(ref)
  if receipt['artifacts'].get(ref['path'])!=ref['sha256']:raise ValueError('unbound native receipt')
  native=load(package/ref['path'])
  if native['source_sha256']!=src['sha256'] or native['source_ast_identity'] is not True or native['gpu_executed'] is not False or len(native['names'])!=22:raise ValueError('native source/scope mismatch')
  for name,h in native['artifacts'].items():
   p=(Path(ref['path']).parent/name).as_posix()
   if receipt['artifacts'].get(p)!=h:raise ValueError('unbound native artifact')
   bind(base/p,h)
 for port in delta['provenance'].values():
  if port['original_source']!=src or port['lean_source_sha256']!=receipt['modules'] or port['native_receipts']!=native_refs:raise ValueError('ReLU port provenance mismatch')
  raw=port['raw_ttir']
  if receipt['artifacts'].get(raw['path'])!=raw['sha256']:raise ValueError('unbound raw TTIR port')
 for table in delta['requirements']:
  ref=table['declaration'];decl=declarations(root/ref['file']).get(ref['declaration'])
  if decl is None or decl['kind']!='specification' or decl['statement_sha256']!=ref['statement_sha256']:raise ValueError('ReLU public statement changed')
 ids=[r['id'] for r in delta['rules']];oldids={r['id'] for r in out['rules']}
 refs=[(t['declaration']['file'],t['declaration']['declaration']) for t in delta['requirements']];oldrefs={(t['declaration']['file'],t['declaration']['declaration']) for t in out['requirements']}
 if len(ids)!=len(set(ids)) or oldids.intersection(ids) or set(out['provenance']).intersection(delta['provenance']) or len(refs)!=len(set(refs)) or oldrefs.intersection(refs):raise ValueError('ReLU authority collision')
 out['rules']+=deepcopy(delta['rules']);out['provenance'].update(deepcopy(delta['provenance']));out['requirements']+=deepcopy(delta['requirements'])
 out.setdefault('independent_source_reviews',[]).append({'package':'repo','path':review_path,'sha256':review_sha256})
 out['summary'].update(new_scalar_families=len(out['rules']),scope_bindings=len(out['provenance']),declaration_cases=sum(len(t['cases']) for t in out['requirements']),registered_declarations=len(out['requirements']),unvalidated_numeric_instances=len(out['provenance']),relu_actual_delta=delta['summary'])
 return out

def _archive_evidence_input(relative, package_path):
    """Relocate only frozen evidence; the caller still checks safety and hashes."""
    rel = Path(relative)
    package = Path(package_path)
    if rel.is_relative_to(package):
        text = rel.as_posix()
        if text.endswith('.lean'):
            return Path(text + '.txt')
        if text.endswith(('.olean', '.olean.private', '.olean.server', '.ilean', '.ir', '.pyc')):
            return Path(text + '.bin')
    return relative
