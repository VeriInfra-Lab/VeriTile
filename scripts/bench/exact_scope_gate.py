"""Validated case-specific callback lookup for the unchanged scope gate."""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scripts.bench.exact_numerical_bindings import derive_authority,validate_exact_queue,canonical,fail

FIELDS={'exact_binding_provenance','exact_callback_requirements','exact_numerical_inputs','exact_numerical_summary'}

def validated_exact_requirements(root,task,public,proof_cases,queue):
 exact_present=bool(FIELDS&set(queue))
 has_new_rules=any(r.get('status')=='pending_numeric_validation' for r in queue['rules'])
 if not exact_present and not has_new_rules:return {}
 fail(FIELDS<=set(queue),'incomplete exact numeric side tables')
 fail(root is not None,'exact numeric validation requires an explicit repository root')
 from scripts.bench.deferred_numerical_experiments import collect_legacy
 authority=derive_authority(root);legacy=collect_legacy(root)
 validate_exact_queue(queue,authority,legacy)
 tables={(r['declaration']['file'],r['declaration']['declaration']):r for r in queue['exact_callback_requirements']}
 result={}
 for key,cases in proof_cases.items():
  if key not in tables:continue  # old declarations retain the original strict logic
  table=tables[key];declaration=public[key]
  fail(table['task']==task,'numeric declaration belongs to another task')
  fail({k:declaration[k] for k in ('file','declaration','statement_sha256')}==table['declaration'],'wrong exact numeric statement')
  fail(sorted(declaration.get('unclosed_numeric_callbacks',[]))==table['callback_union'],'declaration callback inventory differs from source-derived authority')
  expected={canonical({k:row[k] for k in ('dtype_profile','configuration')}):row for row in table['cases']}
  fail(set(cases)<=set(expected),'unsupported dtype/configuration numeric case')
  result[key]=expected
 return result


def validate_exact_case_links(exact,proof,case,applicable,callback_links):
 """Each scalar port must be explicitly deferred, including repeated callbacks."""
 if proof not in exact:return
 for ref in exact[proof][case]['required_binding_refs']:
  key=(ref['assumption_id'],ref['scope_binding_id'])
  fail(key in applicable and key in callback_links.get(ref['callback'],set()),
       'exact scalar port omitted from callback assumptions')
