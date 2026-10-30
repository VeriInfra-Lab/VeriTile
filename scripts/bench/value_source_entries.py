"""Explicit typed-source admissions, independent of legacy root-file discovery.

FPCorrect existing is not an admission. The original Python, actual typed source,
source-shape bridge and independent receipt/review must all be hash bound. A
compile-repaired candidate stays a candidate; this module never substitutes it
silently in the old original-Python source audit.
"""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from pathlib import Path
import hashlib,json,re,sys

def validate_entries(root: Path, registry: dict) -> dict:
    if registry.get('schema') != 1:
        raise ValueError('Unknown explicit source-entry schema')
    sys.path.insert(0,str(root/'scripts'))
    from spec_sheet import split_decls
    result={}
    for row in registry['entries']:
        task=row['task'];prefix='bench/tritonbench_g/'+task+'/'
        if task in result: raise ValueError('Duplicate source entry: '+task)
        if row['source_variant'] not in {'original','compile_repaired_candidate','model'}:
            raise ValueError('Source variant must be explicit: '+task)
        if row.get('native_execution') is not False:
            raise ValueError('New native evidence review required: '+task)
        for key in ['entry','typed_source']:
            if not row[key].startswith(prefix):raise ValueError('Source path outside task: '+task)
        required={row[k] for k in ['entry','typed_source','original_python','review_receipt','root_review']} | {row['bridge']['file']}
        if row['source_variant']=='compile_repaired_candidate':
            if not row.get('candidate_python'):raise ValueError('Candidate source omitted: '+task)
            required.add(row['candidate_python'])
        elif row.get('candidate_python'):
            raise ValueError('Candidate incorrectly classified as original: '+task)
        if not required <= set(row['input_sha256']):raise ValueError('Incomplete source-entry evidence: '+task)
        for name,digest in row['input_sha256'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Invalid source-entry input: '+name)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Stale source-entry input: '+name)
        decls={d['name'] for d in split_decls((root/row['bridge']['file']).read_text())}
        if not row['bridge']['declarations'] or not set(row['bridge']['declarations'])<=decls:
            raise ValueError('Missing source bridge: '+task)
        source=(root/row['typed_source']).read_text()
        definitions={d['name'] for d in split_decls(source)}
        frontend=row.get('frontend_kind','shared_triton_dsl')
        macro_defined=re.search(r'(?m)^(?:macro|syntax)\s+"'+re.escape(row['typed_definition'])+r'"',source) is not None
        if row['typed_definition'] not in definitions and not (frontend=='shared_triton_dsl' and macro_defined):
            raise ValueError('No registered typed source body: '+task)
        if frontend=='shared_triton_dsl':
            if 'triton {' not in source:raise ValueError('No shared DSL source body: '+task)
        elif frontend=='task_local_exact_ast':
            if row.get('ordinary_shared_dsl_admission') is not False:
                raise ValueError('Task-local frontend promoted to shared DSL: '+task)
            if row.get('memory_semantics')!='sequentially_consistent_schedule':
                raise ValueError('Task-local memory profile must be explicit: '+task)
            if not row.get('translation_trust_boundary'):
                raise ValueError('Task-local translation boundary omitted: '+task)
        else:raise ValueError('Unknown source frontend kind: '+task)
        receipt=json.loads((root/row['review_receipt']).read_text())
        review=json.loads((root/row['root_review']).read_text())
        if review.get('receipt_sha256') != row['input_sha256'][row['review_receipt']]:
            raise ValueError('Source review is not bound to its receipt: '+task)
        if row['source_variant']=='compile_repaired_candidate' and row['original_compilation_status']!='failed_original_candidate_only':
            raise ValueError('Candidate erases original compile failure: '+task)
        result[task]=row
    return result

def registered_value_sources(root: Path) -> dict:
    path=root/'bench/tritonbench_g/fp_value_source_entries.json'
    return validate_entries(root,json.loads(path.read_text()))
