#!/usr/bin/env python3
"""Explicit source/public/helper inventory; never a replacement proof ledger.

The frozen value registry remains the authority for model, domain and stage
claims. This adapter adds exact file membership and source-entry roles without
changing its 184-task count or rewriting historical declarations.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import zipfile
from adapter_common import *

VARIANTS = {'original_source_translation', 'compile_repaired_candidate',
            'observed_compiled_graph', 'compile_repaired_candidate_graph',
            'standalone_sc_model', 'source_and_compiled_graph_specialization'}
FRONTENDS = {'shared_triton_dsl', 'task_local_exact_ast'}
NONVALUES = {'recurrence_identity', 'memory_frame_contract', 'protocol_contract'}

def validate_metadata_corrections(task, frozen_row, corrections, review):
    """A reviewed annotation correction never rewrites the frozen value row."""
    if not isinstance(corrections, list) or not corrections:
        raise ValueError('empty/invalid metadata correction set')
    if (review.get('reviewer') != 'root' or not review.get('status') or
            re.search(r'(?:^|_)(?:failed|rejected|not_run)(?:_|$)', review['status']) or
            review.get('reviewed_metadata_corrections_sha256', {}).get(task) != sha(canonical(corrections))):
        raise ValueError('metadata correction lacks exact root review')
    seen = set()
    for correction in corrections:
        path = correction.get('field_path')
        # Scope/model/status changes are not text corrections. Future field
        # types need their own reviewed migration, not an arbitrary JSON patch.
        if path != ['value_extension', 'dtype_profile'] or tuple(path) in seen:
            raise ValueError('unsupported/duplicate metadata correction field')
        seen.add(tuple(path))
        before = frozen_row
        for key in path:
            if not isinstance(before, dict) or key not in before:
                raise ValueError('metadata correction field missing in frozen row')
            before = before[key]
        if correction.get('before') != before:
            raise ValueError('metadata correction before value differs from frozen row')
        if (not isinstance(correction.get('after'), str) or not correction['after'].strip()
                or correction['after'] == before or not isinstance(correction.get('reason'), str)
                or not correction['reason'].strip()):
            raise ValueError('metadata correction lacks changed value/review reason')
    return corrections

def public_files(catalog, task=None):
    """Replacement for convention-based specification_files; includes protocols."""
    return sorted({d['file'] for row in catalog['tasks'] if task is None or row['task']==task
                   for d in row['declarations']})

def proof_files(catalog, task=None):
    return sorted(f['path'] for row in catalog['tasks'] if task is None or row['task']==task
                  for f in row['files'])

def trust_targets(root, catalog, tasks=()):
    """Select every verified catalog file, never infer sources from filenames.

    This is a target inventory, not an acceptance result. Every selected file
    must still pass the existing axiom/specification and comparator worker.
    Validate the entire catalog even when the caller requests one task.
    """
    validate_catalog(root, catalog)
    if len(tasks) != len(set(tasks)):
        raise ValueError('duplicate trust audit task')
    unknown = set(tasks) - set(catalog['expected_task_names'])
    if unknown:
        raise ValueError('unknown trust audit tasks: ' + repr(sorted(unknown)))
    selected = set(tasks)
    return sorted(f['path'] for row in catalog['tasks'] if not selected or row['task'] in selected
                  for f in row['files'])

def source_entries(catalog, *, task=None, variant=None, frontend=None):
    """Never force several source graphs/concurrency entries into one filename."""
    return [{'task':row['task'],**view} for row in catalog['tasks']
            if task is None or row['task']==task for view in row['source_views']
            if (variant is None or view['variant']==variant) and
               (frontend is None or view['frontend']==frontend)]

def graph_stage_coverage(task, variant):
    """Optional supplement helper: partial software stages cannot cover the whole."""
    required=set(task['required_value_stages']);values=set();bits=set()
    for d in task['declarations']:
        if (d['source_variant']!=variant or d['role']!='parametric_value_contract'
                or d.get('scope_relation')!='full_declared_stage'):
            continue
        if d['model_kind'] not in {'backend_value_contract','bit_interpreter_contract'}:
            continue
        values.update(d['covers_stages'])
        if d['model_kind']=='bit_interpreter_contract':bits.update(d['covers_stages'])
    return {'required_stages':sorted(required),'value_stages':sorted(values),'software_stages':sorted(bits),
        'all_required_value_stages':bool(required) and required<=values,
        'all_required_software_stages':bool(required) and required<=bits}

def prepare_baseline(package, output):
    """Materialize only receipt-pinned archive members into a fresh directory."""
    import shutil
    if output.exists():raise ValueError('baseline output must be a fresh directory')
    receipt=load(package/'receipt.json')
    if receipt.get('schema') not in {2,3} or receipt.get('summary',{}).get('tasks')!=184:
        raise ValueError('not a frozen 184-task value metadata package')
    manifest=load(checked(package,'input-sha256.json',receipt['artifact_sha256']['input-sha256.json']))
    archive_path=checked(package,'inputs.zip',receipt['inputs_zip_sha256'])
    with zipfile.ZipFile(archive_path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(manifest):
            raise ValueError('metadata archive membership differs')
        # Validate all names and content before creating the output tree.
        for name,h in manifest.items():
            relative(name)
            if sha(archive.read(name))!=h:raise ValueError('archived input changed: '+name)
        output.mkdir(parents=True)
        for name in ('receipt.json','input-sha256.json','inputs.zip'):shutil.copyfile(package/name,output/name)
        for name in names:
            dest=output/'tree'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(archive.read(name))
    return {'prepared_inputs':len(manifest),'metadata_receipt_sha256':hash_file(output/'receipt.json')}

def baseline(package):
    """Validate every file of a frozen V2/V3 input archive and preserve its claims."""
    receipt = load(package / 'receipt.json')
    if receipt.get('schema') not in {2, 3} or receipt.get('summary', {}).get('tasks') != 184:
        raise ValueError('not a frozen 184-task value metadata package')
    manifest = load(checked(package, 'input-sha256.json', receipt['artifact_sha256']['input-sha256.json']))
    checked(package, 'inputs.zip', receipt['inputs_zip_sha256'])
    with zipfile.ZipFile(package / 'inputs.zip') as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest):
            raise ValueError('V2 archive membership differs')
        for name, h in manifest.items():
            relative(name)
            if sha(archive.read(name)) != h:
                raise ValueError('V2 archived input changed: ' + name)
            checked(package / 'tree', name, h)
    root = package / 'tree'
    registry = load(root / BASE / 'fp_value_extensions.json')
    current = load(root / BASE / 'fp_value_current_status.json')
    old = load(root / registry['baseline']['numerical_obligations.json']['path'])
    old41 = load(root / registry['preserved_original_registry']['path'])
    if len(old['declarations']) != 747 or len(old41['tasks']) != 41:
        raise ValueError('historical 747/41 baseline changed')
    for name, binding in registry['baseline'].items():
        checked(root, binding['path'], binding['sha256'])
    checked(root, registry['preserved_original_registry']['path'], registry['preserved_original_registry']['sha256'])
    return {'receipt_sha256': hash_file(package / 'receipt.json'), 'root': root,
            'manifest': manifest, 'registry': registry, 'current': current,
            'old': old, 'old41': old41,
            'entries': load(root / BASE / 'fp_value_source_entries.json')['entries']}

def verify_source_view(root, view, bindings):
    """Verify explicit selection only. Hash binding is not Python refinement."""
    if view['variant'] not in VARIANTS or view['frontend'] not in FRONTENDS:
        raise ValueError('unknown source variant/frontend')
    if view['frontend'] == 'task_local_exact_ast':
        if (view.get('ordinary_shared_dsl_admission') is not False or
                view.get('memory_semantics') != 'sequentially_consistent_schedule' or
                not view.get('translation_trust_boundary')):
            raise ValueError('task-local SC presented as shared DSL or missing boundary')
    elif view.get('memory_semantics') != 'single_program_total_typed_memory':
        raise ValueError('shared DSL memory profile changed')
    selected = view['selected_definitions']
    if not selected:
        raise ValueError('no explicit source selection')
    for file, names in selected.items():
        if not names or len(names) != len(set(names)) or file not in bindings:
            raise ValueError('source file/name not explicitly bound: ' + file)
        p = checked(root, file, bindings[file])
        text = p.read_text()
        decls = declarations(p)
        macros = set(re.findall(r'(?m)^\s*(?:macro|syntax)\s+"([^"\n]+)"', lean_code_mask(text, strings=False)))
        if not set(names) <= (set(decls) | macros):
            raise ValueError('missing selected source definition: ' + file)
        # Some declared entry aliases refer to a separately declared typed body;
        # full graph equality is supplied by the bound review/Lean proof.
        if view['frontend'] == 'shared_triton_dsl' and not re.search(r'\btriton\s*\{', lean_code_mask(text)):
            raise ValueError('selected shared DSL file has no actual DSL syntax: ' + file)
    if view['variant'].startswith('compile_repaired_candidate'):
        if not view.get('candidate_python') or not view.get('original_python'):
            raise ValueError('candidate must retain both Python variants')
        if view['candidate_python'] == view['original_python']:
            raise ValueError('candidate cannot replace original')

def source_views(task, ext, entry, source_map):
    """Selection is explicit checked-in data, never inferred from helper names."""
    result = []
    if ext:
        selected = source_map.get(task)
        if selected is None and entry:
            selected = [{'id': 'explicit_metadata_source_admission',
                'variant': 'compile_repaired_candidate' if entry['source_variant'] == 'compile_repaired_candidate' else 'original_source_translation',
                'selected_definitions': {entry['typed_source']: [entry['typed_definition']]}}]
        for spec in selected or []:
            result.append({**spec, 'authority': 'frozen_value_registry',
                'frontend': entry.get('frontend_kind', 'shared_triton_dsl') if entry else 'shared_triton_dsl',
                'ordinary_shared_dsl_admission': not entry or entry.get('frontend_kind', 'shared_triton_dsl') == 'shared_triton_dsl',
                'memory_semantics': entry.get('memory_semantics', 'single_program_total_typed_memory') if entry else 'single_program_total_typed_memory',
                'translation_trust_boundary': entry.get('translation_trust_boundary', entry.get('translation_boundary')) if entry else 'Reviewed source translation and Lean source-shape bridge; no general Python/compiler refinement.',
                'original_python': entry.get('original_python') if entry else spec.get('original_python'),
                'candidate_python': entry.get('candidate_python') if entry else spec.get('candidate_python')})
        if not result:
            raise ValueError('extension lacks explicit source selection: ' + task)
    return result

def candidate_python_dependencies(root, extension, view):
    """Bind a candidate graph through its already reviewed package/probe chain."""
    b=view.get('candidate_python_binding')
    if not b or b.get('kind')!='reviewed_native_compile_variant':
        raise ValueError('candidate graph lacks reviewed Python binding')
    packages={p['package']:p for p in extension['packages']}
    if b.get('package') not in packages:raise ValueError('candidate package is not in value registry')
    package=packages[b['package']];deps={}
    def read(name,h):
        deps[name]=h;return load(checked(root,name,h))
    receipt=read(package['receipt'],package['receipt_sha256'])
    review=read(package['root_review'],package['root_review_sha256'])
    if (review.get('reviewer')!='root' or review.get('receipt_sha256')!=package['receipt_sha256']
            or not review.get('status') or re.search(r'(?:^|_)(?:failed|rejected|not_run)(?:_|$)',review['status'])):
        raise ValueError('candidate package root review does not bind receipt')
    native=receipt['native_compile']
    native_name=(Path(package['receipt']).parent/relative(native['receipt'])).as_posix()
    probe=read(native_name,native['receipt_sha256'])
    original=view['original_python'];candidate=view['candidate_python']
    artifact=relative(b['artifact']).as_posix()
    if original!=probe['original_source'] or candidate!=(Path(native_name).parent/artifact).as_posix():
        raise ValueError('candidate/original Python path does not match reviewed probe')
    original_h=probe['original_source_sha256'];candidate_h=probe['artifacts'].get(artifact)
    if (original==candidate or not original.endswith('.py') or not candidate.endswith('.py')
            or receipt['native_source'].get('file')!=original or receipt['native_source'].get('sha256')!=original_h):
        raise ValueError('candidate replaces original or conflicting source hash')
    for name,h in [(original,original_h),(candidate,candidate_h)]:
        checked(root,name,h);deps[name]=h
    return deps, {original:{'sha256':original_h,'variant':'original','authority':'reviewed_candidate_probe_original'},
        candidate:{'sha256':candidate_h,'variant':'candidate','authority':'reviewed_candidate_probe_artifact'}}

def review_manifest(root, evidence_root, manifest):
    """Optional *explicit* final source review; discovery cannot fill this map.

    Each binding names an immutable review document and the exact JSON pointer
    inside it that binds the reviewed final hash. The root reviewer supplies
    this manifest. A naked current-source hash is insufficient.
    """
    result = {}
    if manifest is None:
        return result
    if manifest.get('schema') != 1 or manifest.get('status') != 'accepted_final_source_bindings':
        raise ValueError('unknown source-review manifest schema')
    for name, row in manifest['files'].items():
        relative(name)
        review = row['review']
        document = load(checked(evidence_root, review['path'], review['sha256']))
        receipt_binding = row.get('receipt')
        if (not receipt_binding or document.get('reviewer') != 'root' or not document.get('status')
                or re.search(r'(?:^|_)(?:failed|rejected|not_run)(?:_|$)', document['status'])):
            raise ValueError('final source requires a root review and its receipt')
        receipt = load(checked(evidence_root, receipt_binding['path'], receipt_binding['sha256']))
        if document.get('receipt_sha256') != receipt_binding['sha256']:
            raise ValueError('root review does not bind source receipt')
        if row.get('hash_document') not in {'review', 'receipt'}:
            raise ValueError('unknown source hash document')
        target = document if row['hash_document'] == 'review' else receipt
        pointer = row['hash_pointer']
        valid = False
        if row['hash_document'] == 'review' and pointer == ['reviewed_input_sha256', name]:
            valid = True
        if (row['hash_document'] == 'receipt' and len(pointer) == 3
                and pointer[0] == 'compiled' and pointer[2] == 'source_sha256'):
            record = target['compiled'][pointer[1]]
            valid = record.get('source') == name and type(record.get('exit_code')) is int and record['exit_code'] == 0
        if (row['hash_document'] == 'receipt' and len(pointer) == 3
                and pointer[0] == 'changed_files' and pointer[2] == 'after_sha256'):
            valid = target['changed_files'][pointer[1]].get('path') == name
        if (row['hash_document'] == 'receipt' and len(pointer) == 3
                and pointer[0] == 'changed_sources' and pointer[2] == 'after_sha256'):
            valid = target['changed_sources'][pointer[1]].get('file') == name
        if not valid:
            raise ValueError('review document does not bind a final source revision: ' + name)
        value = target
        for key in pointer:
            value = value[key]
        if value != row['sha256']:
            raise ValueError('review document does not bind final source: ' + name)
        # The input must exactly match the *review's* hash, not an observed hash.
        checked(root, name, row['sha256'])
        result[name] = row
    return result

def validate_supplement(root, evidence_root, supplement, known_tasks):
    """Future V3 admissions use the same explicit roles, including protocol."""
    if supplement.get('schema') != 1:
        raise ValueError('unknown catalog supplement schema')
    tasks = unique_rows(supplement['tasks'], 'task')
    if not set(tasks) <= known_tasks:
        raise ValueError('supplement changes the 184-task population')
    for name, task in tasks.items():
        if not task.get('review') or not task.get('source_sha256'):
            raise ValueError('unreviewed supplement')
        review = task['review']
        evidence = load(checked(evidence_root, review['path'], review['sha256']))
        if (evidence.get('reviewer') != 'root' or not evidence.get('status')
                or re.search(r'(?:^|_)(?:failed|rejected|not_run)(?:_|$)', evidence['status'])
                or review['source_hashes_pointer'] != ['reviewed_input_sha256']):
            raise ValueError('supplement requires explicit successful root final-source review')
        if 'metadata_corrections' in task:
            if evidence.get('reviewed_metadata_corrections_sha256', {}).get(name) != sha(canonical(task['metadata_corrections'])):
                raise ValueError('metadata correction lacks exact supplement review')
        source_hashes = evidence
        for key in review['source_hashes_pointer']:
            source_hashes = source_hashes[key]
        if any(source_hashes.get(p) != h for p, h in task['source_sha256'].items()):
            raise ValueError('supplement source not bound to root review')
        for p, h in task['source_sha256'].items():
            if not p.startswith(BASE + name + '/'):
                raise ValueError('supplement source outside task')
            checked(root, p, h)
        for d in task['declarations']:
            if d['file'] not in task['source_sha256']:
                raise ValueError('supplement public file not bound')
            actual = declarations(root / d['file']).get(d['declaration'])
            if not actual or actual['statement_sha256'] != d.get('statement_sha256'):
                raise ValueError('supplement public statement changed/missing')
            if d['source_variant'] not in VARIANTS:
                raise ValueError('unknown supplement source variant')
            if d['model_kind'] not in NONVALUES | {'backend_value_contract', 'bit_interpreter_contract'}:
                raise ValueError('unknown supplement model')
            if d['model_kind'] in NONVALUES and (d['role'] != d['model_kind'] or d['covers_stages']):
                raise ValueError('protocol/frame/identity promoted to value')
            if d['model_kind'] not in NONVALUES and d['role'] != 'parametric_value_contract':
                raise ValueError('invalid value role')
            if d['model_kind'] == 'bit_interpreter_contract' and d.get('unclosed_numeric_callbacks'):
                raise ValueError('unclosed callback cannot be a closed bit contract')
        for view in task['source_views']:
            verify_source_view(root, view, task['source_sha256'])
    return tasks

def collect(root, frozen, source_map, legacy, *, review=None, evidence_root=None, supplement=None):
    """Observe current files, while keeping reviewed expected hashes separate."""
    root = root.resolve()
    oldtasks = unique_rows(frozen['old']['tasks'], 'task')
    if set(task_directories(root)) - set(oldtasks):
        raise ValueError('task membership differs from frozen 184-task baseline')
    extensions = unique_rows(frozen['registry']['tasks'], 'task')
    entries = unique_rows(frozen['entries'], 'task')
    legacy_tasks = unique_rows(legacy['tasks'], 'task')
    if set(legacy_tasks) != set(oldtasks):
        raise ValueError('legacy source catalog population differs')
    current = unique_rows(frozen['current']['tasks'], 'task')
    if set(current) != set(oldtasks):
        raise ValueError('value registry population differs')
    reviewed = review_manifest(root, evidence_root, review)
    extra = validate_supplement(root, evidence_root, supplement, set(oldtasks)) if supplement else {}
    registered_modules = registered(root)
    allfiles = task_files(root)
    missing_registration = [p for p in allfiles if module_name(p) not in registered_modules]
    # Presence and correctness of registration are distinct from proof success.
    bindings = {p: hash_file(within(root, p)) for p in allfiles}
    python_bindings = {p.relative_to(root).as_posix(): hash_file(p)
        for name in task_directories(root) for p in (root / BASE / name).rglob('*.py')}
    old_expected = load(frozen['root'] / frozen['registry']['baseline']['numerical_obligations_review.json']['path'])['input_sha256']
    python_sources = {}
    coverage = load(frozen['root'] / frozen['registry']['baseline']['coverage_review.json']['path'])
    for port in coverage['ports']:
        name, h = port['python_file'], port['python_sha256']
        row = {'sha256': h, 'variant': 'original', 'authority': 'historical_source_review'}
        if name in python_sources and python_sources[name] != row:
            raise ValueError('conflicting original Python binding')
        python_sources[name] = row
    for entry in entries.values():
        for field, variant in [('original_python', 'original'), ('candidate_python', 'candidate')]:
            name = entry.get(field)
            if name:
                python_sources[name] = {'sha256': entry['input_sha256'][name],
                    'variant': variant, 'authority': 'frozen_explicit_source_admission'}
    variant_evidence = {}
    for task, specs in source_map.items():
        for spec in specs:
            if spec.get('variant','').startswith('compile_repaired_candidate') and task not in entries:
                if task not in extensions:raise ValueError('candidate extension absent')
                deps, variants=candidate_python_dependencies(root,extensions[task],spec)
                variant_evidence.update(deps)
                for name,row in variants.items():
                    old=python_sources.get(name)
                    if old and (old['sha256'],old['variant'])!=(row['sha256'],row['variant']):
                        raise ValueError('conflicting candidate Python binding')
                    if not old:python_sources[name]=row
    missing_python = []
    for name, row in python_sources.items():
        if not (root / name).is_file():
            missing_python.append(name)
        else:
            checked(root, name, row['sha256'])
    expected = {p: (h, 'historical_source_snapshot') for p, h in old_expected.items()}
    for p, h in frozen['current']['input_sha256'].items():
        if p.endswith(('.lean', '.py')):
            expected[p] = (h, 'frozen_final_registry')
    # Explicit final revision overrides are the only way to change an expected hash.
    for p, r in reviewed.items():
        previous = expected.get(p)
        if 'before_expected_sha256' not in r or r['before_expected_sha256'] != (previous[0] if previous else None):
            raise ValueError('stale prior review binding: ' + p)
        expected[p] = (r['sha256'], 'explicit_root_review')
    for t in extra.values():
        expected.update({p: (h, 'explicit_supplement_review') for p, h in t['source_sha256'].items()})
    claims = list(frozen['old']['declarations'])
    claims += [{**d, 'task': t['task']} for t in extensions.values() for d in t['declarations']]
    claims += [{**d, 'task': t['task']} for t in extra.values() for d in t['declarations']]
    claimkeys = [(d['file'], d['declaration']) for d in claims]
    if len(claimkeys) != len(set(claimkeys)):
        raise ValueError('duplicate public claim')
    public_files = {d['file'] for d in claims}
    actual = {}
    for p in allfiles:
        ds = declarations(root / p)
        if p in public_files or Path(p).parent == Path(BASE + Path(p).parts[2]):
            for name, d in ds.items():
                if d['kind'] == 'specification':
                    actual[p, name] = d['statement_sha256']
    wanted = {(d['file'], d['declaration']): d['statement_sha256'] for d in claims}
    if actual != wanted:
        raise ValueError('unreviewed/missing/changed public specifications: ' + repr(
            sorted(k for k in set(actual) | set(wanted) if actual.get(k) != wanted.get(k))[:20]))
    tasks = []
    for name in sorted(oldtasks):
        ext, admission = extensions.get(name), entries.get(name)
        views = source_views(name, ext, admission, source_map)
        if ext:
            primary = [v for v in views if v['variant'] == ext['primary_source_variant']]
            if len(primary) != 1:
                raise ValueError('primary source graph differs from value registry: ' + name)
            if any(p not in ext['source_sha256'] for v in views for p in v['selected_definitions']):
                raise ValueError('unreviewed helper selected as value source: ' + name)
            if admission and primary[0]['selected_definitions'] != {admission['typed_source']: [admission['typed_definition']]}:
                raise ValueError('explicit admission source selector changed: ' + name)
        if name in extra:
            views += extra[name]['source_views']
        for view in views:
            verify_source_view(root, view, bindings)
        source_files = {p for view in views for p in view['selected_definitions']}
        old_entry = legacy_tasks[name]['source_entry']
        if old_entry:
            if old_entry not in bindings:
                raise ValueError('historical source entry missing: ' + old_entry)
            source_files.add(old_entry)
        files = []
        for p in allfiles:
            if Path(p).parts[2] != name:
                continue
            roles = []
            if p in source_files:
                roles.append('selected_source')
            if p in public_files:
                roles.append('public_contract')
            if not roles:
                roles.append('helper')
            exp = expected.get(p)
            files.append({'path': p, 'module': module_name(p), 'roles': roles,
                'observed_sha256': bindings[p], 'review_expected_sha256': exp[0] if exp else None,
                'review_authority': exp[1] if exp else None,
                'review_binding_matches': bool(exp and exp[0] == bindings[p]),
                'libraries': registered_modules.get(module_name(p), [])})
        task_claims = [d for d in claims if Path(d['file']).parts[2] == name]
        original_sources = bool(old_entry or any(v['variant'] == 'original_source_translation' for v in views))
        catalog_task = {'task': name,
            'historical_source_entry': old_entry,
            'historical_source_audit': legacy_tasks[name].get('source_audit'),
            'source_views': views, 'files': files,
            'declarations': task_claims,
            'original_source_present': original_sources,
            'candidate_source_present': any(v['variant'].startswith('compile_repaired_candidate') for v in views),
            'value_registry_row': current[name],
            'native_functional_correctness': 'not_established_by_this_adapter'}
        if name in extra and 'metadata_corrections' in extra[name]:
            correction_review = extra[name]['review']
            review_path = checked(evidence_root, correction_review['path'], correction_review['sha256'])
            corrections = validate_metadata_corrections(name, current[name],
                extra[name]['metadata_corrections'], load(review_path))
            catalog_task['reviewed_metadata_corrections'] = corrections
            catalog_task['metadata_correction_review'] = {
                'path': review_path.resolve().relative_to(root).as_posix(),
                'sha256': correction_review['sha256']}
        tasks.append(catalog_task)
    build_inputs = {Path(*m.split('.')).with_suffix('.lean').as_posix():
        hash_file(within(root, Path(*m.split('.')).with_suffix('.lean').as_posix()))
        for m in registered_modules}
    for p in ('lakefile.toml', 'lean-toolchain', 'lake-manifest.json'):
        build_inputs[p] = hash_file(within(root, p))
    return {'schema': 1, 'kind': 'current_source_catalog_not_completion_ledger',
        'metadata_receipt_sha256': frozen['receipt_sha256'],
        'source_map_sha256': sha(canonical(source_map)),
        'legacy_inventory_sha256': sha(canonical(legacy)),
        'review_manifest_sha256': sha(canonical(review)) if review else None,
        'supplement_sha256': sha(canonical(supplement)) if supplement else None,
        'expected_task_names': sorted(oldtasks),
        'historical_public_count': len(frozen['old']['declarations']),
        'historical_original41_count': len(frozen['old41']['tasks']),
        'registered_modules': registered_modules, 'build_input_sha256': build_inputs,
        'python_input_sha256': python_bindings,
        'python_sources': python_sources,
        'source_variant_evidence_sha256': variant_evidence,
        'missing_bound_python_sources': sorted(missing_python),
        'unregistered_optional_regressions': sorted(p.relative_to(root).as_posix()
            for p in (root / 'bench/tests').rglob('*.lean')
            if module_name(p.relative_to(root).as_posix()) not in registered_modules),
        'unregistered_task_modules': missing_registration, 'tasks': tasks,
        'summary': {'tasks': len(tasks), 'task_modules': len(allfiles),
            'public_files': len(public_files), 'public_claims': len(claims),
            'extension_value_contracts': sum(d.get('role') == 'parametric_value_contract' for t in extensions.values() for d in t['declarations']),
            'protocol_contracts': sum(d.get('model_kind') == 'protocol_contract' for d in claims),
            'original_source_missing': [t['task'] for t in tasks if not t['original_source_present']],
            'candidate_sources': [t['task'] for t in tasks if t['candidate_source_present']],
            'task_local_sc_sources': [t['task'] for t in tasks if any(v['frontend'] == 'task_local_exact_ast' for v in t['source_views'])],
            'source_review_binding_gaps': [f['path'] for t in tasks for f in t['files'] if not f['review_binding_matches']]},
        'interpretation': 'Current file/source-role inventory only. Value/model/stage claims are unchanged registry rows. Presence, metadata checks, software interpretation, native refinement, concurrency progress and independent replay are separate obligations.'}

def validate_catalog(root, catalog):
    if catalog.get('schema') != 1 or catalog.get('kind') != 'current_source_catalog_not_completion_ledger':
        raise ValueError('unknown catalog schema')
    rows = unique_rows(catalog['tasks'], 'task')
    if sorted(rows) != catalog['expected_task_names'] or set(task_directories(root)) - set(rows):
        raise ValueError('catalog task membership differs')
    files = [f for t in rows.values() for f in t['files']]
    actual_python = {p.relative_to(root).as_posix(): hash_file(p)
        for name in task_directories(root) for p in (root / BASE / name).rglob('*.py')}
    if actual_python != catalog.get('python_input_sha256', {}):
        raise ValueError('Python source inventory changed or omitted')
    for name,h in catalog.get('source_variant_evidence_sha256',{}).items():checked(root,name,h)
    missing_python = []
    for name, row in catalog.get('python_sources', {}).items():
        if not (root / name).is_file():
            missing_python.append(name)
        else:
            checked(root, name, row['sha256'])
    if sorted(missing_python) != catalog.get('missing_bound_python_sources', []):
        raise ValueError('bound Python source presence changed')
    bypath = unique_rows(files, 'path')
    if task_files(root) != sorted(bypath):
        raise ValueError('catalog omits or adds a task Lean file')
    for path, f in bypath.items():
        checked(root, path, f['observed_sha256'])
        if f['module'] != module_name(path):
            raise ValueError('module/path mismatch')
        name = Path(path).parts[2]
        if f not in rows[name]['files']:
            raise ValueError('module assigned to wrong task')
        selected = path == rows[name]['historical_source_entry'] or any(
            path in v['selected_definitions'] for v in rows[name]['source_views'])
        public = any(d['file'] == path for d in rows[name]['declarations'])
        wanted = (['selected_source'] if selected else []) + (['public_contract'] if public else [])
        if f['roles'] != (wanted or ['helper']):
            raise ValueError('helper/source/public role drift')
        if f['libraries'] != catalog['registered_modules'].get(f['module'], []):
            raise ValueError('file library assignment differs')
    for row in rows.values():
        for view in row['source_views']:
            verify_source_view(root, view, {p: f['observed_sha256'] for p, f in bypath.items()})
        original = bool(row['historical_source_entry'] or any(v['variant'] == 'original_source_translation' for v in row['source_views']))
        if row['original_source_present'] is not original:
            raise ValueError('candidate/model promoted to original source')
        if 'reviewed_metadata_corrections' in row:
            review = row.get('metadata_correction_review') or {}
            document = load(checked(root, review.get('path'), review.get('sha256')))
            validate_metadata_corrections(row['task'], row['value_registry_row'],
                row['reviewed_metadata_corrections'], document)
    regs = registered(root)
    if regs != catalog['registered_modules']:
        raise ValueError('Lake registration changed')
    actual_public = {}
    wanted_public = {(d['file'], d['declaration']): d['statement_sha256']
        for t in rows.values() for d in t['declarations']}
    public_paths = {f for f, _ in wanted_public}
    for file in bypath:
        if file in public_paths or len(Path(file).parts) == 4:
            actual_public.update({(file, name): d['statement_sha256'] for name, d in declarations(root/file).items()
                                  if d['kind'] == 'specification'})
    if actual_public != wanted_public:
        raise ValueError('public declaration omitted, changed or unreviewed')
    for path, expected in catalog['build_input_sha256'].items():
        checked(root, path, expected)
    wanted = {Path(*m.split('.')).with_suffix('.lean').as_posix() for m in regs}
    wanted |= {'lakefile.toml', 'lean-toolchain', 'lake-manifest.json'}
    if set(catalog['build_input_sha256']) != wanted:
        raise ValueError('build source/config input omitted')
    missing = [f['path'] for f in files if f['module'] not in regs]
    if missing != catalog['unregistered_task_modules']:
        raise ValueError('registration gap hidden')
    return {'catalog_integrity': 'passed', 'registered_all_task_modules': not missing,
            'task_count': len(rows), 'task_modules': len(files)}

def main():
    if sys.argv[1:2] == ['targets']:
        p = argparse.ArgumentParser(description=trust_targets.__doc__)
        p.add_argument('--root', type=Path, required=True)
        p.add_argument('--catalog', type=Path, required=True)
        p.add_argument('tasks', nargs='*')
        a = p.parse_args(sys.argv[2:])
        for path in trust_targets(a.root.resolve(), load(a.catalog), a.tasks):
            print(path)
        return 0
    if sys.argv[1:2]==['prepare-baseline']:
        p=argparse.ArgumentParser(description=prepare_baseline.__doc__)
        p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        a=p.parse_args(sys.argv[2:]);print(__import__('json').dumps(prepare_baseline(a.package,a.output)));return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--source-map', type=Path, required=True)
    p.add_argument('--legacy-inventory', type=Path, required=True)
    p.add_argument('--evidence-root', type=Path)
    p.add_argument('--review-manifest', type=Path)
    p.add_argument('--supplement', type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--check', type=Path)
    a = p.parse_args()
    result = collect(a.root, baseline(a.baseline), load(a.source_map), load(a.legacy_inventory),
        review=load(a.review_manifest) if a.review_manifest else None,
        evidence_root=a.evidence_root,
        supplement=load(a.supplement) if a.supplement else None)
    validate_catalog(a.root, result)
    if a.check:
        if load(a.check) != result:
            raise ValueError('stale current source catalog')
    else:
        write_json(a.output, result)
    print(__import__('json').dumps(result['summary'], indent=2))

if __name__ == '__main__':
    main()
