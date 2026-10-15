#!/usr/bin/env python3
"""Fail-closed review gate for all 183 original-program scopes.

This validates explicit substantive reviews and their exact source/proof
bindings. It does not infer semantic coverage from compilation, existing task
labels, or hashes, and it never treats pending work as numerical deferral.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import math
from pathlib import Path
import sys

from adapter_common import BASE, canonical, checked, hash_file, load, sha, unique_rows, write_json
from current_source_catalog import validate_catalog
from exact_scope_gate import validated_exact_requirements, validate_exact_case_links

EXCLUDED_TASK = 'fp4_to_bf16_conversion'
QUEUE = BASE + 'deferred_numerical_experiments.json'
DIMENSIONS = ('stages', 'dtypes', 'branches')
ACCEPTED = {'verified', 'verified_with_deferred_numerics'}
RULE_STATES = {'pending_exact_binding', 'existing_report_optional_replay', 'shared_model_check', 'pending_numeric_validation'}
COMPLETE_DTYPE_RELATIONS = {'complete_enumerated_wrapper_dtype_domain',
                            'complete_original_declared_domain'}
DTYPES = {'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
          'uint64', 'fp8e4m3fn', 'fp8e4m3fnuz', 'fp8e5m2', 'fp8e5m2fnuz',
          'fp16', 'bf16', 'fp32', 'fp64', 'packed_fp4e2m1'}


def validate_case(case):
    """Check a concrete dtype/constexpr case, not a prose domain claim.

    Exhaustiveness of these cases and correspondence of a theorem to Python
    still require semantic review. This gate only checks exact, typed metadata
    compatibility; it does not derive either fact from a label or a hash.
    """
    if not isinstance(case, dict) or set(case) != {'dtype_profile', 'configuration'}:
        raise ValueError('scope case requires dtype_profile and configuration')
    profile = case['dtype_profile']
    if not isinstance(profile, dict) or set(profile) != {'inputs', 'outputs', 'compute'}:
        raise ValueError('dtype profile requires explicit input/output/compute maps')
    for channel, ports in profile.items():
        if (not isinstance(ports, dict) or (channel != 'compute' and not ports)
                or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or v not in DTYPES
                       for k, v in ports.items())):
            raise ValueError('dtype profile contains missing/unknown dtype ports')

    def concrete(value):
        if isinstance(value, dict):
            return bool(value) and all(isinstance(k, str) and k.strip() and concrete(v)
                                       for k, v in value.items())
        if isinstance(value, list):
            return bool(value) and all(concrete(v) for v in value)
        if isinstance(value, str):
            return bool(value.strip())
        return isinstance(value, (bool, int)) or isinstance(value, float) and math.isfinite(value)

    if not isinstance(case['configuration'], dict) or not concrete(case['configuration']):
        raise ValueError('scope case lacks a concrete configuration')
    return canonical(case)


def case_set(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError('scope obligation requires explicit nonempty cases')
    result = {validate_case(case) for case in cases}
    if len(result) != len(cases):
        raise ValueError('duplicate scope case')
    return result


def declaration_scope(declaration, inventory):
    scope = declaration.get('original_scope')
    if not isinstance(scope, dict) or set(scope) != {'source_inventory_sha256', *DIMENSIONS, 'cases'}:
        raise ValueError('full execution evidence lacks explicit original scope metadata')
    if scope['source_inventory_sha256'] != sha(canonical(inventory)):
        raise ValueError('execution evidence original source inventory changed')
    ids = set()
    for dimension in DIMENSIONS:
        values = scope[dimension]
        expected = {item['id'] for item in inventory[dimension]}
        if not isinstance(values, list) or len(values) != len(set(values)) or not set(values) <= expected:
            raise ValueError('execution evidence has unknown/duplicate original scope IDs')
        ids.update(values)
    if not scope['stages'] or declaration.get('covers_stages') != scope['stages']:
        raise ValueError('execution evidence stage labels do not match original stage IDs')
    stage_ids = set(scope['stages'])
    if any(item['stage_id'] not in stage_ids for item in inventory['branches']
           if item['id'] in scope['branches']):
        raise ValueError('execution evidence branch belongs to an uncovered stage')
    stage_paths = {item['path'] for item in inventory['stages'] if item['id'] in stage_ids}
    if any(item['path'] not in stage_paths for item in inventory['dtypes']
           if item['id'] in scope['dtypes']):
        raise ValueError('execution evidence dtype domain has no covered source stage')
    cases = scope['cases']
    if not isinstance(cases, dict) or set(cases) != ids:
        raise ValueError('execution evidence case map omits/adds scope obligations')
    for values in cases.values():
        case_set(values)
    return scope


def original_python(catalog, task):
    prefix = BASE + task + '/'
    sources = catalog.get('python_sources', {})
    rows = unique_rows(catalog['tasks'], 'task')
    result = {p: row['sha256'] for p, row in sources.items()
              if p.startswith(prefix) and row['variant'] == 'original'}
    for view in rows[task]['source_views']:
        path = view.get('original_python')
        if path is None:
            continue
        binding = sources.get(path)
        if not binding or binding.get('variant') != 'original' or not binding.get('sha256'):
            raise ValueError('explicit original Python lacks original catalog binding: ' + str(path))
        if path.startswith(BASE) and not path.startswith(prefix):
            raise ValueError('original Python belongs to another task: ' + path)
        owners = {row['task'] for row in rows.values() for source in row['source_views']
                  if source.get('original_python') == path}
        if owners != {task}:
            raise ValueError('original Python assigned to multiple tasks: ' + path)
        result[path] = binding['sha256']
    return dict(sorted(result.items()))


def source_inventory(root, bindings):
    """Every original function/control node is retained for explicit review.

    Test functions are still inventoried. Only a review may classify a
    non-JIT test_* function as a harness, with a checked production-call guard.
    A per-file dtype obligation is mandatory even without literal dtype names:
    absence of a dtype token does not prove an unconstrained input domain.
    """
    result = {dimension: [] for dimension in DIMENSIONS}
    for path, expected in sorted(bindings.items()):
        text = checked(root, path, expected).read_text()
        tree = ast.parse(text)
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        production_calls = {node.func.id for fun in functions if not fun.name.startswith('test_')
                            for node in ast.walk(fun) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)}
        for fun in functions:
            node_id = f'{path}:{fun.lineno}:{fun.name}'
            decorators = [ast.unparse(item) for item in fun.decorator_list]
            harness = (fun.name.startswith('test_') and fun.name not in production_calls
                       and not any('jit' in item for item in decorators))
            result['stages'].append({'id': 'stage:' + node_id, 'path': path,
                'function': fun.name, 'line': fun.lineno, 'end_line': fun.end_lineno,
                'decorators': decorators, 'may_be_test_harness': harness})
            for node in ast.walk(fun):
                if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Match)):
                    kind = type(node).__name__
                    if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                        expression = ast.unparse(node.test)
                    elif isinstance(node, (ast.For, ast.AsyncFor)):
                        expression = ast.unparse(node.target) + ' in ' + ast.unparse(node.iter)
                    else:
                        expression = ast.unparse(node.subject)
                    result['branches'].append({'id': f'branch:{node_id}:{node.lineno}:{node.col_offset}:{kind}',
                        'path': path, 'function': fun.name, 'line': node.lineno,
                        'kind': kind, 'expression': expression, 'stage_id': 'stage:' + node_id})
        dtype_tokens = sorted({ast.unparse(node) for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and
            (re_dtype(node.attr) or node.attr in {'dtype', 'to', 'cast'})})
        result['dtypes'].append({'id': 'dtype-domain:' + path, 'path': path,
            'literal_mentions': dtype_tokens,
            'requirement': 'All original input/output/compute dtype dispatches and implicit dtype parameters.'})
    return result


def re_dtype(name):
    return name.startswith(('float', 'bfloat', 'fp', 'int', 'uint')) and any(c.isdigit() for c in name)


def task_bindings(root, catalog, task):
    lean = {f['path']: f['observed_sha256'] for f in task['files']}
    prefix = BASE + task['task'] + '/'
    python = {p: h for p, h in catalog['python_input_sha256'].items() if p.startswith(prefix)}
    python.update(original_python(catalog, task['task']))
    for p, h in {**lean, **python}.items():
        checked(root, p, h)
    return {'lean': lean, 'python': python}


def full_execution_evidence(task, declaration):
    """Supporting identities, frames and fixed profiles cannot cover a dimension."""
    variant = declaration.get('source_variant')
    if variant not in (None, 'original_source_translation'):
        return False
    if variant is None and not task['historical_source_entry']:
        return False
    if declaration.get('coverage_level') not in (None, 'full_value_candidate'):
        return False
    if declaration.get('scope_relation') in ('strict_specialization', 'not_an_execution_contract'):
        return False
    if 'model_kind' in declaration:
        return (declaration['model_kind'] in ('backend_value_contract', 'bit_interpreter_contract')
                and declaration.get('role') == 'parametric_value_contract'
                and declaration.get('scope_relation') == 'full_declared_stage')
    return (task['historical_source_entry'] is not None
            and declaration.get('fp_model_review', {}).get('kind') in {
                'mathematical_contract', 'backend_equivalence', 'bit_interpreter_contract',
                'backend_value_contract', 'rounding_model_contract'})


def validate_deferrals(task, deferrals, queue):
    rules = unique_rows(queue['rules'], 'id')
    result = {}
    for item in deferrals:
        ident = item.get('assumption_id')
        if ident not in rules:
            raise ValueError('unregistered numerical assumption: ' + str(ident))
        rule = rules[ident]
        if rule['status'] not in RULE_STATES or rule.get('queue_status') == 'refuted_unrestricted_assumption':
            raise ValueError('refuted/nondeferrable numerical assumption: ' + ident)
        if task not in rule['affected_tasks']:
            raise ValueError('numerical assumption belongs to another task: ' + ident)
        if item.get('kind') != 'numerical_rule_conformance':
            raise ValueError('only numerical rule conformance may be deferred: ' + ident)
        if item.get('rule_sha256') != sha(canonical(rule)):
            raise ValueError('numerical rule/profile changed: ' + ident)
        bindings = unique_rows(rule.get('scope_bindings', []), 'id')
        binding_id = item.get('scope_binding_id')
        binding = bindings.get(binding_id)
        if not binding or binding.get('task') != task:
            raise ValueError('numerical assumption lacks a registered task scope binding: ' + ident)
        if set(binding) != {'id', 'task', 'declarations', 'callbacks', 'dtype_profile', 'configuration'}:
            raise ValueError('numerical scope binding has an unknown schema: ' + ident)
        case = {key: binding[key] for key in ('dtype_profile', 'configuration')}
        validate_case(case)
        actual_case = {key: item.get(key) for key in ('dtype_profile', 'configuration')}
        validate_case(actual_case)
        if actual_case != case:
            raise ValueError('numerical dtype/configuration does not match registered binding: ' + ident)
        declarations = binding['declarations']
        if not isinstance(declarations, list) or not declarations:
            raise ValueError('numerical scope binding has no exact declaration')
        for declaration in declarations:
            if (not isinstance(declaration, dict)
                    or set(declaration) != {'file', 'declaration', 'statement_sha256'}
                    or not all(isinstance(v, str) and v.strip() for v in declaration.values())
                    or not declaration['file'].startswith(BASE + task + '/')
                    or len(declaration['statement_sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in declaration['statement_sha256'])):
                raise ValueError('numerical scope binding has an invalid declaration')
        if len({canonical(d) for d in declarations}) != len(declarations):
            raise ValueError('duplicate numerical scope declaration')
        callbacks = binding['callbacks']
        if (not isinstance(callbacks, list) or any(not isinstance(c, str) or not c.strip() for c in callbacks)
                or len(callbacks) != len(set(callbacks))):
            raise ValueError('invalid registered numerical callbacks')
        if not isinstance(item.get('required_validation'), str) or not item['required_validation'].strip():
            raise ValueError('numerical assumption lacks follow-up validation: ' + ident)
        key = (ident, binding_id)
        if key in result:
            raise ValueError('duplicate deferred numerical assumption scope binding')
        result[key] = binding
    return result


def validate_used_numerics(task, row, public, proof_cases, queue, *, root=None):
    """Match every used theorem/callback/case to an exact queue registration.

    The queue registration is the semantic authority for which operation a
    numerical rule concerns. No operation compatibility is inferred from prose
    or a rule name. Unregistered combinations fail closed.
    """
    deferrals = row.get('deferred_numerical_assumptions', [])
    bindings = validate_deferrals(task, deferrals, queue)
    exact = validated_exact_requirements(root, task, public, proof_cases, queue)

    def required_callbacks(proof, case):
        if proof in exact:
            return exact[proof][case]['required_callbacks']
        return public[proof].get('unclosed_numeric_callbacks', [])

    callbacks = {callback for key, cases in proof_cases.items() for case in cases
                 for callback in required_callbacks(key, case)}
    callback_bindings = row.get('callback_assumptions', {})
    if not isinstance(callback_bindings, dict) or set(callback_bindings) != callbacks:
        raise ValueError('unclosed numerical callbacks lack exact registered assumptions')
    callback_links = {}
    for callback, references in callback_bindings.items():
        if not isinstance(references, list) or not references:
            raise ValueError('callback requires explicit numerical scope binding references')
        keys = set()
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {'assumption_id', 'scope_binding_id'}:
                raise ValueError('callback requires exact assumption/scope binding IDs')
            key = (reference['assumption_id'], reference['scope_binding_id'])
            if key not in bindings or callback not in bindings[key]['callbacks']:
                raise ValueError('callback is incompatible with registered numerical rule')
            if key in keys:
                raise ValueError('duplicate callback scope binding')
            keys.add(key)
        callback_links[callback] = keys

    used_bindings, used_links = set(), set()
    for proof, cases in proof_cases.items():
        declaration = public[proof]
        reference = {key: declaration[key] for key in ('file', 'declaration', 'statement_sha256')}
        required_ids = set()
        for field in ('extra_numeric_ids', 'explicit_law_ids', 'extra_numeric_law_ids',
                      'model_requirement_ids', 'shared_model_ids'):
            required_ids.update(declaration.get(field, []))
        for case in cases:
            applicable = {key for key, binding in bindings.items()
                if reference in binding['declarations']
                and canonical({k: binding[k] for k in ('dtype_profile', 'configuration')}) == case}
            validate_exact_case_links(exact, proof, case, applicable, callback_links)
            for ident in required_ids:
                matches = {key for key in applicable if key[0] == ident}
                if not matches:
                    raise ValueError('used numerical assumptions omitted or wrong declaration/dtype/configuration: ' + ident)
                used_bindings.update(matches)
            for callback in required_callbacks(proof, case):
                matches = applicable & callback_links[callback]
                if not matches:
                    raise ValueError('callback lacks compatible declaration/dtype/configuration binding: ' + callback)
                used_bindings.update(matches)
                used_links.update((callback, key) for key in matches)
    if used_bindings != set(bindings):
        raise ValueError('unrelated numerical assumptions added')
    if used_links != {(callback, key) for callback, keys in callback_links.items() for key in keys}:
        raise ValueError('unrelated callback scope bindings added')
    return deferrals, callback_bindings


def validate_task(root, catalog, task, row, queue):
    name = task['task']
    if row.get('status') not in ACCEPTED:
        raise ValueError('scope remains ' + str(row.get('status')))
    if row.get('scope_schema') != 2:
        raise ValueError('accepted scope requires explicit schema 2 evidence/case bindings')
    if row.get('structural_gaps') != []:
        raise ValueError('unresolved structural/scope gaps cannot be deferred')
    if task.get('original_source_present') is not True:
        raise ValueError('original-source translation is still missing')
    if row.get('input_sha256') != task_bindings(root, catalog, task):
        raise ValueError('task Lean/Python bindings changed or incomplete')
    originals = original_python(catalog, name)
    if not originals:
        raise ValueError('no original Python source binding')
    inventory = source_inventory(root, originals)
    if row.get('source_inventory') != inventory:
        raise ValueError('original source function/dtype/branch inventory differs')
    coverage = row.get('coverage', {})
    if set(coverage) != set(DIMENSIONS):
        raise ValueError('missing scope dimension')
    public = {(d['file'], d['declaration']): d for d in task['declarations']}
    proof_cases, proof_scopes = {}, {}
    stages = unique_rows(inventory['stages'], 'id')
    stage_coverage = unique_rows(coverage['stages'], 'id')
    for dimension in DIMENSIONS:
        expected = unique_rows(inventory[dimension], 'id')
        actual = unique_rows(coverage[dimension], 'id')
        if set(actual) != set(expected):
            raise ValueError('missing/extra ' + dimension + ' obligations')
        for ident, obligation in actual.items():
            if not isinstance(obligation.get('explanation'), str) or not obligation['explanation'].strip():
                raise ValueError('scope obligation lacks review explanation: ' + ident)
            if obligation.get('status') == 'test_harness':
                stage_id = ident if dimension == 'stages' else expected[ident].get('stage_id')
                if (dimension == 'dtypes' or not stage_id or not stages[stage_id]['may_be_test_harness']
                        or stage_coverage.get(stage_id, {}).get('status') != 'test_harness'):
                    raise ValueError('production scope cannot be excluded as test harness: ' + ident)
                continue
            if obligation.get('status') != 'covered':
                raise ValueError('unreviewed/uncovered scope obligation: ' + ident)
            required_cases = case_set(obligation.get('cases'))
            proofs = obligation.get('proofs')
            if not proofs:
                raise ValueError('covered scope lacks public proof evidence: ' + ident)
            covered_cases = set()
            for proof in proofs:
                key = (proof.get('file'), proof.get('declaration'))
                declaration = public.get(key)
                if declaration is None or declaration['statement_sha256'] != proof.get('statement_sha256'):
                    raise ValueError('unknown/changed scope proof: ' + repr(key))
                if not full_execution_evidence(task, declaration):
                    proof_cases.setdefault(key, set()).update(required_cases)
                    continue
                if key not in proof_scopes:
                    proof_scopes[key] = declaration_scope(declaration, inventory)
                scope = proof_scopes[key]
                if ident not in scope[dimension]:
                    continue
                if (dimension == 'dtypes'
                        and declaration.get('dtype_scope_relation') not in COMPLETE_DTYPE_RELATIONS):
                    continue
                evidence_cases = case_set(scope['cases'][ident])
                covered_cases.update(evidence_cases)
                proof_cases.setdefault(key, set()).update(required_cases & evidence_cases)
            if not covered_cases:
                raise ValueError('no full original execution contract for scope obligation: ' + ident)
            if covered_cases != required_cases:
                raise ValueError('execution evidence does not match required dtype/configuration cases: ' + ident)
    deferrals, callback_bindings = validate_used_numerics(name, row, public, proof_cases, queue, root=root)
    if (row['status'] == 'verified_with_deferred_numerics') != bool(deferrals):
        raise ValueError('conditional status and numerical deferral list disagree')
    review_ref = row.get('review') or {}
    review = load(checked(root, review_ref.get('path'), review_ref.get('sha256')))
    expected = {'schema': 1, 'scope_schema': 2, 'kind': 'original_program_scope_review', 'task': name,
        'reviewer': 'root', 'status': 'reviewed', 'catalog_sha256': sha(canonical(catalog)),
        'input_sha256': row['input_sha256'], 'source_inventory_sha256': sha(canonical(inventory)),
        'coverage_sha256': sha(canonical(coverage)),
        'deferred_numerical_sha256': sha(canonical(deferrals)),
        'callback_assumptions_sha256': sha(canonical(callback_bindings)), 'structural_gaps': []}
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError('scope review does not bind exact current obligations/evidence')
    if not isinstance(review.get('semantic_review'), str) or not review['semantic_review'].strip():
        raise ValueError('scope review lacks substantive scope explanation')
    return row['status']


def check(root, catalog, ledger):
    validate_catalog(root, catalog)
    if ledger.get('schema') != 1 or ledger.get('kind') != 'original_program_scope_ledger':
        raise ValueError('unknown original-program scope ledger')
    tasks = unique_rows(catalog['tasks'], 'task')
    expected = sorted(set(tasks) - {EXCLUDED_TASK})
    if len(tasks) != 184 or len(expected) != 183 or EXCLUDED_TASK not in tasks:
        raise ValueError('original-program population must remain exactly 183 of 184 tasks')
    if ledger.get('expected_tasks') != expected or ledger.get('excluded_tasks') != [EXCLUDED_TASK]:
        raise ValueError('scope ledger changes the requested 183-task population')
    rows = unique_rows(ledger['tasks'], 'task')
    if set(rows) != set(expected):
        raise ValueError('scope ledger omitted/added a requested task')
    errors = []
    if ledger.get('catalog_sha256') != sha(canonical(catalog)):
        errors.append('scope ledger is unbound or belongs to another catalog')
    queue_ref = ledger.get('deferred_queue') or {}
    if queue_ref.get('path') != QUEUE:
        raise ValueError('scope ledger must bind the unified numerical queue')
    queue = load(checked(root, QUEUE, queue_ref.get('sha256')))
    counts = Counter()
    results = []
    for name in expected:
        try:
            status = validate_task(root, catalog, tasks[name], rows[name], queue)
            counts[status] += 1
            results.append({'task': name, 'status': status})
        except (ValueError, OSError, KeyError, TypeError) as error:
            counts['not_accepted'] += 1
            results.append({'task': name, 'status': 'not_accepted', 'error': str(error)})
    if counts['not_accepted']:
        errors.append(f'{counts["not_accepted"]} original-program scopes remain unaccepted')
    return {'schema': 1, 'status': 'failed' if errors else 'passed',
        'requested_tasks': 183, 'counts': dict(counts), 'errors': errors, 'tasks': results,
        'boundary': 'Source-scope review only; full build, axiom and official comparator replay are independent gates. Deferred numerical rules remain assumptions, not validated hardware facts.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = check(args.root.resolve(), load(args.catalog), load(args.ledger))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'failed', 'errors': [str(error)]}
    if args.output:
        write_json(args.output, result)
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    sys.exit(main())
