#!/usr/bin/env python3
"""Current-layout structural and global gates, without acceptance by omission.

The catalog replaces filename inference and frozen proof-gap ledger rewriting.
Its regeneration, scope review, existing source comparisons, value metadata,
and official library replay are mandatory independent gates. Corpus axiom,
specification and comparator replay remains audit_trust.sh --catalog; this
runner never claims that a successful global run completed that corpus gate.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from adapter_common import (canonical, checked, declarations, hash_file, load, sha,
                            split_statement, write_json)
from audit_source import bare_true_proof, lean_code_mask, lean_first_preamble, strip_lean_comments
from current_source_catalog import validate_catalog


IDENT = r"[A-Za-z_][A-Za-z0-9_']*"


def declaration_result(text):
    """Return the declared result, excluding binders, comments and proof text.

    This deliberately recognizes source syntax, not Lean elaborated theorem
    types. Unsupported syntax fails closed; Lean/spec/comparator workers and
    catalog reproduction remain independent mandatory gates.
    """
    statement, _ = split_statement(text)
    code = lean_code_mask(statement)
    depth = 0
    for i, char in enumerate(code):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ':' and depth == 0:
            return code[i + 1:].strip()
    return ''


def result_after_lets(result):
    """Separate the simple, complete-line let bindings used by SC contracts."""
    prefix = ''
    while match := re.match(rf'let\s+{IDENT}\s*:=\s*[^\n]+\n\s*', result):
        prefix += match[0]
        result = result[match.end():]
    return prefix, result


def source_token(names):
    # Macro-selected shared DSL bodies need not be named in a theorem. The SC
    # compiler, however, must receive an explicitly selected source definition.
    return '(?:' + '|'.join(re.escape(n) for n in names) + r")(?![A-Za-z0-9_'])"


def weakened_result(result):
    """An execution in a disjunction/implication antecedent is not a result.

    Arrows inside parenthesized frame observations remain permitted. This is
    conservative source recognition, not a general Lean proposition parser.
    """
    depth = 0
    for char in result:
        if char in '([{⟨':
            depth += 1
        elif char in ')]}⟩':
            depth -= 1
        elif depth == 0 and char in '→∨':
            return True
    return False


def sc_value_alias(result, public_decls, selected):
    """Inspect a directly invoked, same-file Prop definition, never a helper hit.

    This is the task-local trace/value surface: independent controlTrace and
    replay, actual run, and data/lock/register observations. A predicate merely
    named ValueContract (or mentioning Backend) is not a recognized surface.
    """
    if any(op in result for op in ('→', '∨', '∧', '=')):
        return None
    call = re.match(rf'({IDENT})\s+{selected}(?:\.{IDENT})?\s', result)
    if not call:
        return None
    definition = public_decls.get(call[1])
    if not definition or definition['kind'] != 'def' or declaration_result(definition['text']) != 'Prop':
        return None
    header, body = split_statement(definition['text'])
    parameter = re.match(rf'def\s+{re.escape(call[1])}\s+\(({IDENT})\s*:\s*List\s+Command\)',
                         lean_code_mask(header).strip())
    if not parameter or not body.startswith(':='):
        return None
    source = re.escape(parameter[1])
    prefix, body = result_after_lets(lean_code_mask(body[2:]).strip())
    trace = re.search(rf'let\s+({IDENT})\s*:=\s*controlTrace\s+{source}\b', prefix)
    reference = re.search(rf'let\s+({IDENT})\s*:=\s*replay\b', prefix)
    if not trace or not reference or weakened_result(body):
        return None
    execution = re.match(rf'∃\s+({IDENT})\s*,\s*run\b[^=∧∨→]*\(initial\s+{source}\b'
                         rf'[^=∧∨→]*=\s*\(\s*{re.escape(trace[1])}\s*,\s*\1\s*\)\s*∧', body)
    if not execution:
        return None
    final, ref = re.escape(execution[1]), re.escape(reference[1])
    observations = body[execution.end():]
    if not all(re.search(pattern, observations) for pattern in (
            rf'{final}\.mem\.data\b[^=∧∨→]*=\s*denote\b[^∧∨→]*\b{ref}\.data\b',
            rf'{final}\.mem\.locks\s*=\s*{ref}\.locks\b',
            rf'{final}\.regs\b[^=∧∨→]*=\s*denote\b[^∧∨→]*\b{ref}\.regs\b')):
        return None
    return definition['name']


def reviewed_named_surface(root, task):
    """Recognize reviewed typed/SC public surfaces without expanding task scope.

    Eligibility comes from exact catalog statement/file/source bindings and
    the reviewed role and source variant. It is not inferred from task names,
    arbitrary backend predicates, comments, theorem hypotheses or proof bodies.
    The global catalog-reproduction gate checks those metadata authorities.
    """
    files = {f['path']: f for f in task['files']}
    def bound(path, role):
        entry = files.get(path, {})
        return (role in entry.get('roles', []) and entry.get('review_binding_matches') is True
                and entry.get('review_expected_sha256') == entry.get('observed_sha256')
                and bool(entry.get('review_authority')))
    cache = {}
    for claim in task['declarations']:
        value = (claim.get('role') == 'parametric_value_contract'
                 and claim.get('model_kind') in {'backend_value_contract', 'bit_interpreter_contract'}
                 and claim.get('scope_relation') == 'full_declared_stage' and bool(claim.get('covers_stages')))
        protocol = (claim.get('role') == claim.get('model_kind') == 'protocol_contract'
                    and claim.get('scope_relation') == 'not_an_execution_contract'
                    and claim.get('covers_stages') == [])
        path = claim['file']
        if (claim.get('kind') != 'specification' or not (value or protocol)
                or not claim.get('review_basis') or not bound(path, 'public_contract')):
            continue
        if path not in cache:
            cache[path] = declarations(root / path)
        actual = cache[path][claim['declaration']]
        if actual['kind'] != 'specification' or actual['statement_sha256'] != claim['statement_sha256']:
            continue
        result = declaration_result(actual['text'])
        if weakened_result(result):
            continue
        for view in task['source_views']:
            if (view['variant'] != claim.get('source_variant') or not view.get('authority')
                    or not view.get('translation_trust_boundary')
                    or not all(bound(p, 'selected_source') for p in view['selected_definitions'])):
                continue
            family, alias = None, None
            if view['frontend'] == 'shared_triton_dsl' and value:
                if re.match(rf'∃\s+({IDENT})\s*,\s*FP\.Structural\.exec(?![\w.\'])[^∧∨→]*?'
                            rf'=\s*some\s+\1\s*∧\s*\1\.mem\s*=\s*[^∧∨→]+(?:∧|$)', result):
                    family = 'typed_execution_memory'
            elif view['frontend'] == 'task_local_exact_ast':
                selected = source_token(n for names in view['selected_definitions'].values() for n in names)
                prefix, conclusion = result_after_lets(result)
                if value:
                    for compiler, reference in [('compileForward', 'forwardMemoryReference'),
                                                ('compileFinal', 'finalMemoryReference')]:
                        if re.fullmatch(rf'{compiler}\s+{selected}\s+[^=∧∨→]+='
                                        rf'\s*{reference}\s+[^=∧∨→]+', conclusion):
                            family = 'sc_compiled_memory'
                    execution = re.match(rf'∃\s+({IDENT})\s*,\s*(?:compileAndRun|compileBackward)'
                        rf'\s+{selected}\s+[^=∧∨→]+=\s*some\s*\(\s*({IDENT})\s*,\s*\1\s*\)\s*∧', conclusion)
                    if execution and re.search(rf'let\s+{re.escape(execution[2])}\s*:=\s*controlTrace\b', prefix):
                        final = re.escape(execution[1])
                        observation = conclusion[execution.end():].strip()
                        if (re.match(rf'{final}\.mem\s*=\s*\(replay\b[^∧∨→]+\)\.mem\s*∧', observation)
                                or re.match(rf'Related\b[^∧∨→]*\b{final}\.values\s*\(replay\b[^∧∨→]+\)\s*∧', observation)):
                            family = 'sc_compiled_trace_value'
                    alias = sc_value_alias(result, cache[path], selected)
                    if alias:
                        family = 'sc_trace_value_definition'
                elif protocol and re.fullmatch(
                        rf'∃\s+({IDENT})\s+({IDENT})\s*,\s*compileBackward\s+{selected}'
                        rf'\s+[^∧∨→]+?=\s*some\s*\(\s*\1\s*,\s*\2\s*\)\s*∧\s*'
                        rf'ProtocolInvariant\s+[^∧∨→]+\b\2\.control', conclusion):
                    family = 'sc_source_protocol'
            if family:
                return {'task': task['task'], 'file': path, 'declaration': claim['declaration'],
                        'statement_sha256': claim['statement_sha256'], 'role': claim['role'],
                        'source_variant': claim['source_variant'], 'source_view': view['id'],
                        'surface': family, 'referenced_definition': alias}
    return None


def structural_checks(root, catalog):
    """Apply whole-port structural rules to every explicitly inventoried file.

    The lexical named-surface rule retains its original conservative scope;
    actual theorem types and axioms are checked by the unchanged Lean worker.
    Additional typed/SC surfaces require reviewed_named_surface's explicit
    public-result checks; no newly named predicate is automatically exempted.
    """
    validate_catalog(root, catalog)
    errors = []
    if catalog['unregistered_task_modules']:
        errors.append('unregistered task modules: ' + repr(catalog['unregistered_task_modules']))
    if catalog.get('missing_bound_python_sources'):
        errors.append('missing bound Python sources: ' + repr(catalog['missing_bound_python_sources']))
    blockers = (root / 'bench/tritonbench_g/proof_blockers.md').read_text()
    completion = (root / 'bench/tritonbench_g/completion_audit.md').read_text()
    section = re.search(r'^## Correctness-Surface Blockers$(.*?)(?=^## |\Z)', blockers, re.M | re.S)
    registered = set(re.findall(r'^\|\s*`([A-Za-z0-9_]+)`\s*\|', section[1] if section else '', re.M))
    surface = re.compile(r'ComputeCorrect\.Realizes|ComputeRefine\.Realizes|ComputeCorrect\.General|correct_target|⊨')
    correct_marker = re.compile(r'^\s*(?:--\s*)?Correctness-surface blocker:(.*)$', re.M)
    translation_marker = re.compile(r'^\s*(?:--\s*)?Translation-surface blocker:(.*)$', re.M)
    placeholders = (re.compile(r'\bsorry\b'), re.compile(r'(?<!\.)\badmit\b'),
                    re.compile(r':=\s*(?:by\s+)?trivial\s*$'))
    whole_trivial = re.compile(r':=\s*by[ \t]*\n\s*trivial[ \t]*\n(?![ \t])')
    exempt, translated, checked_files = set(), set(), 0
    reviewed_surfaces = []
    for task in catalog['tasks']:
        name = task['task']
        has_surface, markers = False, []
        for entry in task['files']:
            path = entry['path']
            if entry.get('review_expected_sha256') != entry['observed_sha256']:
                errors.append('unreviewed or changed source: ' + path)
            text = checked(root, path, entry['observed_sha256']).read_text()
            checked_files += 1
            code = strip_lean_comments(text)
            for line_number, line in enumerate(code.splitlines(), 1):
                if bare_true_proof(line) or any(p.search(line) for p in placeholders):
                    errors.append(f'placeholder proof: {path}:{line_number}')
            if whole_trivial.search(code):
                errors.append('whole-proof trivial: ' + path)
            if re.search(r'\(hAlg\s*:', text):
                errors.append('unexpected algorithm-layer hAlg blocker: ' + path)
            has_surface |= bool(surface.search(lean_code_mask(text)))
            markers.extend(correct_marker.findall(lean_first_preamble(text)))
        witness = reviewed_named_surface(root, task) if not has_surface else None
        if witness:
            reviewed_surfaces.append(witness)
        if has_surface:
            if markers:
                errors.append('stale correctness-surface marker: ' + name)
        elif markers or name in registered or not witness:
            # These historical markers concern the selected shared-DSL/math
            # surface. A supplementary typed/SC contract must not silently
            # discharge that distinct blocker or make its registry row stale.
            exempt.add(name)
            if not markers or not any(m.strip() for m in markers) or name not in registered:
                errors.append('missing named correctness surface/registered blocker: ' + name)
        # Keep historical selected-view blocker rules intact. New typed/task-SC
        # views are checked by exact source bindings and the mandatory scope
        # review; they are never mislabeled as ordinary shared-DSL source views.
        historical = task['historical_source_entry']
        if historical:
            text = (root / historical).read_text()
            preamble = text[:text.find('triton {')] if 'triton {' in text else text
            translations = translation_marker.findall(preamble)
            if translations:
                if any(not m.strip() for m in translations):
                    errors.append('empty translation-surface blocker: ' + name)
                if f'`{name}`' not in blockers:
                    errors.append('undocumented translation-surface blocker: ' + name)
                if any(m.strip() for m in translations):
                    translated.add(name)
        readme = root / 'bench/tritonbench_g' / name / 'README.md'
        if readme.is_file() and 'Status: TODO' in readme.read_text():
            if any(len(Path(f['path']).parts) == 4 for f in task['files']):
                errors.append('compiled port with TODO README: ' + name)
    errors.extend('stale correctness-surface registry row: ' + name for name in sorted(registered - exempt))
    listed, in_block, seen_item = [], False, False
    for line in completion.splitlines():
        if line.strip() == 'The current documented blocker set is:':
            in_block = True
            continue
        if not in_block or (not line.strip() and not seen_item):
            continue
        if not line.strip():
            break
        match = re.match(r'- `([^`]+)`', line.strip())
        if match:
            listed.append(match[1]); seen_item = True
    if set(listed) != translated or len(listed) != len(set(listed)):
        errors.append('completion audit translation blocker set differs: ' + repr({
            'missing': sorted(translated - set(listed)), 'stale': sorted(set(listed) - translated),
            'duplicates': sorted(n for n in set(listed) if listed.count(n) > 1)}))
    return {'status': 'failed' if errors else 'passed', 'checked_task_files': checked_files,
            'task_count': len(catalog['tasks']), 'errors': sorted(set(errors)),
            'reviewed_named_surfaces': reviewed_surfaces,
            'boundary': 'Structural source checks only; no compilation, axiom, comparator or scope acceptance inferred.'}


def command_gate(root, output, name, command):
    """Run a fixed gate, retaining actual nonzero status and its complete log."""
    log = output / (name + '.log')
    with log.open('w') as stream:
        try:
            code = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT).returncode
        except OSError as error:
            stream.write(str(error) + '\n')
            code = 127
    return {'status': 'passed' if code == 0 else 'failed', 'exit_code': code,
            'command': command, 'log': log.name, 'log_sha256': hash_file(log)}


def run(args):
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('global audit output must be a fresh directory')
    output.mkdir(parents=True)
    catalog_path = args.catalog.resolve()
    catalog = load(catalog_path)
    catalog_hash = hash_file(catalog_path)
    gates = {}
    missing_tools = [name for name in ('python3', 'rg', 'bash') if shutil.which(name) is None]
    gates['source_tools'] = {'status': 'failed' if missing_tools else 'passed', 'missing': missing_tools}
    try:
        gates['structure'] = structural_checks(root, catalog)
    except (OSError, ValueError, KeyError, TypeError) as error:
        gates['structure'] = {'status': 'failed', 'errors': [str(error)]}
    # Required artifact paths are resolved before changing the subprocess cwd.
    package = args.baseline_package.resolve()
    baseline = output / 'baseline'
    gates['baseline'] = command_gate(root, output, 'baseline', [sys.executable, '-B',
        'bench/current_source_catalog.py', 'prepare-baseline', '--package', str(package), '--output', str(baseline)])
    regenerate = [sys.executable, '-B', 'bench/current_source_catalog.py', '--root', str(root),
        '--baseline', str(baseline), '--source-map', str(args.source_map.resolve()),
        '--legacy-inventory', str(args.legacy_inventory.resolve()),
        '--evidence-root', str(args.evidence_root.resolve()),
        '--review-manifest', str(args.review_manifest.resolve()), '--check', str(catalog_path)]
    if args.supplement:
        regenerate += ['--supplement', str(args.supplement.resolve())]
    gates['catalog_reproduction'] = command_gate(root, output, 'catalog_reproduction', regenerate)
    gates['scope_review'] = command_gate(root, output, 'scope_review', [sys.executable, '-B',
        'bench/check_original_program_scope.py', '--root', str(root), '--catalog', str(catalog_path),
        '--ledger', str(args.scope_ledger.resolve())])
    gates['legacy_source'] = command_gate(root, output, 'legacy_source', [sys.executable, '-B',
        'bench/legacy_source_adapter.py', '--root', str(root), '--tools-root', str(root),
        '--catalog', str(catalog_path), '--output', str(output / 'legacy-source')])
    value_command = [sys.executable, '-B', 'bench/current_value_metadata.py', '--root', str(root),
        '--baseline', str(baseline), '--catalog', str(catalog_path),
        '--source-map', str(args.source_map.resolve()), '--legacy-inventory', str(args.legacy_inventory.resolve()),
        '--evidence-root', str(args.evidence_root.resolve()), '--review-manifest', str(args.review_manifest.resolve()),
        '--output', str(output / 'value-metadata.json')]
    if args.supplement:
        value_command += ['--supplement', str(args.supplement.resolve())]
    gates['value_metadata'] = command_gate(root, output, 'value_metadata', value_command)
    gates['deferred_numerics'] = command_gate(root, output, 'deferred_numerics',
        [sys.executable, '-B', 'bench/deferred_numerical_experiments.py', '--check'])
    for name, flag in [('comparator_preflight', '--check-tools'), ('library_replay', '--library')]:
        gates[name] = command_gate(root, output, name, [sys.executable, '-B',
            'scripts/check_comparator.py', flag])
    try:
        validate_catalog(root, catalog)
        if hash_file(catalog_path) != catalog_hash:
            raise ValueError('catalog file changed during global audit')
        gates['unchanged_inputs'] = {'status': 'passed'}
    except (OSError, ValueError, KeyError, TypeError) as error:
        gates['unchanged_inputs'] = {'status': 'failed', 'errors': [str(error)]}
    passed = all(g['status'] == 'passed' for g in gates.values())
    result = {'schema': 1, 'kind': 'current_source_global_gates',
        'catalog_sha256': sha(canonical(catalog)), 'status': 'passed' if passed else 'failed',
        'gates': gates, 'corpus_replay': 'not_run_by_global_runner',
        'integration_build': 'not_run_by_global_runner',
        'native_functional_correctness': 'not_established',
        'whole_task_acceptance': 'not_established_by_global_gates_alone'}
    write_json(output / 'receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'catalog', 'baseline-package', 'source-map', 'legacy-inventory',
                 'evidence-root', 'review-manifest', 'scope-ledger', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    args = parser.parse_args()
    result = run(args)
    print(__import__('json').dumps({'status': result['status'],
        'gates': {name: gate['status'] for name, gate in result['gates'].items()},
        'boundary': 'Global gates only; corpus replay and integration build are separate.'}, indent=2))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    sys.exit(main())
