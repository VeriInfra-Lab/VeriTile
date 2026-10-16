#!/usr/bin/env python3
"""Inventory current TritonBench-G proof obligations without claiming proof replay.

Run from any directory; --check checks the generated report and all captured inputs.
This deliberately separates source/declaration discovery, reviewed scope, historical
acceptance, and proof acceptance. Neither a metadata label nor an absent placeholder
establishes completion. It does not execute Python kernels or elaborate Lean.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PORTS = ROOT / 'bench/tritonbench_g'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench'))
from port_layout import source_file, source_import_files, source_text, SUPPLEMENTAL_SPEC_NAMES
from check_proof_gap_manifest import headline_names
from audit_source import is_triton_jit, source_audit_metadata, lean_code_mask

REPORT = PORTS / 'current_completion_inventory.json'
MARKDOWN = PORTS / 'current_completion_inventory.md'
LEVELS = ('full_value_candidate', 'specialization', 'precomputed_input_slice',
          'projection_only', 'pre_rounding_slice', 'blocked_summary', 'unreviewed')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def table_section(text, heading):
    section = text.split(heading + '\n', 1)[1]
    return re.split(r'\n#{1,3} ', section, maxsplit=1)[0]


def input_files():
    files = set(PORTS.glob('*/*.py')) | set(PORTS.glob('*/**/*.lean'))
    files |= set(PORTS.glob('*/README.md')) | set(PORTS.glob('*/source_scope.md'))
    files |= {PORTS / n for n in ('coverage_review.json', 'proof_gap_manifest.tsv',
        'verification_round.md', 'review_criteria.md', 'completion_audit.md',
        'proof_blockers.md', 'README.md')}
    files |= {ROOT / p for p in ('bench/port_layout.py', 'bench/audit_source.py',
        'bench/check_proof_gap_manifest.py', 'bench/coverage_review.py',
        'bench/inventory_tritonbench_g_completion.py', 'scripts/kernel-manifest.tsv',
        'VeriTile/Triton/DSL/RandomProfile.lean')}
    files |= set((ROOT / 'bench').glob('source_*.py'))
    files |= set(PORTS.glob('source_*profiles.json'))
    files |= {PORTS / n for n in ('numerical_obligations_review.json',
                                 'numerical_obligations.json', 'numerical_obligations.md',
                                 'numeric_experiment_tasks.json', 'numeric_experiment_handoff.md',
                                 'fp_proof_gaps.json', 'fp_proof_gaps.md')}
    files.add(ROOT / 'bench/numerical_obligations.py')
    return files


def collect():
    files = input_files()
    data = {rel(p): p.read_bytes() for p in sorted(files)}
    read = lambda p: data[rel(p)].decode()
    fingerprint = {p: sha(b) for p, b in data.items()}
    history = read(PORTS / 'verification_round.md')
    accepted = set(re.findall(r'^\| `([^`]+)` \|',
        table_section(history, '## Historical completed cases'), re.M))
    progress = dict(re.findall(r'^\| `([^`]+)` \| (.*?) \|$',
        table_section(history, '## Per-directory progress'), re.M))
    manifest = list(csv.DictReader(io.StringIO(read(PORTS / 'proof_gap_manifest.tsv')),
                                   delimiter='\t'))
    manifest_by_file = defaultdict(list)
    for row in manifest:
        manifest_by_file[row['file']].append(row)
    reviews = json.loads(read(PORTS / 'coverage_review.json'))['ports']
    review_by_file = {r['file']: r for r in reviews}
    registrations = defaultdict(list)
    fields = 'id file theorem kind status source source_ref config label notes'.split()
    for row in csv.DictReader((l for l in read(ROOT / 'scripts/kernel-manifest.tsv').splitlines()
                              if l and not l.startswith('#')), fieldnames=fields, delimiter='\t'):
        if row['file'].startswith('bench/tritonbench_g/'):
            registrations[Path(row['file']).parts[2]].append(row)
    tasks = []
    for directory in sorted(PORTS.iterdir()):
        if not directory.is_dir() or directory.name.startswith('_'):
            continue
        name = directory.name
        pyfiles = sorted(directory.glob('*.py'))
        leanfiles = sorted(directory.rglob('*.lean'))
        lean_declarations = {}
        decl_re = re.compile(r'^\s*(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*'
            r'(def|abbrev|theorem|lemma|specification|axiom|constant|opaque|structure|inductive|denotation)'
            r'\s+([\w\'\u00ab\u00bb.]+)', re.M)
        for path in leanfiles:
            code = lean_code_mask(read(path))
            named = defaultdict(list)
            for match in decl_re.finditer(code):
                named[match[1]].append(match[2])
            lean_declarations[rel(path)] = dict(named)
        errors = []
        try:
            entry = source_file(directory)
        except ValueError as e:
            entry = None
            errors.append(str(e))
        public_files = ([directory / 'Correct.lean', directory / 'FPEquiv.lean']
                        if entry and entry.name == 'Kernels.lean' else [entry] if entry else [])
        if entry:
            public_files.extend(directory / name for name in SUPPLEMENTAL_SPEC_NAMES
                                if (directory / name).is_file())
        declarations, file_reviews = [], []
        for public in public_files:
            actual = headline_names(read(public))
            recorded = {r['declaration'] for r in manifest_by_file[rel(public)]}
            if actual != recorded:
                errors.append(f'Public manifest mismatch {rel(public)}: '
                              f'missing {sorted(actual-recorded)}, extra {sorted(recorded-actual)}')
            review = review_by_file.get(rel(public))
            issues = []
            reviewed = {}
            if not review:
                if actual:
                    issues.append('missing_review')
            else:
                if fingerprint[rel(public)] != review['lean_sha256']:
                    issues.append('stale_public_lean_hash')
                if fingerprint.get(review['python_file']) != review['python_sha256']:
                    issues.append('stale_python_hash')
                if entry.name == 'Kernels.lean' or (directory / 'FPCorrect.lean').is_file():
                    support = {rel(p): fingerprint[rel(p)] for p in leanfiles if p != public}
                    if support != review.get('support_sha256'):
                        issues.append('stale_or_incomplete_support_hashes')
                reviewed = {d['declaration']: d for d in review['declarations']}
                if set(reviewed) != actual:
                    issues.append('reviewed_declaration_set_mismatch')
                pyast = ast.parse(read(ROOT / review['python_file']))
                functions = {(n.name, n.lineno) for n in pyast.body
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                for row in manifest_by_file[rel(public)]:
                    d = reviewed.get(row['declaration'], {})
                    if row['coverage_level'] != d.get('coverage_level'):
                        issues.append('manifest_review_level_mismatch:' + row['declaration'])
                    if not d.get('python_functions') or any(
                        (f['name'], f['line']) not in functions for f in d.get('python_functions', [])):
                        issues.append('unresolved_python_link:' + row['declaration'])
            # Count actual source declarations even while a manifest is stale.
            # Removed manifest rows remain diagnostics, never current headlines.
            manifest_rows = {r['declaration']: r for r in manifest_by_file[rel(public)]}
            for declaration_name in sorted(actual):
                row = manifest_rows.get(declaration_name, {})
                d = reviewed.get(declaration_name, {})
                declarations.append({**row, 'file': rel(public), 'declaration': declaration_name,
                    'coverage_level': d.get('coverage_level', 'unreviewed'),
                    'manifest_coverage_level': row.get('coverage_level'),
                    'numeric_model': d.get('numeric_model'),
                    'evidence': d.get('scope', 'Unreviewed source declaration.'),
                    'python_functions': d.get('python_functions', []),
                    'actual_declaration_present': True})
            file_reviews.append({'file': rel(public), 'issues': sorted(set(issues))})
        jits = []
        for pyfile in pyfiles:
            tree = ast.parse(read(pyfile))
            functions = [n for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            jitnodes = [n for n in functions if any(is_triton_jit(d) for d in n.decorator_list)]
            for node in sorted(jitnodes, key=lambda n: n.lineno):
                calls = [ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)]
                callers = [n.name for n in jitnodes if any(isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name) and c.func.id == node.name for c in ast.walk(n))]
                direct = [d['file'] + '::' + d['declaration'] for d in declarations
                          if any(f['name'] == node.name and f['line'] == node.lineno
                                 for f in d['python_functions'])]
                writes = sorted(set(c for c in calls if c == 'tl.store' or c.startswith('tl.atomic_')))
                jits.append({'file': rel(pyfile), 'name': node.name, 'line': node.lineno,
                    'signature': ast.unparse(node.args), 'write_operations': writes,
                    'jit_callers': callers, 'direct_public_links': direct,
                    'link_scope': 'direct_review_links' if direct else
                        'unlinked_memory_writer' if writes else 'unlinked_helper'})
        audit = {'selected_view_extracted': False, 'full_all_jit_comparison_performed': False}
        if entry:
            try:
                selected = source_text(entry)
                audit.update(selected_view_extracted=True, selected_view_sha256=sha(selected.encode()),
                    reachable_source_files=[rel(p) for p in source_import_files(entry)],
                    selectors={k: source_audit_metadata(read(entry), k)
                               for k in ('target', 'Lean', 'args', 'params')})
            except ValueError as e:
                audit['extraction_error'] = str(e)
        counts = Counter(d['coverage_level'] for d in declarations)
        gaps = []
        if not pyfiles or not entry:
            gaps.append('missing_local_source_pair')
        if not public_files:
            gaps.append('missing_public_proof_entries')
        if public_files and not declarations:
            gaps.append('missing_public_proof_declarations')
        if any(j['link_scope'] == 'unlinked_memory_writer' for j in jits):
            gaps.append('jit_memory_writer_without_direct_public_link')
        gaps.extend('public_' + level for level in LEVELS[1:] if counts[level])
        if any(r['issues'] for r in file_reviews) or errors:
            gaps.append('metadata_or_layout_validation_failure')
        if entry and not audit['selected_view_extracted']:
            gaps.append('source_view_extraction_failure')
        gaps.append('current_independent_replay_not_verified_by_inventory')
        if not pyfiles or not entry:
            category = 'missing_source'
        elif not declarations:
            category = 'incomplete_public_proof_entries'
        elif counts['unreviewed']:
            category = 'unreviewed_public_statement'
        elif counts['blocked_summary']:
            category = 'blocked_public_statement'
        elif any(counts[c] for c in ('precomputed_input_slice', 'pre_rounding_slice', 'projection_only')):
            category = 'partial_public_statement'
        elif counts['specialization']:
            category = 'specialized_public_statement'
        else:
            category = 'full_value_labels_require_acceptance'
        scope = directory / 'source_scope.md'
        tasks.append({'task': name, 'layout': 'split' if entry and entry.name == 'Kernels.lean'
                      else 'legacy' if entry else 'scaffold',
            'source_entry': rel(entry) if entry else None,
            'python_files': [rel(p) for p in pyfiles], 'python_jits': jits,
            'lean_files': [rel(p) for p in leanfiles],
            'named_lean_declarations': lean_declarations,
            'public_files': [rel(p) for p in public_files],
            'public_declarations': declarations, 'review_validation': file_reviews,
            'source_audit': audit, 'source_scope_file': rel(scope) if scope.exists() else None,
            'source_scope_text': read(scope) if scope.exists() else None,
            'historically_accepted': name in accepted,
            'historical_progress_text': progress.get(name),
            'historical_progress_conflicts_with_source_presence':
                progress.get(name) == 'pending source import' and bool(pyfiles and entry),
            'declaration_level_counts': dict(sorted(counts.items())),
            'kernel_manifest_entries': registrations[name],
            'scope_category': category, 'remaining_obligation_signals': gaps,
            'layout_or_manifest_errors': errors,
            'current_complete_acceptance': 'not_established_by_this_inventory'})
    # A specialization/partial label is a scope description. It does not say
    # whether a proof needs numerical evidence. The user accepts documented
    # input/configuration/stage scope; review actual numerical premises apart.
    from numerical_obligations import load_review, task_status
    numerical_review, numerical_rows, numerical_rules = load_review(
        [d for task in tasks for d in task['public_declarations']])
    for task in tasks:
        task['accepted_scope_review'] = task_status(task, numerical_rows)
        for declaration in task['public_declarations']:
            row = numerical_rows[declaration['file'], declaration['declaration']]
            declaration['numerical_obligations'] = {
                key: row[key] for key in ('numerical_status', 'extra_numeric_ids', 'shared_model_ids', 'fp_model_review')}
    receipt_paths = sorted(set(re.findall(r'(?:/tmp/|Logs/)[^\s`<>\)\],;]+', history)))
    receipts = [{'path': p, 'exists': (Path(p) if p.startswith('/') else ROOT / p).exists(),
                 'contents_validated': False} for p in receipt_paths]
    tasknames = {t['task'] for t in tasks}
    if len(tasknames) != len(tasks) or tasknames != {
            directory.name for directory in PORTS.iterdir()
            if directory.is_dir() and not directory.name.startswith('_')}:
        raise RuntimeError('Task identity changed during declaration discovery')
    counter = lambda key: dict(sorted(Counter(t[key] for t in tasks).items()))
    summary = {
        'task_directories': len(tasks),
        'source_pairs': sum(bool(t['python_files'] and t['source_entry']) for t in tasks),
        'layout_counts': counter('layout'), 'scope_category_counts': counter('scope_category'),
        'python_jits': sum(len(t['python_jits']) for t in tasks),
        'jit_link_counts': dict(sorted(Counter(j['link_scope'] for t in tasks
                                               for j in t['python_jits']).items())),
        'public_entry_files': sum(len(t['public_files']) for t in tasks),
        'public_files': sum(len({d['file'] for d in t['public_declarations']}) for t in tasks),
        'public_declarations': sum(len(t['public_declarations']) for t in tasks),
        'lexically_discovered_named_lean_declarations': dict(sorted(Counter({
            kind: sum(len(names) for t in tasks for kinds in t['named_lean_declarations'].values()
                      for key, names in kinds.items() if key == kind)
            for kind in {k for t in tasks for kinds in t['named_lean_declarations'].values()
                         for k in kinds}}).items())),
        'public_declaration_levels': dict(sorted(Counter(d['coverage_level'] for t in tasks
                                                         for d in t['public_declarations']).items())),
        'numeric_models': dict(sorted(Counter(d['numeric_model'] for t in tasks
                                              for d in t['public_declarations']).items())),
        'selected_source_views_extracted': sum(t['source_audit']['selected_view_extracted'] for t in tasks),
        'historical_acceptance_rows': len(accepted),
        'outside_historical_accepted_set': len(tasknames - accepted),
        'imported_outside_historical_accepted_set': sum(bool(t['source_entry']) and
            not t['historically_accepted'] for t in tasks),
        'progress_table_rows': len(progress),
        'progress_table_missing_tasks': sorted(tasknames - set(progress)),
        'progress_table_source_import_conflicts': [t['task'] for t in tasks
            if t['historical_progress_conflicts_with_source_presence']],
        'tasks_with_metadata_errors': [t['task'] for t in tasks if t['layout_or_manifest_errors']
            or any(r['issues'] for r in t['review_validation'])],
        'historical_local_evidence_paths': len(receipts),
        'historical_local_evidence_paths_existing': sum(r['exists'] for r in receipts),
        'current_acceptance_verified_by_this_inventory': 0,
        'accepted_scope_policy': numerical_review['policy'],
        'task_fp_adequacy_status': dict(sorted(Counter(t['accepted_scope_review']['fp_adequacy_status'] for t in tasks).items())),
        'task_numerical_status': dict(sorted(Counter(t['accepted_scope_review']['numerical_status'] for t in tasks).items())),
        'declaration_numerical_status': dict(sorted(Counter(r['numerical_status'] for r in numerical_rows.values()).items())),
        'extra_numeric_obligations': sum(r['kind'] != 'shared_model_requirement' for r in numerical_rules.values()),
        'shared_model_requirements': sum(r['kind'] == 'shared_model_requirement' for r in numerical_rules.values()),
        'shared_model_tasks': sum(bool(t['accepted_scope_review']['shared_model_ids']) for t in tasks),
        'source_translation_blocked_tasks': [t['task'] for t in tasks if t['accepted_scope_review']['source_translation_blocked']],
    }
    priorities = [
        {'tasks': [t['task'] for t in tasks if t['accepted_scope_review']['fp_adequacy_status'] == 'math_contract_only'],
         'work': 'Supply actual floating semantics and an independent rounded value contract or meaningful source-bound candidate equivalence. Real arithmetic, typed names and self-equivalence do not close this gap; see fp_proof_gaps.md.'},
        {'tasks': [t['task'] for t in tasks if t['accepted_scope_review']['fp_adequacy_status'] == 'rounding_contract_only'],
         'work': 'The current model rounds cast/store sites but retains real arithmetic. Add arithmetic-precision/rounding semantics and the corresponding value or candidate-equivalence proof; a shared RoundingModel instance alone is insufficient.'},
        {'tasks': [t['task'] for t in tasks if t['accepted_scope_review']['numerical_status'] == 'awaiting_numeric_validation'],
         'work': 'Supply the exact typed-fp32 multiplication binding and four missing BGMV native addition profiles listed in numeric_experiment_tasks.json. Preserve actual site domains and report provenance; existing scalar reports are not counted as missing.'},
        {'tasks': [t['task'] for t in tasks if t['accepted_scope_review']['numerical_status'] == 'requires_numeric_model_revision'],
         'work': 'The unrestricted fp16 rounding identity has a counterexample. Revise the rounding model/reference or justify a representable intermediate-value domain; experiments cannot validate the unrestricted identity.'},
        {'tasks': [t['task'] for t in tasks if t['scope_category'] == 'missing_source'],
         'work': 'Import pinned source pairs and establish their public contracts; no source-level completion claim exists for these six tasks.'},
        {'tasks': ['quant_transpose_kernel'],
         'work': 'The documented scaled stage remains accepted in scope; complete-source translation is independently blocked. Do not infer a whole-source theorem from the accepted stage.'},
        {'tasks': [],
         'work': 'Instantiate shared rounding contracts and native/host correspondence when claiming hardware behavior, and run official independent replay on a supported host. These boundaries are separate from extra numerical-law counts. Additional configurations or omitted stages are optional scope expansion under the accepted policy.'},
    ]
    changed = [p for p, digest in fingerprint.items()
               if not (ROOT / p).is_file() or sha((ROOT / p).read_bytes()) != digest]
    if changed:
        raise RuntimeError('Inputs changed while reading; rerun: ' + ', '.join(changed))
    if files != input_files():
        raise RuntimeError('Input file set changed while reading; rerun the inventory.')
    result = {'schema': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Filesystem entries, Python AST JIT/call/write discovery, comment-stripped public Lean declaration discovery, review/source/support hashes, official selected-source extraction, and historical-ledger enumeration. No Lean elaboration, comparator replay or all-profile source certification is performed.',
        'interpretation': 'The user accepts normal input requirements and documented configuration/stage scopes, but not a replacement of floating arithmetic by real arithmetic. FP model adequacy is primary; an empty extra-law list is not completion, and relative equivalence is not absolute native functional correctness. Legacy scope_category labels describe coverage and do not count missing numerical assumptions. accepted_scope_review and numerical_obligations track actual numerical laws, existing evidence, pending bindings and idealizations independently. Accepted stage scope does not prove omitted source stages, GPU refinement or independent replay. Historical acceptance and direct JIT links do not establish current completion.',
        'summary': summary, 'priorities': priorities,
        'historical_local_evidence_paths': receipts, 'tasks': tasks,
        'input_fingerprints_sha256': fingerprint}
    return result


def render(report):
    from numerical_obligations import STATUS_LABELS, STATUS_ORDER, FP_ADEQUACY_LABELS
    s = report['summary']
    out = ['# TritonBench-G current completion inventory', '',
        'Generated by `python3 bench/inventory_tritonbench_g_completion.py`; verify freshness with the same command plus `--check`.', '',
        '**This is an obligation inventory, not a new proof-acceptance certificate.**', '',
        f"There are **{s['task_directories']} tasks**, **{s['source_pairs']} local Python/Lean pairs**, "
        f"and **{s['layout_counts'].get('scaffold', 0)} missing source pairs**. "
        f"The imported set contains **{s['python_jits']} JIT functions**. "
        f"The historical acceptance table has **{s['historical_acceptance_rows']} tasks**, leaving "
        f"**{s['outside_historical_accepted_set']} outside that ledger** "
        f"({s['imported_outside_historical_accepted_set']} imported and {s['layout_counts'].get('scaffold', 0)} scaffolds). "
        'These historical counts do not certify the current tree.', '',
        f"The review inventory contains **{s['public_declarations']} public entry records in {s['public_files']} files**. "
        'Records are keyed by file and declaration spelling; identical spellings in different '
        'namespaces can share a record. Lean environment checks distinguish those declarations. '
        f"Official source discovery extracts {s['selected_source_views_extracted']} selected views; "
        'that checks extraction, not all JIT bodies, profiles, source correspondence or theorem validity. '
        f"Historical local evidence references: {s['historical_local_evidence_paths_existing']} of "
        f"{s['historical_local_evidence_paths']} paths exist. No current replay receipt is validated by this inventory.", '',
        '## Current statistics under the accepted scope policy', '',
        'Normal input requirements and explicitly documented configuration/stage scopes are accepted. '
        'The primary table states the actual public proof model and claim; mathematical and cast/store-only results remain gaps for floating arithmetic. '
        'Existing reports remain scoped experimental evidence. '
        'See [the exact numerical assumptions and experiment tasks](numerical_obligations.md).', '',
        '| FP proof model / claim | Tasks |', '|---|---:|']
    for status, label in FP_ADEQUACY_LABELS.items():
        count = s['task_fp_adequacy_status'].get(status, 0)
        if count:
            out.append(f'| {label} | {count} |')
    out += ['', 'Accepted input/stage scope does not accept replacing floating arithmetic by real arithmetic. '
            'A relative candidate-vs-original result does not certify absolute native function accuracy. '
            'See [the concrete FP gaps](fp_proof_gaps.md). The former 129 count mixed model gaps with atom-free proofs and is withdrawn as a completion interpretation.', '',
            '### Subordinate count of extra numerical premises (not completion)', '',
            '| Numerical evidence status | Tasks | Public records |', '|---|---:|---:|']
    for status in (*STATUS_ORDER, 'missing_source'):
        out.append(f"| {STATUS_LABELS[status]} | {s['task_numerical_status'].get(status, 0)} | {s['declaration_numerical_status'].get(status, 0)} |")
    out += ['', f"Shared rounding-model requirements affect {s['shared_model_tasks']} tasks and are recorded separately. "
            'Mathematical/native correspondence and independent replay remain separate assurance boundaries.', '',
            '## Historical coverage tags (not numerical-experiment deficits)', '',
            '| Public statement scope | Entry records |', '|---|---:|']
    out += [f'| {level} | {s["public_declaration_levels"].get(level, 0)} |' for level in LEVELS]
    out += ['', 'A `full_value_candidate` label does not establish completion. A specialization can '
        'prove a useful conditional claim while leaving other source profiles or branches open. '
        'Prepared-input and pre-rounding slices do not prove their omitted producers or final conversions. '
        'Unreviewed source declarations remain in the total even when the manifest is stale. '
        'Task categories aggregate declaration tags: a retained specialized declaration can '
        'classify a task even when another declaration covers more cases. They are not '
        'completed-task counts; redundant entry cleanup does not prove additional behavior. '
        'The JSON includes recorded public scopes, Python function links and source fingerprints.', '',
        '## Source and ledger discrepancies', '',
        '- Missing per-directory progress rows: ' + ', '.join('`' + n + '`' for n in s['progress_table_missing_tasks']) + '.',
        '- Progress says source import is pending despite an existing pair: ' + ', '.join('`' + n + '`' for n in s['progress_table_source_import_conflicts']) + '.',
        '- Metadata/hash validation failures: ' + (', '.join(s['tasks_with_metadata_errors']) or 'none in captured inputs') + '.', '',
        'The following JITs have no direct public declaration link. Memory-writing helpers may be '
        'inlined into a caller; linkage alone does not establish absence or presence of a complete proof.', '',
        '| Task | JIT | Writes | JIT callers |', '|---|---|---|---|']
    for task in report['tasks']:
        for j in task['python_jits']:
            if not j['direct_public_links']:
                out.append(f"| `{task['task']}` | `{j['name']}` | {', '.join(j['write_operations']) or 'none'} | {', '.join(j['jit_callers']) or 'none'} |")
    out += ['', '## Next independent work', '']
    out += ['- ' + (', '.join('`' + n + '`' for n in p['tasks']) or 'Shared boundaries') + ': ' + p['work']
            for p in report['priorities']]
    out += ['', '## Every task', '',
        'Historical acceptance is shown separately from current statement scope. All rows still '
        'require current proof acceptance evidence. Counts are `F/S/I/P/R/B/U`: full-value candidate, '
        'specialization, prepared-input slice, projection only, pre-rounding slice, blocked summary, unreviewed.', '',
        '| Task | Layout | JITs | Public F/S/I/P/R/B/U | Historical accepted | Descriptive scope tag | FP proof model / claim | Numerical premises |',
        '|---|---|---:|---|---|---|---|---|']
    for t in report['tasks']:
        counts = '/'.join(str(t['declaration_level_counts'].get(level, 0)) for level in LEVELS)
        out.append(f"| `{t['task']}` | {t['layout']} | {len(t['python_jits'])} | {counts} | {'yes' if t['historically_accepted'] else 'no'} | {t['scope_category']} | {FP_ADEQUACY_LABELS[t['accepted_scope_review']['fp_adequacy_status']]} | {STATUS_LABELS[t['accepted_scope_review']['numerical_status']]} |")
    out += ['', '## Verification boundary', '', report['method'], '', report['interpretation'], '',
        'The report captures hashes of every discovered local Python/Lean file, per-task README and '
        'source-scope document, review/manifest/ledger input and discovery script. Freshness is '
        'checked before writing. These hashes bind this inventory; they are not proof replay receipts.', '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = collect()
    if args.check:
        old = json.loads(REPORT.read_text())
        report['generated_at_utc'] = old.get('generated_at_utc')
        if old != report or MARKDOWN.read_text() != render(report):
            print('Completion inventory is stale; regenerate it.', file=sys.stderr)
            return 1
    else:
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        MARKDOWN.write_text(render(report))
    print(json.dumps(report['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
