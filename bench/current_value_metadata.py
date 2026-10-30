#!/usr/bin/env python3
"""Replay frozen value checks and validate every current supplementary contract.

The old generator runs unchanged against its exact archived tree, including its
whole-public-set check. Current public membership is then checked independently
by exact catalog regeneration, with explicit reviewed stage/dtype metadata for
new declarations. Neither layer claims whole-program scope or native validity.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from adapter_common import BASE, canonical, checked, hash_file, load, sha, unique_rows, write_json
from current_source_catalog import (NONVALUES, baseline, collect, validate_catalog,
                                    validate_supplement)

DTYPE_RELATIONS = {'selected_explicit_profiles_only', 'complete_enumerated_wrapper_dtype_domain',
                   'new_source_declared_profiles', 'complete_original_declared_domain'}
SCOPE_FIELDS = ('file', 'declaration', 'statement_sha256', 'model_kind', 'role', 'scope_relation',
                'source_variant', 'covers_stages', 'dtype_profile', 'dtype_scope_relation',
                'unclosed_numeric_callbacks', 'extra_numeric_law_ids', 'native_binding_status',
                'original_scope')


def value_scope(task):
    """Exact metadata whose interpretation requires a separate current review."""
    return {'required_value_stages': task.get('required_value_stages'),
            'declarations': [{key: declaration.get(key) for key in SCOPE_FIELDS}
                             for declaration in task['declarations']]}


def check_supplement_scope(supplement, evidence_root):
    summaries = []
    for task in supplement['tasks']:
        if not task['declarations']:
            continue
        required = task.get('required_value_stages')
        if (not isinstance(required, list) or not required or len(required) != len(set(required))
                or not all(isinstance(s, str) and s.strip() for s in required)):
            raise ValueError('supplement lacks explicit required stage set: ' + task['task'])
        review_ref = task['review']
        review = load(checked(evidence_root, review_ref['path'], review_ref['sha256']))
        if review.get('reviewed_value_scope_sha256', {}).get(task['task']) != sha(canonical(value_scope(task))):
            raise ValueError('supplement stage/dtype roles lack exact current review: ' + task['task'])
        covered, software = set(), set()
        for declaration in task['declarations']:
            label = task['task'] + '::' + declaration['declaration']
            if not isinstance(declaration.get('dtype_profile'), str) or not declaration['dtype_profile'].strip():
                raise ValueError('supplement lacks explicit typed dtype profile: ' + label)
            if declaration.get('dtype_scope_relation') not in DTYPE_RELATIONS:
                raise ValueError('unknown supplementary dtype scope relation: ' + label)
            relation = declaration.get('scope_relation')
            if relation not in {'full_declared_stage', 'strict_specialization', 'not_an_execution_contract'}:
                raise ValueError('unknown supplementary stage relation: ' + label)
            stages = declaration['covers_stages']
            if len(stages) != len(set(stages)) or not set(stages) <= set(required):
                raise ValueError('supplement covers undeclared/duplicate stages: ' + label)
            if declaration['native_binding_status'] != 'not_established':
                raise ValueError('supplement makes an unreviewed native-binding claim: ' + label)
            if declaration['model_kind'] in NONVALUES:
                if relation != 'not_an_execution_contract' or stages:
                    raise ValueError('nonvalue supplement promoted to execution: ' + label)
            elif relation == 'not_an_execution_contract':
                raise ValueError('value supplement misclassified as nonexecution: ' + label)
            elif relation == 'full_declared_stage' and not stages:
                raise ValueError('full-stage supplement covers no named stage: ' + label)
            if (declaration['model_kind'] not in NONVALUES and relation == 'full_declared_stage'
                    and declaration['source_variant'] == 'original_source_translation'):
                covered.update(stages)
                if declaration['model_kind'] == 'bit_interpreter_contract':
                    software.update(stages)
        summaries.append({'task': task['task'], 'required_value_stages': required,
            'original_full_declared_value_stages': sorted(covered),
            'original_full_declared_software_stages': sorted(software),
            'uncovered_declared_stages': sorted(set(required) - covered),
            'whole_original_program_coverage': 'requires_separate_183_scope_ledger'})
    return summaries


def check(root, prepared_baseline, catalog, source_map, legacy, evidence_root, review, supplement):
    frozen = baseline(prepared_baseline)
    # The frozen tree contains the original complete checker and all its inputs.
    # Never run it on a filtered projection of current files or skip its scan.
    process = subprocess.run([sys.executable, '-B', 'bench/fp_value_extensions.py', '--check'],
                             cwd=frozen['root'])
    if process.returncode:
        raise ValueError('frozen complete value metadata replay failed: ' + str(process.returncode))
    baseline(prepared_baseline)  # A successful command cannot conceal rewritten inputs.
    # A new live legacy registry requires a newly reviewed metadata baseline;
    # supplements may add declarations, not silently rewrite historical rows.
    for name in ('fp_value_extensions.json', 'fp_value_current_status.json',
                 'fp_value_current_status.md', 'fp_value_source_entries.json'):
        path = BASE + name
        checked(root, path, frozen['manifest'][path])
    current = collect(root, frozen, source_map, legacy, review=review,
                      evidence_root=evidence_root, supplement=supplement)
    validate_catalog(root, current)
    if current != catalog:
        raise ValueError('current value catalog differs from full regeneration')
    if supplement:
        validate_supplement(root, evidence_root, supplement, set(catalog['expected_task_names']))
        scopes = check_supplement_scope(supplement, evidence_root)
    else:
        scopes = []
    return {'schema': 1, 'kind': 'current_value_metadata_replay', 'status': 'passed',
        'frozen_complete_replay_exit_code': process.returncode,
        'frozen_metadata_receipt_sha256': frozen['receipt_sha256'],
        'catalog_sha256': sha(canonical(catalog)), 'current_public_claims': catalog['summary']['public_claims'],
        'supplement_scopes': scopes,
        'boundary': 'Exact frozen metadata and current supplementary declaration checks. Whole-original-program scope, compiled proof replay and native numerical validation remain separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'baseline', 'catalog', 'source-map', 'legacy-inventory',
                 'evidence-root', 'review-manifest'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = check(args.root.resolve(), args.baseline.resolve(), load(args.catalog),
            load(args.source_map), load(args.legacy_inventory), args.evidence_root.resolve(),
            load(args.review_manifest), load(args.supplement) if args.supplement else None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'failed', 'errors': [str(error)]}
    if args.output:
        write_json(args.output, result)
    print(__import__('json').dumps(result, indent=2))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    sys.exit(main())
