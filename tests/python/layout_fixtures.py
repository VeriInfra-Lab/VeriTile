"""Reviewed source and ABI expectations for the benchmark regression tests."""

from functools import cache
import json
from pathlib import Path

from scripts.bench.jit_correspondence import check_records

ROOT = Path(__file__).resolve().parents[2]


@cache
def pinned_cases() -> dict[str, list[dict]]:
    path = ROOT / 'tests/fixtures/jit-correspondence.json'
    return json.loads(path.read_text())


@cache
def historical_order() -> dict[str, list[int]]:
    return json.loads((ROOT / 'tests/fixtures/jit-pair-order.json').read_text())


def pinned_pairs(case: Path) -> list[dict]:
    key = (ROOT / case).relative_to(ROOT).as_posix()
    rows = pinned_cases()[key]
    return [rows[i] for i in historical_order().get(key, range(len(rows)))]


def check_pinned_pairs(root: Path, case: Path) -> list[dict]:
    key = (ROOT / case).relative_to(ROOT).as_posix()
    rows = check_records(root, pinned_cases()[key])
    return [rows[i] for i in historical_order().get(key, range(len(rows)))]


def provenance_path(case: Path, filename: str = 'candidate-provenance.json') -> Path:
    return ROOT / 'validation/tasks' / case.name / filename


@cache
def variant_fixtures() -> dict:
    return json.loads((ROOT / 'tests/fixtures/python-variants.json').read_text())


@cache
def variant_source(path: Path) -> str:
    """Select a reviewed original or candidate from the current shared module.

    The selectors retain module-level host functions as well as JIT functions.
    They reconstruct old import and naming scopes for AST comparison tests.
    """
    from collections import defaultdict
    import ast
    import io
    import symtable
    import tokenize

    recipe = variant_fixtures().get(path.relative_to(ROOT).as_posix())
    if recipe is None:
        return path.read_text()
    source = (path.parent / f'{path.parent.name}.py').read_text()
    lines = source.splitlines()
    statements = defaultdict(list)
    for node in ast.parse(source).body:
        table = symtable.symtable(ast.unparse(node), '<statement>', 'exec')
        names = sorted(symbol.get_name() for symbol in table.get_symbols()
                       if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace())
        first = min([node.lineno, *(d.lineno for d in getattr(node, 'decorator_list', ()))])
        statements[type(node).__name__, tuple(names)].append('\n'.join(lines[first - 1:node.end_lineno]))

    parts = []
    for selector in recipe['statements']:
        if isinstance(selector, str):
            parts.append(selector)
        else:
            kind, names, index = selector
            part = statements[kind, tuple(names)][index]
            # Edit identifier spans in the original text. Rebuilding every token
            # can change the whitespace before an explicit line continuation.
            part_lines = part.split('\n')
            tokens = list(tokenize.generate_tokens(io.StringIO(part + '\n').readline))
            for token in reversed(tokens):
                if token.type == tokenize.NAME and token.string in recipe['names']:
                    line, start = token.start
                    end = token.end[1]
                    original = part_lines[line - 1]
                    part_lines[line - 1] = original[:start] + recipe['names'][token.string] + original[end:]
            parts.append('\n'.join(part_lines).rstrip())
    return '\n\n'.join(parts) + '\n'


def variant_provenance(case: Path) -> dict:
    """Read historical variant labels with pins for their current test views."""
    data = json.loads(provenance_path(case).read_text())
    for label in ('source', 'original', 'candidate'):
        filename = data.get(label) or data.get(label + '_python')
        if filename and label + '_sha256' in data:
            key = (case / Path(filename).name).relative_to(ROOT).as_posix()
            data[label + '_sha256'] = variant_fixtures().get(key, {}).get('sha256', data[label + '_sha256'])
    for variant in data.get('candidates', []):
        if isinstance(variant, dict):
            for field in ('source', 'python', 'file', 'candidate'):
                if isinstance(variant.get(field), str) and variant[field].endswith('.py'):
                    key = (case / variant[field]).relative_to(ROOT).as_posix()
                    if key in variant_fixtures():
                        variant['sha256'] = variant_fixtures()[key]['sha256']
    return data


def split_swiglu_fixture(directory: Path, manifest: Path | None = None) -> Path:
    """Exercise imported-helper checking with the historical two-file shape."""
    original = ROOT / 'bench/tritonbench_g/swiglu_triton'
    for name in ('swiglu_triton.py', 'index_commuted.py'):
        (directory / name).write_text(variant_source(original / name))
    if manifest is not None:
        pairs = [dict(row) for row in pinned_pairs(original)]
        for row in pairs:
            if row['python_function'].endswith('_index_commuted'):
                row['python'] = 'bench/tritonbench_g/swiglu_triton/index_commuted.py'
        manifest.write_text(json.dumps(pairs))
    return directory / 'index_commuted.py'
