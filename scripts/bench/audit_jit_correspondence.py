#!/usr/bin/env python3
"""Inventory selected Python JITs and screen for omitted explicit int64 casts.

This deliberately does NOT certify unflagged tasks. A task is correspondence-
checked only when its explicit jit-correspondence.json passes the AST checker.
Missing source, shared/imported source and other translation forms stay visible.
"""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.bench.jit_correspondence import check_manifest

ROOT = Path(__file__).resolve().parents[2]


def uncomment(text):
    # Preserve line numbers, including nested Lean comments.
    result, i, depth = [], 0, 0
    while i < len(text):
        if text[i:i+2] == '/-':
            depth += 1; result.append('  '); i += 2
        elif depth and text[i:i+2] == '-/':
            depth -= 1; result.append('  '); i += 2
        elif not depth and text[i:i+2] == '--':
            end = text.find('\n', i)
            if end < 0: end = len(text)
            result.append(' ' * (end-i)); i = end
        else:
            result.append('\n' if text[i] == '\n' else ' ' if depth else text[i]); i += 1
    return ''.join(result)


def scan(root=ROOT):
    ledger = json.loads((root/'validation/task_progress.json').read_text())
    rows = []
    for task in ledger['tasks']:
        selected = task['kernel_revision']
        row = {'task': task['task'], 'python': selected['source'], 'lean': selected['kernel_source']}
        py, lean = root/row['python'], root/row['lean']
        if not py.is_file() or not lean.is_file():
            row['status'] = 'source_unavailable'
            rows.append(row); continue
        row['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [py, lean]}
        tree = ast.parse(py.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
            ast.unparse(d.func if isinstance(d, ast.Call) else d) == 'triton.jit'
            for d in n.decorator_list)]
        casts = []
        for fn in functions:
            for n in ast.walk(fn):
                if not isinstance(n, ast.Call): continue
                if (isinstance(n.func, ast.Attribute) and n.func.attr == 'to'
                    and n.args and ast.unparse(n.args[0]) == 'tl.int64') or (
                    ast.unparse(n.func) == 'tl.cast' and len(n.args)>1 and ast.unparse(n.args[1]) == 'tl.int64'):
                    casts.append({'function': fn.name, 'line': n.lineno, 'expression': ast.unparse(n)})
        code = uncomment(lean.read_text())
        explicit = bool(re.search(r'\.to\s*\(\s*tl\.int64\b|tl\.cast\([^\n]*,\s*tl\.int64\b', code))
        row.update(jit_functions=[f.name for f in functions], python_int64_casts=casts,
                   direct_quotation=bool(re.search(r'\btriton\s*(?:\([^\n]*\))?\s*\{', code)),
                   lean_explicit_int64_cast=explicit)
        manifest = lean.parent/'jit-correspondence.json'
        if manifest.is_file():
            row['checked_pairs'] = check_manifest(root, manifest)
            row['status'] = 'exact_ast_correspondence'
        elif casts and row['direct_quotation'] and not explicit:
            row['status'] = 'missing_explicit_int64_cast'
        else:
            row['status'] = 'not_certified_by_this_screen'
        rows.append(row)
    return {'schema': 1, 'scope': 'Selected source inventory and explicit int64-cast screen; unflagged tasks are NOT full correspondence passes. No semantic counterexample is inferred from an omitted cast.',
            'counts': dict(Counter(r['status'] for r in rows)), 'tasks': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = scan()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'tasks': len(result['tasks']), 'counts': result['counts']}, indent=2))


if __name__ == '__main__':
    main()
