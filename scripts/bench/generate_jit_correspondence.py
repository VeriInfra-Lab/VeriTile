"""Discover Python/Lean body pairs and generate their correspondence snapshot.

No jit-correspondence.json is read during discovery. Complete pasted bodies, their original
Python signatures, Lean call declarations and helper links determine the pairs.
The existing checker validates every generated pair. Pointer types come from
the existing typed call declaration: this tool does not infer unannotated
pointer types from arithmetic, or certify that a host launch supplies them.

By default, check directly without writing anything. --write explicitly saves
the generated snapshot; --check compares it with the saved snapshot without
updating it. Neither generation nor a source match is a proof certificate.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import ast
import copy
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import re
from dataclasses import dataclass

from scripts.bench import jit_correspondence as checker
from scripts.bench.audit_source import lean_code_mask

ROOT = Path(__file__).resolve().parents[2]
IDENT = r"[A-Za-z_][\w.']*"
MARKER = r"\b(?:triton|triton_function|triton_procedure|triton_rebinding_procedure|triton_source_procedure)\b"


def local(root: Path, path: Path | str) -> Path:
    value = (root / path).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError(f"source outside repository: {path}")
    return value


def body_key(statements: list[ast.stmt]) -> str:
    tree = ast.Module(copy.deepcopy(statements), [])
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body[0].value.value = inspect.cleandoc(tree.body[0].value.value)
    return ast.dump(tree)


def clause(header: str, label: str) -> str | None:
    found = re.findall(r"\b" + re.escape(label) + r"\s*\[([^\]]*)\]", header)
    if len(found) > 1:
        raise ValueError(f"duplicate {label} clause")
    return found[0] if found else None


def bindings(source: str) -> dict[str, str]:
    source = source.replace('«', '').replace('»', '')
    result = {}
    for item in source.split(',') if source.strip() else []:
        match = re.fullmatch(r"\s*(\w+)\s*:=\s*(.+?)\s*", item)
        if not match or match[1] in result:
            raise ValueError("invalid or duplicate source binding")
        result[match[1]] = match[2].replace('«', '').replace('»', '')
    return result


def import_chains(tree: ast.Module, names: set[str]) -> dict[str, list[str]]:
    """Read import/fallback spelling; the checker validates the full structure."""
    result = {name: [] for name in names}
    for statement in tree.body:
        def visit(node):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name in result:
                        result[name].append(node.module + '.' + alias.name)
            elif isinstance(node, ast.Try):
                for child in node.body:
                    visit(child)
                for handler in node.handlers:
                    for child in handler.body:
                        visit(child)
        visit(statement)
    return result


@dataclass
class Quotation:
    path: str
    definition: str
    text: str
    header: str
    body: str
    key: str


def quotations(root: Path, case: Path) -> list[Quotation]:
    # Read executable source entries and imported benchmark source entries,
    # never choose quotation-shaped snippets out of proof files.
    pending = sorted(set(case.glob('*Kernels.lean')) | set((case/'Sources').rglob('*.lean')))
    seen, result = set(), []
    while pending:
        path = local(root, pending.pop(0))
        if path in seen:
            continue
        seen.add(path)
        text = path.read_text()
        code = lean_code_mask(text)
        for module in re.findall(r"(?m)^import\s+(bench\.[\w.]+)", code):
            imported = local(root, module.replace('.', '/') + '.lean')
            if imported.name.endswith('Kernels.lean') or 'Sources' in imported.parts:
                pending.append(imported)
        declarations = list(re.finditer(r"(?m)^(?:(?:private|protected|noncomputable)\s+)*def\s+(" + IDENT + r")\b", code))
        counts = Counter(d[1] for d in declarations)
        duplicated = {name for name, count in counts.items() if count > 1}
        repeated_signatures = {
            d[1] for d in declarations if d[1] in duplicated
            and re.match(r"\s*:\s*CallSignature\b", code[d.end():])
        }
        scopes = []
        if duplicated:
            for start in re.finditer(r'(?m)^namespace[ \t]+([\w.]+)[ \t]*$', code):
                end = re.search(rf'(?m)^end[ \t]+{re.escape(start[1])}[ \t]*$', code[start.end():])
                if end is not None:
                    scopes.append((start.end(), start.end() + end.start(), start[1]))
        for index, declaration in enumerate(declarations):
            end = declarations[index+1].start() if index+1 < len(declarations) else len(code)
            if not re.search(MARKER, code[declaration.end():end]):
                continue
            name = declaration[1]
            source_header = code[declaration.end():end].partition('{')[0]
            if name in duplicated or any(re.search(rf"\b{re.escape(sig)}\b", source_header)
                                         for sig in repeated_signatures):
                enclosing = [s for s in scopes if s[0] <= declaration.start() < s[1]]
                if enclosing:
                    name = max(enclosing)[2] + '::' + name
            try:
                header, body = checker.quotation(text, name, raw_python=True)
                statements = ast.parse('def __pasted__():\n' + body).body[0].body
            except (ValueError, SyntaxError):
                # An auxiliary legacy quotation is not a pasted-body source.
                # Every discovered Python JIT still must find an exact pair.
                continue
            result.append(Quotation(path.relative_to(root).as_posix(), name, text, header, body, body_key(statements)))
    return result


def record(root: Path, py_path: str, function: ast.FunctionDef, quote: Quotation) -> dict:
    header = quote.header
    row = dict(python=py_path, lean=quote.path, python_function=function.name, lean_definition=quote.definition)
    if re.search(r'\b(?:triton_source_procedure|with_triton_params)\b', header):
        row['kind'] = 'source_procedure'
        raw = clause(header, 'fp32_constants') or ''
        row['expected_fp32_constants'] = [n.strip() for n in raw.split(',') if n.strip()]
        return row
    if re.search(r'\btriton_(?:rebinding_)?procedure\b', header):
        row['kind'] = 'procedure_jit_function'
        row['signature_definition'] = re.search(r'\btriton_(?:rebinding_)?procedure\s+(' + IDENT + ')', header)[1]
        if re.search(r'\btriton_rebinding_procedure\b', header):
            row['expected_parameter_rebinding'] = True
    elif re.search(r'\btriton_function\b', header):
        if re.search(r':\s*UniformJitFunction\b', header):
            row['kind'] = 'uniform_jit_function'
        elif clause(header, 'constexpr_args') is not None:
            row['kind'] = 'statement_jit_function'
        else:
            row['kind'] = 'binary_pure_jit_function' if len(function.args.args) == 2 else 'pure_jit_function'
            imports = clause(header, 'with_triton_imports')
            if imports is not None:
                row['expected_imports'] = import_chains(ast.parse((root/py_path).read_text()), set(bindings(imports)))
        return row
    else:
        control = re.search(r'\bwith_control_(call|function)\s+(?:"[^"\n]+"\s+)?(' + IDENT + ')', header)
        if control:
            row['kind'] = 'control_source'
            row['signature_definition'] = control[2]
            row['expected_constexpr_bools'] = {n: {'true': True, 'false': False}[v] for n,v in bindings(clause(header,'constexpr_bools') or '').items()}
            if control[1] == 'function':
                row['expected_constexpr_dtypes'] = {n:v.removeprefix('.') for n,v in bindings(clause(header,'dtype_constants') or '').items()}
                row['expected_jit_calls'] = {n:json.loads(v) for n,v in bindings(clause(header,'jit_calls') or '').items()}
        else:
            signature = re.search(r'\bwith_triton_(?:signature|call)\s+\(?\s*(' + IDENT + ')', header)
            if not signature:
                raise ValueError('pasted body requires its input call declaration')
            row['signature_definition'] = signature[1]
    source, _ = checker.source_scope(quote.text, quote.definition)
    row['expected_signature'] = checker.call_signature(source, row['signature_definition'])
    if row.get('kind'):
        return row

    mappings = {
        'call_args':'expected_call_bindings', 'pointer_offsets':'expected_pointer_offsets',
        'signed_pointer_offsets':'expected_pointer_offsets', 'optional_pointers':'expected_optional_pointers',
        'guarded_optional_pointers':'expected_optional_pointers', 'pure_functions':'expected_pure_functions',
        'binary_functions':'expected_binary_functions', 'jit_functions':'expected_jit_functions',
        'jit_procedures':'expected_jit_procedures', 'rebinding_jit_procedures':'expected_jit_procedures',
        'source_procedures':'expected_source_procedures', 'integer_args':'expected_integer_args',
        'arange_args':'expected_arange_args', 'ordered_sum_trees':'expected_ordered_sum_trees',
    }
    for name, key in mappings.items():
        raw = clause(header, name)
        if raw is not None:
            row[key] = bindings(raw)
    if clause(header,'rebinding_jit_procedures') is not None:
        row['expected_parameter_rebinding'] = True
    for name, key in [('fp32_constants','expected_fp32_constants'),('with_triton_natural_reads','expected_natural_reads')]:
        raw = clause(header,name)
        if raw is not None:
            row[key] = [n.strip() for n in raw.split(',') if n.strip()]
    raw = clause(header, 'pointer_cells')
    if raw is not None:
        pairs = re.findall(r'(\w+)\s*:=\s*\(([^,()]+),([^,()]+)\)',raw)
        row['expected_pointer_cells'] = {n:[a.strip(),b.strip()] for n,a,b in pairs}
    if re.search(r'\bfp32_bits\s*\[',header):
        row['expected_philox_call'] = {label:bindings(clause(header,label) or '') for label in ['fp32_bits','i64_args','arange_args','uniform_helpers']}
        row.pop('expected_arange_args',None)
    if re.search(r'\bscalar_infinity_where\s+in\b',header):
        row['expected_scalar_infinity_where'] = True
    if re.search(r'\b(?:static_strides|welford_tree)\b',header):
        strides = {}
        for node in ast.walk(function):
            if (isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='tl.full'
                    and len(node.value.args)==3 and isinstance(node.value.args[0],ast.Tuple)
                    and not node.value.args[0].elts and isinstance(node.value.args[1],ast.Name)
                    and ast.unparse(node.value.args[2])=='tl.int64'):
                strides[node.targets[0].id] = node.value.args[1].id
        row['expected_static_strides'] = strides
    welford = re.search(r'\bwelford_tree\s+(\w+)',header)
    if welford:
        width = re.search(r'\b' + re.escape(welford[1]) + r'\s*:\s*DSL\.WelfordSource\.Tree\s+(\w+)',header)
        if not width:
            raise ValueError('missing typed Welford tree width')
        row['expected_welford_tree'] = dict(argument=welford[1],width=width[1])
    used = {n.id for n in ast.walk(function) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    namespaces = import_chains(ast.parse((root/py_path).read_text()),used & {'libdevice','triton_helpers'})
    namespace_bindings = {n:targets[0] for n,targets in namespaces.items() if len(targets)==1}
    if namespace_bindings:
        row['expected_namespace_imports'] = namespace_bindings
    constants = set(row['expected_signature']['constexpr'])
    closed = {}
    for name,typ,value in re.findall(r'\blet\s+(«?\w+»?)\s*:\s*(Nat|FloatDType|SourceDType)\s*:=\s*((?:FloatDType\.|SourceDType\.|\.)?\w+)\b(?=[ \t]*(?:\n|$))',checker.strip_lean_comments(header)):
        name = name.strip('«»')
        if name in constants and (value.isdigit() if typ=='Nat' else value.startswith(('.',typ+'.'))):
            closed[name] = value if typ=='Nat' else typ+'.'+value.rsplit('.',1)[-1]
    if closed:
        row['expected_constexpr_bindings'] = closed
    return row


def check_pair(root: Path, row: dict, quote: Quotation) -> None:
    py = (root/row['python']).read_text()
    kind = row.get('kind')
    common = [py,quote.text,row['python_function'],row['lean_definition']]
    if kind == 'source_procedure':
        checker.check_source_procedure(*common,row['expected_fp32_constants'])
    elif kind == 'procedure_jit_function':
        checker.check_procedure_function(*common,row['signature_definition'],row['expected_signature'],row.get('expected_parameter_rebinding',False))
    elif kind == 'statement_jit_function':
        checker.check_statement_function(*common)
    elif kind == 'uniform_jit_function':
        checker.check_uniform_function(*common)
    elif kind in ('pure_jit_function','binary_pure_jit_function'):
        checker.check_pure_function(*common,2 if kind=='binary_pure_jit_function' else 1,row.get('expected_imports'))
    elif kind == 'control_source':
        checker.check_control_source(*common,row['signature_definition'],row['expected_signature'],row['expected_constexpr_bools'],row.get('expected_constexpr_dtypes'),row.get('expected_jit_calls'))
    else:
        keys = inspect.signature(checker.check).parameters
        checker.check(*common,**{k:v for k,v in row.items() if k in keys and k not in ('python_function','lean_definition')})


def linked_helper(root: Path, parent: dict, name: str, target: str, helper: dict) -> bool:
    if helper['lean'] != parent['lean'] or helper['lean_definition'] != target:
        return False
    if helper['python'] == parent['python']:
        return helper['python_function'] == name
    module=ast.parse((root/parent['python']).read_text())
    return any(node.level==0 and node.module==Path(helper['python']).stem
               and any((alias.asname or alias.name)==name and alias.name==helper['python_function']
                       for alias in node.names)
               for node in module.body if isinstance(node,ast.ImportFrom))


def snapshot_key(records: list[dict]) -> list[str]:
    # Variant labels are human-readable identifiers, not the invocation ABI.
    # The ABI and every generated expected_* field still compare exactly.
    return sorted(json.dumps({k:v for k,v in row.items() if k not in ('invocation_variant','profile_variant')},
                             sort_keys=True) for row in records)


def snapshot_matches(saved: list[dict], generated: list[dict]) -> bool:
    def key(row):
        return tuple(row[k] for k in ('python','python_function','lean','lean_definition'))
    old={key(r):r for r in saved};new={key(r):r for r in generated}
    if len(old)!=len(saved) or len(new)!=len(generated) or set(old)!=set(new):
        return False
    # Older snapshots can omit newly extracted, checked metadata (e.g. a
    # literal constexpr). Every expectation they actually pin remains exact.
    return all(new[k].get(field)==value for k,row in old.items() for field,value in row.items()
               if field not in ('invocation_variant','profile_variant'))


def generate(root: Path, case: Path | str, selected: dict[tuple[str,str],set[tuple[str,str]]] | None = None,
             python_paths: list[Path | str] | None = None) -> list[dict]:
    root = root.resolve();case = local(root,case)
    if not case.is_dir():
        raise ValueError('case must be an existing source directory')
    index = {}
    for quote in quotations(root,case):
        index.setdefault(quote.key,[]).append(quote)
    candidates, failures, expected = [], {}, []
    sources = set(case.rglob('*.py')) | {local(root,p) for p in python_paths or []}
    collection=next((p for p in case.parents if p.parent.name=='bench'),None)
    references=collection/'source-references.json' if collection else None
    if references and references.is_file():
        reference_map=json.loads(references.read_text())
        if not isinstance(reference_map,dict):
            raise ValueError('shared source references must be a mapping')
        shared=reference_map.get(case.name)
        if shared is not None:
            if not isinstance(shared,str):
                raise ValueError('shared Python source reference must be a path')
            sources.add(local(root,shared))
    for path in sorted(sources):
        if '__pycache__' in path.parts:
            continue
        path = local(root,path);py = path.relative_to(root).as_posix();tree = ast.parse(path.read_text())
        functions = [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and any(
            ast.unparse(d.func if isinstance(d,ast.Call) else d)=='triton.jit' for d in n.decorator_list)]
        if len({fn.name for fn in functions})!=len(functions):
            raise ValueError(f'ambiguous Python JIT names: {py}')
        for fn in sorted(functions,key=lambda n:n.lineno):
            key = (py,fn.name);expected.append(key)
            for quote in index.get(body_key(fn.body),[]):
                if selected and key in selected and (quote.path,quote.definition) not in selected[key]:
                    continue
                try:
                    row = record(root,py,fn,quote)
                    check_pair(root,row,quote)
                    candidates.append(row)
                except (ValueError,KeyError,SyntaxError,AttributeError) as error:
                    failures.setdefault(key,[]).append(f'{quote.definition}: {error}')
    if not expected:
        raise ValueError('no Python @triton.jit functions found')
    if selected and set(selected)-set(expected):
        raise ValueError('explicit binding names an undiscovered Python JIT')
    # A parent can only use a helper whose complete Python body has its own
    # compatible pair; identical parent text cannot select the wrong callee.
    while True:
        retained=[]
        for row in candidates:
            links = {}
            for label in ['expected_pure_functions','expected_binary_functions','expected_jit_functions',
                          'expected_jit_procedures','expected_source_procedures']:
                links.update(row.get(label,{}))
            links.update(row.get('expected_philox_call',{}).get('uniform_helpers',{}))
            if all(any(linked_helper(root,row,name,target,h) for h in candidates)
                   for name,target in links.items()):
                retained.append(row)
        if len(retained)==len(candidates):
            break
        candidates=retained
    output=[]
    for key in expected:
        rows=[r for r in candidates if (r['python'],r['python_function'])==key]
        demanded = {target for parent in candidates if parent['python']==key[0]
                    for label in ['expected_pure_functions','expected_binary_functions','expected_jit_functions',
                                  'expected_jit_procedures','expected_source_procedures']
                    for name,target in parent.get(label,{}).items() if name==key[1]}
        if demanded:
            rows=[r for r in rows if r['lean_definition'] in demanded]
        # Prefer a case's own literal declaration over the identical imported
        # declaration with the same ABI; retain genuinely distinct typed views.
        owned={json.dumps([r.get('expected_signature'),r.get('expected_constexpr_bindings',{})],sort_keys=True)
               for r in rows if local(root,r['lean']).is_relative_to(case)}
        rows=[r for r in rows if local(root,r['lean']).is_relative_to(case) or
              json.dumps([r.get('expected_signature'),r.get('expected_constexpr_bindings',{})],sort_keys=True) not in owned]
        if not rows:
            detail='; '.join(failures.get(key,[])[:2]) or 'complete body differs, a helper is unresolved, or a binding is stale'
            raise ValueError(f'No exact source pair for {key[0]}::{key[1]}: {detail}')
        if selected and key in selected and {(r['lean'],r['lean_definition']) for r in rows}!=selected[key]:
            raise ValueError(f'An explicit source binding did not resolve: {key[0]}::{key[1]}')
        if len(rows)>1:
            profiles=[json.dumps([r.get('expected_signature'),r.get('expected_constexpr_bindings',{})],sort_keys=True) for r in rows]
            if len(set(profiles))!=len(profiles) and not (selected and key in selected):
                choices=', '.join(r['lean']+'::'+r['lean_definition'] for r in rows)
                raise ValueError(f'Ambiguous source pair for {key[0]}::{key[1]}; use --bind to select: {choices}')
            for row,profile in zip(rows,profiles,strict=True):
                row['invocation_variant']='auto_'+hashlib.sha256((profile+row['lean_definition']).encode()).hexdigest()[:12]
        output.extend(rows)
    checker.check_records(root,output)
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path,help='directory containing Python source and Lean source entries')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--python',type=Path,action='append',default=[],help='additional shared Python reference outside the case; may repeat')
    parser.add_argument('--bind',action='append',default=[],metavar='PYTHON::FUNCTION=LEAN::DEFINITION',help='select an otherwise ambiguous pair; repeat for distinct invocation variants')
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--write',action='store_true',help='explicitly generate/update jit-correspondence.json')
    actions.add_argument('--check',action='store_true',help='compare with saved JSON, without rewriting it')
    parser.add_argument('--print',action='store_true',dest='print_manifest',help='print the generated records')
    args=parser.parse_args();selected={}
    try:
        for value in args.bind:
            left,right=value.split('=',1);py,fn=left.rsplit('::',1);lean,definition=right.split('::',1)
            py=local(args.root,py).relative_to(args.root.resolve()).as_posix()
            lean=local(args.root,lean).relative_to(args.root.resolve()).as_posix()
            selected.setdefault((py,fn),set()).add((lean,definition))
        rows=generate(args.root,args.case,selected,args.python)
        manifest=local(args.root,local(args.root,args.case)/'jit-correspondence.json')
        if args.check:
            saved=json.loads(manifest.read_text())
            checker.check_records(args.root,saved)
            if not snapshot_matches(saved,rows):
                raise ValueError('Generated correspondence differs from the saved snapshot; review source/input changes before --write')
        written=False
        if args.write:
            unchanged=manifest.is_file() and snapshot_key(json.loads(manifest.read_text()))==snapshot_key(rows)
            if not unchanged:
                manifest.write_text(json.dumps(rows,indent=2)+'\n')
                written=True
        print(json.dumps(rows if args.print_manifest else {'status':'PASS','source_pairs':len(rows),'manifest_written':written,'proof_acceptance':False},indent=2))
    except (ValueError,OSError,SyntaxError) as error:
        parser.exit(1,str(error)+'\n')


if __name__=='__main__':
    main()
