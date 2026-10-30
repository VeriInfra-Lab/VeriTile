#!/usr/bin/env python3
"""Check complete original reverse-cumsum expression/control ASTs.

Only source notation is normalized: direct parameter/literal antiquotes,
braces, pointer-constructor keyword names, tuple/list dimensions, literal
boolean spelling, and the independent tuple assignment of program IDs.
Negative controls must reject changed loop bounds, padding, casts and math.
"""
from __future__ import annotations
# Make the repository packages available for direct invocation.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,ast,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bench import audit_source as source
from scripts.check_reversed_cumsum_native import SOURCES

def digest(b):return hashlib.sha256(b).hexdigest()
def dump(t):return ast.dump(t,include_attributes=False)

def braces(text):
    lines=[];current=[];depth=0;parens=[]
    def flush(block=False):
        line=''.join(current).strip();current.clear()
        if not line:return
        if block:
            if not (re.fullmatch(r'for\s+\w+\s+in\s+range\(.+\)',line,re.S) or re.fullmatch(r'if\s+.+',line,re.S)):
                raise ValueError('Unsupported control header: '+line)
            line+=':'
        lines.append('    '*depth+line)
    for c in text:
        if c in '([':parens.append(c);current.append(c)
        elif c in ')]':
            if not parens or parens.pop()!=('(' if c==')' else '['):raise ValueError('Unbalanced delimiter')
            current.append(c)
        elif c=='{':
            if parens:raise ValueError('Expression brace')
            flush(True);depth+=1
        elif c=='}':
            if parens or depth==0:raise ValueError('Invalid closing brace')
            flush();depth-=1
        elif c=='\n' and not parens:flush()
        else:current.append(c)
    if parens or depth:raise ValueError('Unclosed source')
    flush();return ast.parse('\n'.join(lines))

class Canonical(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id in ('true','false'):return ast.Constant(node.id=='true')
        return node
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple):
            target=node.targets[0]
            if not isinstance(node.value,ast.Tuple) or len(target.elts)!=len(node.value.elts):raise ValueError('Tuple assignment shape')
            if not all(isinstance(v,ast.Call) and ast.unparse(v.func)=='tl.program_id' and len(v.args)==1 and isinstance(v.args[0],ast.Constant) and not v.keywords for v in node.value.elts):raise ValueError('Tuple assignment is not independent program IDs')
            if not all(isinstance(t,ast.Name) for t in target.elts):raise ValueError('Non-name tuple target')
            return [ast.Assign(targets=[t],value=v) for t,v in zip(target.elts,node.value.elts)]
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node);name=ast.unparse(node.func)
        if name=='tl.make_block_ptr':
            fields=('base','shape','strides','offsets','block_shape','order')
            if len(node.args)>len(fields):raise ValueError('Extra block-pointer argument')
            args=dict(zip(fields,node.args))
            for kw in node.keywords:
                if kw.arg not in fields or kw.arg in args:raise ValueError('Invalid block-pointer keyword')
                args[kw.arg]=kw.value
            if set(args)!=set(fields):raise ValueError('Incomplete block pointer')
            for key in fields[1:]:
                v=args[key];args[key]=ast.Tuple(elts=v.elts if isinstance(v,(ast.List,ast.Tuple)) else [v],ctx=ast.Load())
            node.args=[];node.keywords=[ast.keyword(arg=k,value=args[k]) for k in fields]
        for kw in node.keywords:
            if kw.arg=='boundary_check':
                if not isinstance(kw.value,(ast.List,ast.Tuple)):raise ValueError('Boundary indices not literal sequence')
                kw.value=ast.Tuple(elts=kw.value.elts,ctx=ast.Load())
        return node

def check(original,lean,params):
    actual=source.strip_region_declarations(lean)
    def anti(m):
        value=m[1]
        if value not in params and not value.isdigit():raise ValueError('Non-direct antiquotation')
        return value
    actual=re.sub(r'\$\((\w+)\)',anti,actual)
    actual=re.sub(r'\((\[[0-9, ]*\])\s*:\s*List Nat\)',r'\1',actual)
    if '$' in actual:raise ValueError('Unresolved source splice')
    expected=Canonical().visit(ast.parse(original));actual=Canonical().visit(braces(actual))
    if dump(expected)!=dump(actual):
        raise ValueError('Complete expression/control AST differs\nEXPECTED '+ast.unparse(ast.fix_missing_locations(expected))+'\nACTUAL '+ast.unparse(ast.fix_missing_locations(actual)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),ROOT/'scripts/check_reversed_cumsum_native.py',ROOT/'scripts/bench/audit_source.py']
    for directory,_,_ in SOURCES.values():paths.extend([ROOT/'bench/tritonbench_g'/directory/(directory+'.py'),ROOT/'bench/tritonbench_g'/directory/'Proofs/Source.lean'])
    captured={p:p.read_bytes() for p in paths};rows=[]
    for kind,(directory,pin,_) in SOURCES.items():
        base=ROOT/'bench/tritonbench_g'/directory;raw=captured[base/(directory+'.py')]
        if digest(raw)!=pin:raise ValueError('Pinned original changed')
        name='chunk_global_reversed_cumsum_'+kind+'_kernel'
        fn=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name)
        params=tuple(x.arg for x in fn.args.args)
        expected_params=('s','o','T','BT') if kind=='scalar' else ('s','z','s_s_h','s_s_t','s_s_d','T','S','BT','BS')
        if params!=expected_params:raise ValueError('Original signature changed')
        original=ast.unparse(ast.Module(body=fn.body,type_ignores=[]))
        selector='reversedScalarSource%' if kind=='scalar' else 'reversedVectorSource%'
        meta='/-\nSource-audit Lean: '+selector+'\nSource-audit args: version=original\n-/\n'
        lean=source.lean_first_triton_body(source.select_lean_audit_source(captured[base/'Proofs/Source.lean'].decode(),meta))
        out='o' if kind=='scalar' else 'z'
        # Dynamic input and output types are independent formal parameters.
        region=re.search(r'tl\.region\s+s\s*=\s*DTYPE,\s*'+out+r'\s*=\s*OUT_DTYPE',lean)
        if region is None:raise ValueError('Storage declarations differ')
        check(original,lean,params)
        mutations=[('- $(1), -$(1), -$(1)', '- $(1), -$(2), -$(1)'),
                   ('tl.cdiv($(T), $(BT)) - $(1)','tl.cdiv($(T), $(BT)) + $(1)'),
                   ('i_t*$(BT)' if kind=='scalar' else 'i_t * $(BT)','i_t + $(BT)'),
                   ('boundary_check=([0] : List Nat)' if kind=='scalar' else 'boundary_check=([0, 1] : List Nat)', 'boundary_check=([0] : List Nat), padding_option="zero"'),
                   ('(b_o).to(p_o.dtype.element_ty)' if kind=='scalar' else '(b_c).to(p_z.dtype.element_ty)', 'b_o' if kind=='scalar' else 'b_c'),
                   ('b_z += b_zz' if kind=='scalar' else 'b_z += tl.sum(b_s, 0)', 'b_z = b_z'),
                   ('b_s - tl.cumsum(b_s, axis=0)' if kind=='scalar' else 'o_i[:, None] <= o_i[None, :]', 'b_s + tl.cumsum(b_s, axis=0)' if kind=='scalar' else 'o_i[:, None] < o_i[None, :]')]
        for before,after in mutations:
            if lean.count(before)<1:raise ValueError('Missing negative-control anchor '+before)
            try:check(original,lean.replace(before,after,1),params)
            except (ValueError,SyntaxError):pass
            else:raise ValueError('Accepted changed source: '+before)
        (a.output/(kind+'-source.txt')).write_text(lean)
        rows.append({'kind':kind,'full_ast_match':True,'negative_controls_rejected':len(mutations),'source_body_sha256':digest(lean.encode())})
    for path,raw in captured.items():
        dest=a.output/'sources'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    changed=[str(p.relative_to(ROOT)) for p,b in captured.items() if p.read_bytes()!=b]
    receipt={'rows':rows,'changed_inputs':changed,'input_sha256':{str(p.relative_to(ROOT)):digest(b) for p,b in captured.items()},'scope':__doc__,'complete_task':False,'success':not changed}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return int(bool(changed))
if __name__=='__main__':raise SystemExit(main())
