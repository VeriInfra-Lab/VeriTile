from pathlib import Path
import ast,json,hashlib
P=Path(__file__).resolve().parent;source=P/'native/original.py';data=source.read_text();tree=ast.parse(data)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='token_softmax_fwd')
text=''.join(data.splitlines(keepends=True)[min([fn.lineno]+[d.lineno for d in fn.decorator_list])-1:fn.end_lineno]);(P/'host-exact.py').write_text(text)
block=next(n for n in fn.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='BLOCK_SIZE');assert ast.unparse(block.value)=='triton.next_power_of_2(max_input_len)'
initial=next(n for n in fn.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='num_warps');assert ast.literal_eval(initial.value)==4
branches=[n for n in fn.body if isinstance(n,ast.If)];assert len(branches)==2
assert [ast.unparse(n.test) for n in branches]==['BLOCK_SIZE >= 2048','BLOCK_SIZE >= 4096']
assert [ast.unparse(n.body[0]) for n in branches]==['num_warps = 8','num_warps = 16']
rows=json.loads((P/'native/receipt.json').read_text())['records']
for r in rows:
 width=r['constexprs']['BLOCK_SIZE'];expected=16 if width>=4096 else 8 if width>=2048 else 4
 assert width>0 and width&(width-1)==0 and r['options']['num_warps']==expected
x={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'host_exact_sha256':hashlib.sha256((P/'host-exact.py').read_bytes()).hexdigest(),'host_first_line':min([fn.lineno]+[d.lineno for d in fn.decorator_list]),'host_last_line':fn.end_lineno,'profiles_checked':len(rows),'host_warp_branches':[[1,4],[2048,8],[4096,16]],'compiler_option_notes':'num_stages=3 is explicit capture configuration; original host leaves it to the compiler default. Fusion true/false are separate explicit compiler profiles of the unmodified JIT body; the host does not itself select false.','native/host.py':'Historical partial context excerpt; authoritative exact full function is host-exact.py and original.py.'}
(P/'host-profile-audit.json').write_text(json.dumps(x,indent=2)+'\n');print('PASS',len(rows),'host warp selections')
