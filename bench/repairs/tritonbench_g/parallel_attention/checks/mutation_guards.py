from pathlib import Path
import hashlib,json,re
from inspect_ptx import BASE,scan
from check_norm import audit_norm
from check_scalar import audit_scalar
from check_dot import audit_dot

ROOT=Path(__file__).resolve().parent
OUT=Path('/tmp/parallel-attention-mutations')
OUT.mkdir(exist_ok=True)
rows=[]
def reject(name,source,changed,check):
    target=OUT/(name+source.suffix);target.write_text(changed)
    assert changed!=source.read_text()
    try:check(target)
    except (AssertionError,KeyError,ValueError) as error:
        rows.append({'mutation':name,'source':str(source),'mutant':str(target),
          'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'rejected':True,
          'diagnostic':str(error)[:700]})
    else:raise AssertionError('mutation escaped: '+name)

source=BASE/'actual-binder-ordinary-scaled-forward/kernel.ptx'
text=source.read_text()
changed=text.replace(', 4, 31, -1;',', 2, 31, -1;',1)
reject('normalizer_wrong_xor',source,changed,lambda p:audit_norm(p))
events=scan(source)
e=next(e for e in events if e['loc']==57 and e['op']=='fma.rn.f32')
lines=text.splitlines(keepends=True);old=lines[e['line']-1]
lines[e['line']-1]=old.replace(e['args'][1]+', '+e['args'][2],e['args'][0]+', '+e['args'][0])
reject('normalizer_wrong_square_seed',source,''.join(lines),lambda p:audit_norm(p))

source=BASE/'actual-binder-ordinary-scaled-backward/kernel.ptx';text=source.read_text()
e=next(e for e in scan(source) if e['loc']==156 and e['op']=='mul.f32')
lines=text.splitlines(keepends=True)
lines[e['line']-1]=lines[e['line']-1].replace(', '+e['args'][0]+', '+e['args'][1]+';',', '+e['args'][1]+', '+e['args'][0]+';')
reject('coefficient_operand_swap',source,''.join(lines),lambda p:audit_scalar(p,True,False,False))

source=BASE/'actual-binder-sharded-scaled-backward/kernel.ptx';text=source.read_text()
e=next(e for e in scan(source) if e['loc']==149 and e['op']=='add.f32')
lines=text.splitlines(keepends=True)
lines[e['line']-1]=lines[e['line']-1].replace(', '+e['args'][0]+', '+e['args'][1]+';',', '+e['args'][1]+', '+e['args'][0]+';')
reject('sharded_dq_seed_order',source,''.join(lines),lambda p:audit_scalar(p,True,True,True))

source=BASE/'actual-binder-tiny-scaled-backward/kernel.ptx';text=source.read_text()
e=next(e for e in scan(source) if e['loc']==254 and e['op']=='mul.f32' and e['node'][1][0]=='fma.rn.f32')
lines=text.splitlines(keepends=True)
lines[e['line']-1]=lines[e['line']-1].replace(', '+e['args'][0]+', '+e['args'][1]+';',', '+e['args'][1]+', '+e['args'][0]+';')
reject('tiny_padded_coefficient_order',source,''.join(lines),lambda p:audit_scalar(p,True,False,False))

source=BASE/'actual-binder-ordinary-scaled-forward/kernel.llir';text=source.read_text()
matches=list(re.finditer(r'@llvm.fmuladd.f32\(float [^,]+, float [^,]+, float (%[\w.]+)\)',text))
m=matches[0]
changed=text[:m.start(1)]+'0.000000e+00'+text[m.end(1):]
reject('dot_dropped_carry_edge',source,changed,lambda p:audit_dot(p,{'BK':64,'BV':64}))

(ROOT/'mutation-guards.json').write_text(json.dumps({'status':'PASS','count':len(rows),'checks':rows,
 'compiler_or_gpu_invocations':0,'producer_files_modified':False,'source_sha256':hashlib.sha256((ROOT.parent/'parallel_attention_repaired.py').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('Rejected',len(rows),'structural mutants.')
