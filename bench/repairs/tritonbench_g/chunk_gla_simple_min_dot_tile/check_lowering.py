"""Inspect the saved full-kernel native binder compilation; no new compiler run."""
import hashlib
import json
import re
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'.veritile-cache/task-deliveries/chunk_gla_simple_min_dot_tile'
base=OUT/'native-admission/B1-H1-T17-K129-V17-BT16'
ir=base.with_suffix('.ttir').read_text()
ptx=base.with_suffix('.ptx').read_text()
dots=re.findall(r'tt.dot ([^\n]+)',ir)
assert len(dots)==3 and all('tf32' not in x for x in dots)
assert 'scf.for' in ir and 'iter_args' in ir
assert 'fma.rn.f32' in ptx and 'mma.sync' not in ptx
assert 'ex2.approx' in ptx
loop=re.search(r'iter_args\((%\w+) = %\w+, (%\w+) = %\w+\)',ir)
assert loop
assert re.search(r'tt.dot [^,\n]+, [^,\n]+, '+re.escape(loop[1])+r' ',ir)
assert re.search(r'tt.dot [^,\n]+, [^,\n]+, '+re.escape(loop[2])+r' ',ir)
final_carry=re.findall(r'tt.dot [^,\n]+, [^,\n]+, (%\w+) ',ir)[-1]
assert re.search(re.escape(final_carry)+r' = arith.mulf ',ir)
def pin(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
result={'checker':pin(Path(__file__).resolve()),'source':pin(HERE/'chunk_gla_simple_min_dot_tile.py'),
'input_artifacts':[pin(base.with_suffix(ext)) for ext in ('.ttir','.ptx')],
'profile':{'target':'SM80','BT':16,'K':129,'V':17,'BK':64,'BV':32,'num_warps':4},
'checks':['three IEEE FP32 dot sites','two live loop accumulator carries','final SV carries gated history','scalar FMA lowering','exp2 lowering','no tensorcore MMA'],
'claim':'Inspected full-kernel compiler realization; no GPU run and no all-shape compiler correctness proof.'}
(OUT/'lowering.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: actual binder profile, all three live-carry IEEE dots, scalar FMA and exp2; no GPU execution')
