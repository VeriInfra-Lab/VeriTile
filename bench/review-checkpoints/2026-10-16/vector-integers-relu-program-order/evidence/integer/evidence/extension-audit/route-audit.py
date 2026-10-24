from pathlib import Path
import json,re,hashlib,collections
root=Path(__file__).resolve().parent
rows=json.loads((root/'frontend-probes.json').read_text())
promo={(r['x'],r['y']):r for r in json.loads((root/'promotion-matrix.json').read_text())}
ints={'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
flts={'float16','bfloat16','float32','float64'}
checks=[]
for r in rows:
    if r['x'] not in ints or r['y'] not in ints:continue
    assert r['status']=='frontend_generated'
    ir=(root/r['ttir_path']).read_text()
    adds=[l for l in ir.splitlines() if 'arith.addi' in l and '%output' in l]
    expected=int(re.search(r'\d+',promo[r['x'],r['y']]['add_dtype']).group())
    got=int(re.search(r'tensor<1024xi(\d+)>',adds[-1]).group(1));assert got==expected
    sw=8 if r['x']=='bool' else int(re.search(r'\d+',r['x']).group())
    store=next(l for l in ir.splitlines() if 'tt.store' in l)
    assert f'tt.ptr<i{sw}>' in store
    for side in ['x','y']:
        if r[side]=='bool':
            assert any('tt.load' in l and f'%{side}_' in l and 'tt.ptr<i8>' in l for l in ir.splitlines())
            assert any('arith.cmpi ne' in l and f'%{side}_' in l for l in ir.splitlines())
    checks.append({'x':r['x'],'y':r['y'],'add_width':got,'store_width':sw,'status':'all_route_checks_passed'})
mixed=[]
for r in rows:
    if not ((r['x'] in ints and r['y'] in flts)or(r['y'] in ints and r['x'] in flts)):continue
    assert r['status']=='frontend_generated'
    lines=r['arithmetic_and_cast_lines']
    c=promo[r['x'],r['y']]['add_dtype']
    floatdtype={'fp16':'f16','fp32':'f32','fp64':'f64'}[c]
    assert any('arith.addf' in l and f'tensor<1024x{floatdtype}>' in l for l in lines)
    if r['x'] in ints:
        width=8 if r['x']=='bool' else int(re.search(r'\d+',r['x']).group())
        convert='fptoui' if r['x'].startswith('uint') else 'fptosi'
        assert any('arith.'+convert in l and f'tensor<1024xi{width}>' in l for l in lines)
        if c=='fp16':assert any('arith.extf' in l and 'tensor<1024xf16> to tensor<1024xf32>' in l for l in lines)
    mixed.append({'x':r['x'],'y':r['y'],'add_dtype':c,'status':'route_checks_passed','route':lines})
fnuz=[r for r in rows if 'fnuz' in r['x'] or 'fnuz' in r['y']]
assert len(fnuz)==64 and all(r['status']=='frontend_error' for r in fnuz)
assert len(checks)==81 and len(mixed)==72
receipt={'task':'source-bound default sm90 frontend route checks','frontend_profiles':289,'frontend_successes':189,'frontend_errors':100,'integer_route_checks':checks,'mixed_float_integer_route_checks':mixed,'fnuz_pairs_rejected_at_sm90':64,'native_backend_compiled':False,'gpu_executed':False,'formal_native_semantics_proved':False,'debug_overflow_boundary':'Frontend options debug=false,sanitize_overflow=true are retained. Intermediate overflow computations do not by themselves prove whether runtime traps exist after native lowering. Debug/sanitizer environment and finite-width launch/address errors remain structural obligations.','bool_semantics':'i8 load, compare !=0 -> i1; bool/bool arith.addi i1; store pointer reinterpreted i8, conversion to signed8; mixed float bool store uses fptosi, fp16 output first extends to fp32.','bindings':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'frontend-probes.json',root/'promotion-matrix.json',root/'receipt.json']}}
p=root/'route-review.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'route_review_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'integer_checks':len(checks),'mixed_checks':len(mixed),'fnuz_rejections':len(fnuz)}))
