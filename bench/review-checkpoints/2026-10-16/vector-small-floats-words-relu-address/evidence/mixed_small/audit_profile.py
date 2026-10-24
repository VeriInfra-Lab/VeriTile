"""Check the complete SSA conversion/load/add/store route for all36 profiles."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
previous = root.parent / 'vector-addition-dtype-extension-wip'
audit = previous / 'evidence/extension-audit'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(previous / 'evidence/receipt.json') == '85dc08106852c4a3f69516046182954693eb5afd0158064db358c708a70b9100'
rows = json.loads((audit / 'frontend-probes.json').read_text())
ints = {'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
floats = {'float16','bfloat16'}
output = root / 'evidence/source-profile'
output.mkdir(exist_ok=True)
checked = []
var_re = r'%[A-Za-z0-9_]+'
for row in rows:
    x,y = row['x'],row['y']
    if not ((x in ints and y in floats) or (y in ints and x in floats)):
        continue
    assert row['status'] == 'frontend_generated'
    path = audit / row['ttir_path']; assert sha(path) == row['ttir_sha256']
    lines = path.read_text().splitlines()
    defs = {}
    for line in lines:
        m = re.match(r'\s*(' + var_re + r') = ([A-Za-z0-9_.]+) (.*)',line)
        if m:
            defs[m[1]] = {'op':m[2],'text':m[3],'vars':re.findall(var_re,m[3]),'line':line.strip()}
    integer_first = x in ints
    int_side, float_side = ('x','y') if integer_first else ('y','x')
    integral = x if integer_first else y
    floating = y if integer_first else x
    storage = 'f16' if floating == 'float16' else 'bf16'
    compute = 'f16' if floating == 'float16' else 'f32'
    width = 1 if integral == 'bool' else int(re.search(r'\d+',integral).group())
    def unique(predicate):
        matches = [v for v,d in defs.items() if predicate(v,d)]
        assert len(matches) == 1,(x,y,matches)
        return matches[0]
    int_load = unique(lambda v,d: v.startswith('%'+int_side+'_') and d['op']=='tt.load')
    float_load = unique(lambda v,d: v.startswith('%'+float_side+'_') and d['op']=='tt.load')
    assert f'tt.ptr<{storage}>' in defs[float_load]['text']
    assert f'tt.ptr<i{8 if integral == "bool" else width}>' in defs[int_load]['text']
    assert len(defs[int_load]['vars']) == len(defs[float_load]['vars']) == 2  # no other operand
    mask = defs[int_load]['vars'][1]
    assert defs[float_load]['vars'][1] == mask
    int_value = int_load
    if integral == 'bool':
        int_value = unique(lambda v,d: d['op']=='arith.cmpi' and d['text'].startswith('ne,') and d['vars'][0]==int_load)
        assert 'dense<0>' in defs[defs[int_value]['vars'][1]]['text']
    int_op = 'arith.uitofp' if integral=='bool' or integral.startswith('uint') else 'arith.sitofp'
    int_cast = unique(lambda v,d: d['op']==int_op and d['vars']==[int_value])
    assert f'tensor<1024xi{width}> to tensor<1024x{compute}>' in defs[int_cast]['text']
    float_value = float_load
    if floating=='bfloat16':
        float_value = unique(lambda v,d: d['op']=='arith.extf' and d['vars']==[float_load])
        assert 'tensor<1024xbf16> to tensor<1024xf32>' in defs[float_value]['text']
    ordered = [int_cast,float_value] if integer_first else [float_value,int_cast]
    add = unique(lambda v,d: d['op']=='arith.addf' and d['vars']==ordered)
    assert f'tensor<1024x{compute}>' in defs[add]['text']
    result = add
    if integer_first:
        if floating=='float16':
            result = unique(lambda v,d: d['op']=='arith.extf' and d['vars']==[add])
            assert 'tensor<1024xf16> to tensor<1024xf32>' in defs[result]['text']
        conversion = 'arith.fptoui' if integral.startswith('uint') else 'arith.fptosi'
        result = unique(lambda v,d: d['op']==conversion and d['vars']==[result])
        target = f'i{8 if integral=="bool" else width}'
        assert f'tensor<1024xf32> to tensor<1024x{target}>' in defs[result]['text']
    else:
        target = storage
        if floating=='bfloat16':
            result = unique(lambda v,d: d['op']=='arith.truncf' and d['vars']==[add])
            assert 'tensor<1024xf32> to tensor<1024xbf16>' in defs[result]['text']
    stores = [line.strip() for line in lines if line.strip().startswith('tt.store ')]
    assert len(stores)==1
    ptr,stored,store_mask = re.findall(var_re,stores[0])
    assert stored==result and store_mask==mask and f'tt.ptr<{target}>' in stores[0]
    offsets = []
    for pointer,expected in [(defs[int_load]['vars'][0],int_side),(defs[float_load]['vars'][0],float_side),(ptr,'output')]:
        node = defs[pointer]
        if node['op']=='tt.bitcast':node = defs[node['vars'][0]]
        assert node['op']=='tt.addptr'
        base,off = node['vars']; offsets.append(off)
        assert defs[base]['op']=='tt.splat' and defs[base]['vars']==['%'+expected+'_ptr']
    assert len(set(offsets))==1
    offset_ssa=offsets[0]
    offset_node=defs[offset_ssa]
    assert offset_node['op']=='arith.addi' and 'tensor<1024xi32>' in offset_node['text']
    assert len(offset_node['vars'])==2
    block_splat, lanes=offset_node['vars']
    assert defs[lanes]['op']=='tt.make_range'
    assert 'end = 1024 : i32, start = 0 : i32' in defs[lanes]['text']
    assert 'tensor<1024xi32>' in defs[lanes]['text']
    assert defs[block_splat]['op']=='tt.splat' and len(defs[block_splat]['vars'])==1
    assert 'i32 -> tensor<1024xi32>' in defs[block_splat]['text']
    block=defs[block_splat]['vars'][0]
    assert defs[block]['op']=='arith.muli' and ': i32' in defs[block]['text']
    pid,block_size=defs[block]['vars']
    assert defs[pid]['op']=='tt.get_program_id' and defs[pid]['text'].startswith('x : i32')
    assert defs[block_size]['op']=='arith.constant' and defs[block_size]['text'].startswith('1024 : i32')
    mask_node=defs[mask]
    assert mask_node['op']=='arith.cmpi' and mask_node['text'].startswith('slt,')
    assert 'tensor<1024xi32>' in mask_node['text']
    assert mask_node['vars'][0]==offset_ssa and len(mask_node['vars'])==2
    limit=defs[mask_node['vars'][1]]
    assert limit['op']=='tt.splat' and limit['vars']==['%n_elements']
    assert 'i32 -> tensor<1024xi32>' in limit['text']
    checked.append({'x':x,'y':y,'integer_first':integer_first,'compute':compute,
                    'integer_conversion_ssa':int_cast,'float_operand_ssa':float_value,
                    'ordered_add_operands':ordered,'add_ssa':add,'stored_ssa':result,
                    'mask_ssa':mask,'shared_offset_ssa':offsets[0],
                    'load_other_operand':False,'offset_expression':'signed i32 pid.x*1024+arange(0,1024)',
                    'mask_expression':'signed i32 offsets<n_elements','offset_and_mask_ssa_checked':True,
                    'route_status':'all_ssa_address_mask_load_conversion_order_store_checks_passed',
                    'ttir_path':path.name,'ttir_sha256':sha(path)})
    (output/path.name).write_bytes(path.read_bytes())
assert len(checked)==36
receipt={'count':36,'profiles':checked,'source_sha256':json.loads((audit/'receipt.json').read_text())['source_sha256'],
         'prior_audit_sha256':sha(audit/'receipt.json'),'prior_profiles_sha256':sha(audit/'frontend-probes.json'),
         'new_frontend_runs':0,'native_backend_runs':0,'gpu_runs':0,
         'options':'inherited sm90, debug=false, sanitize_overflow=true',
         'conclusion':'direct int→fp16, fp16 add→extf fp32 before integer store; bf16→fp32 operand promotion, integer→fp32, fp32 add, and bf16 truncf only for bf16 output',
         'limitations':'SSA observations are not formal compiler/source refinement or native poison/trap/debug/address-overflow proof'}
p=output/'receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'count':len(checked),'receipt_sha256':sha(p)}))
