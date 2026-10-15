"""Observe all29 frozen f64 SSA data/address routes; no new compilation."""
import hashlib,json,re
from pathlib import Path
root=Path(__file__).resolve().parent
prior=root.parent/'vector-addition-dtype-extension-wip'
audit=prior/'evidence/extension-audit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(prior/'evidence/receipt.json')=='85dc08106852c4a3f69516046182954693eb5afd0158064db358c708a70b9100'
rows=json.loads((audit/'frontend-probes.json').read_text())
selected=[r for r in rows if r['status']=='frontend_generated' and 'float64' in (r['x'],r['y'])]
assert len(selected)==29
ints={'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
floats={'float16':'f16','bfloat16':'bf16','float32':'f32','float64':'f64','float8_e4m3fn':'f8E4M3FN','float8_e5m2':'f8E5M2'}
output=root/'evidence/source-routes';output.mkdir(exist_ok=True)
var=r'%[A-Za-z0-9_]+'
checked=[]
for row in selected:
 x,y=row['x'],row['y'];path=audit/row['ttir_path'];assert sha(path)==row['ttir_sha256']
 lines=path.read_text().splitlines();defs={}
 for i,line in enumerate(lines):
  m=re.match(r'\s*('+var+r') = ([A-Za-z0-9_.]+) (.*)',line)
  if m:defs[m[1]]={'op':m[2],'text':m[3],'vars':re.findall(var,m[3]),'position':i+1,'line':line.strip()}
 def unique(pred):
  matched=[v for v,d in defs.items() if pred(v,d)]
  assert len(matched)==1,(x,y,matched)
  return matched[0]
 def next_op(value,op,src,dst,rtne=False):
  new=unique(lambda v,d:d['op']==op and d['vars']==[value])
  text=defs[new]['text'];sep=' -> ' if op=='tt.fp_to_fp' else ' to '
  assert f'tensor<1024x{src}>'+sep+f'tensor<1024x{dst}>' in text,(x,y,text)
  assert defs[value]['position']<defs[new]['position']
  if rtne:assert ', rounding = rtne :' in text
  return new
 loads={};masks=[];input_routes={};operands=[]
 for side,dtype in [('x',x),('y',y)]:
  load=unique(lambda v,d:v.startswith('%'+side+'_') and d['op']=='tt.load')
  loads[side]=load;node=defs[load]
  pointer_dtype='i8' if dtype=='bool' else ('i'+re.search(r'\d+',dtype).group() if dtype in ints else floats[dtype])
  assert f'tt.ptr<{pointer_dtype}>' in node['text'] and len(node['vars'])==2
  masks.append(node['vars'][1]);v=load;route=[]
  if dtype in ints:
   width=1 if dtype=='bool' else int(re.search(r'\d+',dtype).group())
   if dtype=='bool':
    normalized=unique(lambda z,d:d['op']=='arith.cmpi' and d['text'].startswith('ne,') and d['vars'][0]==load)
    assert 'dense<0>' in defs[defs[normalized]['vars'][1]]['text']
    assert 'tensor<1024xi8>' in defs[normalized]['text']
    assert defs[load]['position']<defs[normalized]['position']
    v=normalized;route.append('boolLoad:i8!=0->i1')
   op='arith.uitofp' if dtype=='bool' or dtype.startswith('uint') else 'arith.sitofp'
   v=next_op(v,op,f'i{width}','f64');route.append(f'{op}:i{width}->f64')
  elif dtype in {'float16','bfloat16'}:
   v=next_op(v,'arith.extf',floats[dtype],'f32');route.append('arith.extf:'+floats[dtype]+'->f32')
   v=next_op(v,'arith.extf','f32','f64');route.append('arith.extf:f32->f64')
  elif dtype=='float32':
   v=next_op(v,'arith.extf','f32','f64');route.append('arith.extf:f32->f64')
  elif dtype.startswith('float8_'):
   v=next_op(v,'tt.fp_to_fp',floats[dtype],'f64');route.append('tt.fp_to_fp:'+floats[dtype]+'->f64')
  else:assert dtype=='float64'
  operands.append(v);input_routes[side]=route
 assert defs[loads['x']]['position']<defs[loads['y']]['position']
 assert masks[0]==masks[1];mask=masks[0]
 add=unique(lambda v,d:d['op']=='arith.addf' and d['vars']==operands)
 assert 'tensor<1024xf64>' in defs[add]['text']
 assert len([d for d in defs.values() if d['op']=='arith.addf'])==1
 assert all(defs[v]['position']<defs[add]['position'] for v in operands)
 result=add;output_route=[]
 if x in ints:
  width=8 if x=='bool' else int(re.search(r'\d+',x).group());target='i'+str(width)
  op='arith.fptoui' if x.startswith('uint') else 'arith.fptosi'
  result=next_op(result,op,'f64',target);output_route.append(op+':f64->'+target)
  if x=='bool':output_route.append('boolStore:preserve-i8-byte')
 else:
  target=floats[x]
  if x!='float64':
   op='tt.fp_to_fp' if x.startswith('float8_') else 'arith.truncf'
   result=next_op(result,op,'f64',target,rtne=x.startswith('float8_'))
   output_route.append(op+':f64->'+target+(':rtne' if x.startswith('float8_') else ''))
 stores=[(i+1,l.strip()) for i,l in enumerate(lines) if l.strip().startswith('tt.store ')]
 assert len(stores)==1
 position,store=stores[0];ptr,value,store_mask=re.findall(var,store)
 assert value==result and store_mask==mask and f'tt.ptr<{target}>' in store
 assert defs[result]['position']<position
 offsets=[]
 for pointer,side in [(defs[loads['x']]['vars'][0],'x'),(defs[loads['y']]['vars'][0],'y'),(ptr,'output')]:
  node=defs[pointer]
  if node['op']=='tt.bitcast':node=defs[node['vars'][0]]
  assert node['op']=='tt.addptr';base,offset=node['vars'];offsets.append(offset)
  assert defs[base]['op']=='tt.splat' and defs[base]['vars']==['%'+side+'_ptr']
 assert len(set(offsets))==1
 offset=offsets[0];node=defs[offset]
 assert node['op']=='arith.addi' and 'tensor<1024xi32>' in node['text']
 block_splat,lanes=node['vars'];assert defs[lanes]['op']=='tt.make_range'
 assert 'end = 1024 : i32, start = 0 : i32' in defs[lanes]['text']
 assert defs[block_splat]['op']=='tt.splat' and 'i32 -> tensor<1024xi32>' in defs[block_splat]['text']
 block=defs[block_splat]['vars'][0];assert defs[block]['op']=='arith.muli' and ': i32' in defs[block]['text']
 pid,block_size=defs[block]['vars']
 assert defs[pid]['op']=='tt.get_program_id' and defs[pid]['text'].startswith('x : i32')
 assert defs[block_size]['op']=='arith.constant' and defs[block_size]['text'].startswith('1024 : i32')
 node=defs[mask];assert node['op']=='arith.cmpi' and node['text'].startswith('slt,')
 assert 'tensor<1024xi32>' in node['text'] and node['vars'][0]==offset
 assert defs[node['vars'][1]]['op']=='tt.splat' and defs[node['vars'][1]]['vars']==['%n_elements']
 checked.append({'x':x,'y':y,'input_routes':input_routes,'output_route':output_route,'loads':loads,'ordered_add_operands':operands,'add_ssa':add,'stored_ssa':result,'mask_ssa':mask,'offset_ssa':offset,'load_other_operand':False,'offset_expression':'signed i32 pid.x*1024+arange(0,1024)','mask_expression':'signed i32 offsets<n_elements','ttir_path':path.name,'ttir_sha256':sha(path),'status':'checked_complete_load_conversion_add_store_and_address_graph'})
 (output/path.name).write_bytes(path.read_bytes())
receipt={'count':29,'profiles':checked,'source_sha256':json.loads((audit/'receipt.json').read_text())['source_sha256'],'prior_audit_sha256':sha(audit/'receipt.json'),'prior_frontend_profiles_sha256':sha(audit/'frontend-probes.json'),'new_frontend_compilations':0,'native_backend_compilations':0,'gpu_executions':0,'options':'inherited sm90, debug=false, sanitize_overflow=true','limitations':['Observed SSA graphs are not a formal original-source/compiler refinement theorem.','Address expressions are observed but the typed scalar core does not model them.','Graph order and no-other loads do not prove native poison/trap/sanitizer behavior.','Bool store reinterpretation is logical byte typing, not a new native arithmetic instruction.']}
p=output/'receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'count':29,'receipt_sha256':sha(p)}))
