"""Check connected scalar SSA paths in all selected original frontend outputs."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((ROOT/'evidence/frontend/receipt.json').read_text())
var=r'%[A-Za-z0-9_\-]+'
type_names={'fp16':'f16','bf16':'bf16','fp32':'f32'}
checked=[]
for profile in receipt['profiles']:
    path=ROOT/profile['ttir_path'];assert sha(path)==profile['ttir_sha256']
    text=path.read_text();public=text.split('tt.func public ',1)[1].split('tt.func private ',1)[0]
    defs={}
    for i,line in enumerate(public.splitlines()):
        match=re.match(r'\s*('+var+r') = ([A-Za-z0-9_.]+) (.*)',line)
        if match:defs[match[1]]={'op':match[2],'args':re.findall(var,match[3]),'text':match[3],'line':i}
    def one(op,args=None,fragment=None):
        matches=[v for v,d in defs.items() if d['op']==op and (args is None or d['args']==args)
                 and (fragment is None or fragment in d['text'])]
        assert len(matches)==1,(profile,op,args,fragment,matches)
        return matches[0]
    it,ot=type_names[profile['input_dtype']],type_names[profile['output_dtype']]
    load=one('tt.load',fragment=f'tensor<1024x!tt.ptr<{it}>>')
    assert len(defs[load]['args'])==3
    input_ptr,mask,other=defs[load]['args']
    if it=='f32':
        row=load
        assert defs[other]['op']=='arith.constant' and 'dense<0xFF800000>' in defs[other]['text']
    else:
        row=one('arith.extf',[load],f'tensor<1024x{it}> to tensor<1024xf32>')
        assert defs[other]['op']=='arith.truncf' and f'tensor<1024xf32> to tensor<1024x{it}>' in defs[other]['text']
        const=defs[other]['args'][0]
        assert defs[const]['op']=='arith.constant' and 'dense<0xFF800000>' in defs[const]['text']
    maximum=one('tt.call',[row],'standard.max__fp32S1024S')
    max_tile=one('tt.splat',[maximum],'f32 -> tensor<1024xf32>')
    sub=one('arith.subf',[row,max_tile],'tensor<1024xf32>')
    exp=one('math.exp',[sub],'tensor<1024xf32>')
    total=one('tt.call',[exp],'standard.sum__fp32S1024S')
    sum_tile=one('tt.splat',[total],'f32 -> tensor<1024xf32>')
    div=one('arith.divf',[exp,sum_tile],'tensor<1024xf32>')
    stored=div if ot=='f32' else one('arith.truncf',[div],f'tensor<1024xf32> to tensor<1024x{ot}>')
    stores=[line for line in public.splitlines() if line.strip().startswith('tt.store ')]
    assert len(stores)==1
    output_ptr,value,output_mask=re.findall(var,stores[0])
    assert value==stored and f'tensor<1024x!tt.ptr<{ot}>>' in stores[0]
    for a,b in [(row,maximum),(maximum,max_tile),(max_tile,sub),(sub,exp),(exp,total),(total,sum_tile),(sum_tile,div)]:
        assert defs[a]['line']<defs[b]['line']
    m,om=defs[mask],defs[output_mask]
    assert m['op']==om['op']=='arith.cmpi' and m['text'].startswith('slt,') and om['text'].startswith('slt,')
    assert m['args'][0]==om['args'][0] and defs[m['args'][0]]['op']=='tt.make_range'
    assert defs[m['args'][1]]['args']==defs[om['args'][1]]['args']
    assert 'arith.maxnumf ' in text and 'arith.addf ' in text
    # The source's native pointer/index arithmetic is observed, not equated with Nat arithmetic.
    assert defs[input_ptr]['op']==defs[output_ptr]['op']=='tt.addptr'
    checked.append({**{k:profile[k] for k in ['task','input_dtype','output_dtype','ttir_path','ttir_sha256']},
        'data_load':load,'fp32_row':row,'maximum':maximum,'subtraction':sub,'exponential':exp,
        'sum':total,'division':div,'stored_value':stored,'input_padding':'fp32 negative infinity cast to input dtype before load',
        'output_conversion':'identity fp32 alias' if ot=='f32' else 'arith.truncf:f32->'+ot,
        'mask':'arange(0,1024) < loaded sequence length on both load and store',
        'status':'connected_original_frontend_scalar_path_checked'})
result={'schema':1,'count':len(checked),'profiles':checked,'frontend_receipt_sha256':sha(ROOT/'evidence/frontend/receipt.json'),
        'native_execution':False,'compiler_refinement_proved':False,'reduction_schedule_proved':False,
        'boundary':'This checks observed raw frontend SSA for the exact source extraction and fixed configuration. It does not establish native lowering, floating numerical laws, Nat/i32 address correspondence, failure behavior or whole-grid acceptance.'}
(ROOT/'evidence/scalar-graph-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'count':len(checked),'sha256':sha(ROOT/'evidence/scalar-graph-audit.json')}))
