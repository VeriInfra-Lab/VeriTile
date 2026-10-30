"""Actual source AST replay: exceptional-prefix repair and unchanged finite controls.
CPU FP32 operation replay; native instruction conformance is a separate model boundary.
"""
from pathlib import Path
import importlib.util, json, hashlib, sys
sys.dont_write_bytecode = True
import torch
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / '.veritile-cache/kernel-evidence/softmax_reducev_neginf_checked'
SOURCE = HERE / 'softmax_reducev_neginf_checked.py'
BASELINE = HERE.parent / 'softmax_reducev/softmax_reducev_repaired.py'
spec = importlib.util.spec_from_file_location('saved_replay', ROOT / '.veritile-cache/kernel-evidence/softmax_reducev/reproduce_source_defect.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

def invoke(path, logits, *, dim=1, offsets=False, singleton=False, values=None, sentinel=0):
    count=len(logits)
    q=torch.tensor([logits],dtype=torch.float32)
    v=torch.ones(count,1,dim,dtype=torch.float32) if values is None else values.clone()
    out=torch.full((1,1,dim),99.,dtype=torch.float32)
    pages=torch.arange(count,dtype=torch.int32).reshape(1,count)
    starts=torch.tensor([0],dtype=torch.int32); lengths=torch.tensor([count],dtype=torch.int32)
    data=[q,v,out,pages,starts,lengths]
    if offsets:data=[r.offset_view(t) for t in data]
    q,v,out,pages,starts,lengths=data
    if singleton:
        q=q.as_strided(q.shape,(count+97,1));out=out.as_strided(out.shape,(dim+211,dim+223,1))
        pages=pages.as_strided(pages.shape,(count+101,1));starts=starts.as_strided(starts.shape,(233,));lengths=lengths.as_strided(lengths.shape,(239,))
        data=[q,v,out,pages,starts,lengths]
    before=[bytes(t.untyped_storage()) for t in data]
    r.SOURCE=path
    wrapper,kernel=r.harness(path)
    assert wrapper(*map(r.Tensor,data),count,sentinel) is None
    assert len(kernel.calls)==1
    for index,t in enumerate(data):
        after=bytes(t.untyped_storage())
        if index==2:
            begin=4*t.storage_offset();end=begin+4*t.numel()
            assert after[:begin]==before[index][:begin] and after[end:]==before[index][end:]
        else:assert after==before[index]
    return out.detach().clone(), kernel.calls[0]

def main():
    report=[]
    floor=-340282346638528859811704183484516925440.0
    assert torch.tensor(floor,dtype=torch.float32).view(torch.int32).item() & 0xffffffff == 0xff7fffff
    for prefix,tail,offsets,singleton in [(64,0.,False,False),(64,-1000.,False,False),(64,floor,False,False),(128,0.,False,False),(192,-1000.,True,True)]:
        q=[-float('inf')]*prefix+[tail]
        old,_=invoke(BASELINE,q,offsets=offsets,singleton=singleton)
        new,launch=invoke(SOURCE,q,offsets=offsets,singleton=singleton)
        assert torch.isnan(old).all() and torch.equal(new,torch.ones_like(new))
        assert torch.equal(torch.softmax(torch.tensor(q,dtype=torch.float32),dim=0).sum(),torch.tensor(1.))
        report.append(dict(case='negative_infinity_prefix',prefix=prefix,finite_tail=str(tail),offsets=offsets,singleton=singleton,old='NaN',repaired='0x3f800000',launch=launch))
    generator=torch.Generator().manual_seed(24680)
    for count,dim in [(1,1),(63,2),(64,8),(65,16),(129,32),(257,128)]:
        q=(torch.randn(count,generator=generator)*15).tolist()
        v=torch.randn(count,1,dim,generator=generator)
        a,_=invoke(BASELINE,q,dim=dim,values=v,offsets=True)
        b,_=invoke(SOURCE,q,dim=dim,values=v,offsets=True)
        assert torch.equal(a.view(torch.int32),b.view(torch.int32))
        report.append(dict(case='finite_bitwise_control',count=count,dimension=dim,CPU_FP32_equal=True))
    for label,q in [('all_negative_infinity',[-float('inf')]*65),('positive_infinity',[0.]*64+[float('inf')]),('NaN',[0.]*64+[float('nan')])]:
        value,_=invoke(SOURCE,q)
        assert torch.isnan(value).all()
        report.append(dict(case=label,result='NaN',normalized_Real_scope=False))
    q=[-float('inf')]*64+[0.]
    v=torch.ones(65,1,1);v[0]=float('inf')
    value,_=invoke(SOURCE,q,values=v)
    assert torch.isnan(value).all()
    report.append(dict(case='zero_weight_active_infinite_V',result='NaN',reason='IEEE zero times infinity is NaN'))
    OUT.mkdir(exist_ok=True)
    result=dict(status='PASS',cases=report,cases_count=len(report),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),GPU_execution=False,scope='Actual admitted wrapper and kernel AST; strict active bounds and full backing-storage frames; CPU FP32 exp/reductions, not GPU instruction conformance.')
    (OUT/'neginf-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(report),source_sha256=result['source_sha256'])))
if __name__=='__main__':main()
