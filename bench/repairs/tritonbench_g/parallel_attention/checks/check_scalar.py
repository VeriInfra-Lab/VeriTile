from inspect_ptx import scan,BASE
from pathlib import Path
import json

def audit_scalar(path,use_scale,post_seed,scaled_sum_on_z):
    es=scan(path);checks=[]
    scale=None
    if use_scale:
        scales=[e for e in es if e['loc']==134 and e['op']=='mul.f32']
        assert scales
        scale=scales[0]['node'][1]
        assert scale[0]=='input'
        for e in scales:assert e['node'][1]==scale
        checks.append({'site':'query-scale','form':'left','count':len(scales),'ptx_first':scales[0]['line']})
    else:
        assert not any(e['loc'] in [134,160] and e['op']=='mul.f32' for e in es)
    def abstract(n):
        if n==scale:return 'S'
        if n[0]=='dot':return 'D'+str(n[1])
        if n[0]=='input':
            if n[1] in ['0','0f00000000']:return '0'
            if n[1].startswith('%p'):return 'P'
            return 'X'
        if n[0] in ['selp.b32','selp.f32']:
            a,b=abstract(n[1]),abstract(n[2])
            if a=='0':return ('mask',b)
            if b=='0':return ('mask',a)
            return ('shard-select',a,b)
        return (n[0],*[abstract(x) for x in n[1:]])
    add=lambda a,b:('add.f32',a,b)
    mul=lambda a,b:('mul.f32',a,b)
    fma=lambda a,b,c:('fma.rn.f32',a,b,c)
    mask=lambda a:('mask',a)
    def ds(line,left=False):
        d='D'+str(line)
        if post_seed:return ('shard-select',add('X',d) if left else add(d,'X'),d)
        return d
    dqp=ds(149);dqd=ds(175);dkd=ds(263,True)
    if use_scale:
        kdf=fma('S','X',mul('S','D236')) if scaled_sum_on_z else fma('S','D236',mul('S','X'))
    else:kdf=ds(236,True)
    def doubled(d):return add(d,d)
    def scale_twice(d):return fma('S',d,mul('S',d))
    expected={
        156:mul(doubled(dqp),'D154'),
        184:mul(scale_twice(mask(dqd)),mask('D181')) if use_scale else mask(mul(doubled(dqd),'D181')),
        242:mul(mask(mul('S','D229') if use_scale else 'D229'),mask(doubled(kdf))),
        271:mul(mask(mul('S','D258')),scale_twice(mask(dkd))) if use_scale else mask(mul('D258',doubled(dkd)))
    }
    for loc,target in expected.items():
        # In the no-scale diagonal the final select is part of the coefficient.
        op='selp.b32' if (not use_scale and loc in [184,271]) else 'mul.f32'
        selected=[e for e in es if e['loc']==loc and e['op']==op]
        if loc!=242:assert selected,(path,loc,'missing coefficient')
        for e in selected:
            actual=abstract(e['node'])
            assert actual==target,(path,e['line'],target,actual)
        checks.append({'site':loc,'count':len(selected),'form':target,'ptx_lines':[e['line'] for e in selected]})
    # All surviving scale sites have scale on the left.
    for loc in [160,230,258]:
        selected=[e for e in es if e['loc']==loc and e['op']=='mul.f32']
        if not use_scale:assert not selected
        for e in selected:assert e['node'][1]==scale,(path,e['line'],'scale operand order')
    # A T=1 specialization hoists the entire coefficient for constant padded
    # query columns. Its product has the opposite operand order from the
    # dynamically masked DK coefficient, and remains explicit syntax.
    padded=[e for e in es if e['loc']==254 and e['op']=='fma.rn.f32']
    for e in padded:
        target=fma('S','0',mul('S','0'))
        assert abstract(e['node'])==target
        consumers=[x for x in es if x['loc']==254 and x['op']=='mul.f32' and e['node'] in x['node'][1:]]
        assert len(consumers)==1 and abstract(consumers[0]['node'])==mul(target,'0')
    assert len(padded)<=1
    checks.append({'site':'constant-padded-DK','count':len(padded),
                   'ptx_lines':[e['line'] for e in padded],'form':'direct scale-twice-zero product'})
    return checks

if __name__=='__main__':
    for path in sorted(BASE.glob('actual-binder-*-backward/kernel.ptx')):
        name=path.parent.name
        rows=audit_scalar(path,'noscale' not in name,'sharded' in name,'sharded' in name)
        print(name,[(r['site'],r['count']) for r in rows])
