from pathlib import Path
from functools import lru_cache
import re,json,collections
from inspect_ptx import BASE,DOT_LINES

def poly_add(a,b):
    out=dict(a)
    for k,v in b: out[k]=out.get(k,0)+v
    return tuple(sorted((k,v) for k,v in out.items() if v))
def poly_scale(a,b): return tuple((k,v*b) for k,v in a if v*b)
def constant(x): return () if x==0 else (('',x),)
def as_constant(x):
    if not x:return 0
    if len(x)==1 and x[0][0]=='':return x[0][1]
    return None

def split_args(text):
    level=0;out=[];start=0
    for i,c in enumerate(text):
        if c in '([<':level+=1
        if c in ')]>':level-=1
        if c==',' and level==0:out.append(text[start:i].strip());start=i+1
    out.append(text[start:].strip())
    return out

def audit_dot(path,constants=None):
    text=path.read_text(); defs={}; locations={}; lines={}
    for lineno,line in enumerate(text.splitlines(),1):
        m=re.match(r'\s*(%[\w.\-]+) = (.*)',line)
        if m:defs[m[1]]=m[2];lines[m[1]]=lineno
        m=re.match(r'(!\d+) = !DILocation\(line: (\d+)',line)
        if m:locations[m[1]]=int(m[2])
    fmas={};used=set();loads={};callargs={}
    for name,s in defs.items():
        m=re.search(r'@llvm.(?:fmuladd|fma)\.(f32|v(\d+)f32)\((.*?)\)',s)
        if not m:continue
        loc=locations[re.search(r'!dbg (!\d+)',s)[1]]
        if loc not in DOT_LINES:continue
        width=int(m[2]) if m[2] else 1
        args=[re.sub(r'^(float|<\d+ x float>) ','',x) for x in split_args(m[3])]
        callargs[name]=(loc,width,args)
    @lru_cache(None)
    def element(name,index=None):
        s=defs.get(name,'')
        key=name if index is None else name+'#'+str(index)
        if name in callargs:return key
        m=re.match(r'load (float|<\d+ x float>), ptr addrspace\(3\) ([^,]+),',s)
        if m:
            loads[key]=(m[2],4*(index or 0));return key
        m=re.match(r'extractelement <\d+ x float> ([^,]+), i\d+ (\d+)',s)
        if m:return element(m[1],int(m[2]))
        if index is not None:
            m=re.match(r'insertelement <\d+ x float> ([^,]+), float ([^,]+), i\d+ (\d+)',s)
            if m:return element(m[2]) if index==int(m[3]) else element(m[1],index)
            m=re.match(r'shufflevector <(\d+) x float> ([^,]+), <\d+ x float> ([^,]+), <\d+ x i32> (.*?)(?:, !dbg|$)',s)
            if m:
                width,a,b,mask=m.groups();width=int(width)
                entries=['0']*10000 if mask=='zeroinitializer' else re.findall(r'i32 (\d+|poison)',mask)
                selected=entries[index]
                assert selected!='poison',(path,name,index,'selected poison shuffle')
                selected=int(selected)
                return element(a,selected) if selected<width else element(b,selected-width)
        return key
    for name,(loc,width,args) in callargs.items():
        for index in ([None] if width==1 and '.f32(' in defs[name] else range(width)):
            key=element(name,index)
            a,b,c=[element(arg,index) for arg in args]
            assert a in loads and b in loads,(path,name,index,'unresolved dot operand',a,b)
            fmas[key]=(loc,a,b,c);lines[key]=lines[name]
    for name,(loc,a,b,c) in fmas.items():
        if c in fmas and fmas[c][0]==loc:used.add(c)
    roots=[r for r in fmas if r not in used]
    @lru_cache(None)
    def chain(name):
        loc,a,b,c=fmas[name]
        if c in fmas and fmas[c][0]==loc:
            seq,seed=chain(c);return seq+((a,b),),seed
        return ((a,b),),c
    active=set()
    @lru_cache(None)
    def value(name,tid):
        key=(name,tid)
        if key in active:return ((name,1),)
        active.add(key)
        try:return value_impl(name,tid)
        finally:active.remove(key)
    def value_impl(name,tid):
        if re.fullmatch(r'-?\d+',name):return constant(int(name))
        s=name if name.startswith('getelementptr') else defs.get(name,'')
        if 'read.ptx.sreg.tid.x' in s:return constant(tid)
        if s.startswith('phi i'):
            choices=[value(x,tid) for x in re.findall(r'\[ ([^,]+), [^\]]+\]',s)]
            if choices and all(x==choices[0] for x in choices):return choices[0]
        if s.startswith('getelementptr '):
            inner=re.sub(r'^getelementptr(?: inbounds)?(?: nuw)?\s+','',s)
            if inner.startswith('('):inner=inner[1:-1]
            parts=split_args(inner)
            ty=parts[0];base=re.sub(r'^ptr(?: addrspace\(\d+\))? ','',parts[1])
            idx=[re.fullmatch(r'i(?:32|64) ([^, ]+)',x) for x in parts[2:] if not x.startswith('!')]
            if ty in ['i8','float','i32','i64'] and len(idx)==1 and idx[0]:
                width={'i8':1,'float':4,'i32':4,'i64':8}[ty]
                return poly_add(value(base,tid),poly_scale(value(idx[0][1],tid),width))
        m=re.match(r'(?:sext|zext|trunc|ptrtoint|inttoptr)(?: nneg)? (?:i\d+|ptr(?: addrspace\(\d+\))?) ([^ ]+) to',s)
        if m:return value(m[1],tid)
        m=re.match(r'(add|sub|mul|shl|lshr|ashr|and|or|xor)(?: (?:nuw|nsw|exact|disjoint))* i\d+ ([^,]+), ([^, ]+)',s)
        if m:
            op,a,b=m.groups();a=value(a,tid);b=value(b,tid);ac=as_constant(a);bc=as_constant(b)
            if op=='add' or (op=='or' and 'disjoint' in s):return poly_add(a,b)
            if op=='sub':return poly_add(a,poly_scale(b,-1))
            if op=='mul' and bc is not None:return poly_scale(a,bc)
            if op=='mul' and ac is not None:return poly_scale(b,ac)
            if op=='shl' and bc is not None:return poly_scale(a,2**bc)
            if ac is not None and bc is not None:
                return constant({'lshr':lambda:ac>>bc,'ashr':lambda:ac>>bc,'and':lambda:ac&bc,
                                 'or':lambda:ac|bc,'xor':lambda:ac^bc}[op]())
        return ((name,1),)
    def address(v,tid):
        base,offset=loads[v]
        return poly_add(value(base,tid),constant(offset))
    results=[];failures=[]
    for root in roots:
        loc=fmas[root][0];seq,seed=chain(root)
        # The typed shared-load address strides must follow reduction index
        # 0,1,... in BOTH operands for every physical thread in the CTA.
        strides=[]
        for side in [0,1]:
            observed=set()
            for tid in range(128):
                ps=[address(pair[side],tid) for pair in seq]
                ds=[as_constant(poly_add(ps[i],poly_scale(ps[0],-1))) for i in range(len(ps))]
                stride=ds[1] if len(ds)>1 else 0
                if stride is None or any(d!=i*stride for i,d in enumerate(ds)) or stride<=0:
                    failures.append({'root':root,'source_line':loc,'side':side,'thread':tid,'length':len(seq),
                                     'deltas':ds[:16],'addresses':ps[:4]})
                    break
                observed.add(stride)
            strides.append(sorted(observed))
        if constants:
            expected=constants['BK'] if loc in [55,81,154,181,229,258] else constants['BV'] if loc in [149,175,236,263] else 32
            assert len(seq)==expected,(path,root,loc,len(seq),expected)
        results.append({'source_line':loc,'llvm_root_line':lines[root],
                        'llvm_root':root,'length':len(seq),'seed':seed,'strides':strides})
    return {'roots':results,'failures':failures,'thread_count':128,
            'scalar_fma_instruction_count':len(fmas)}

if __name__=='__main__':
    import sys
    for path in sorted(BASE.glob('actual-binder-*/kernel.llir')):
        if len(sys.argv)>1 and sys.argv[1] not in path.parent.name:continue
        constants=json.loads((path.parent/'receipt.json').read_text())['constants']
        a=audit_dot(path,constants)
        output=Path(__file__).parent/'routing';output.mkdir(exist_ok=True)
        (output/(path.parent.name+'.json')).write_text(json.dumps(a,indent=2)+'\n')
        print(path.parent.name,'roots',len(a['roots']),'failures',len(a['failures']))
        if a['failures']:print(json.dumps(a['failures'][:3]))
