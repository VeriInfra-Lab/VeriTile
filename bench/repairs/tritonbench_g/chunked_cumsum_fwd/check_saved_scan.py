#!/usr/bin/env python3
"""Audit the saved SM80 scan instruction slices; does not compile or run CUDA.

All 128 lanes are symbolically replayed. Address/control integer instructions
are evaluated on a full logical tile to identify every output coordinate.
Prepared words and rate loads are distinct symbols. Only actual mul/add/FMA,
shuffle, selection and shared-memory instructions build the ordered scan DAG.
This is compiler-artifact evidence, not an alternate kernel interpreter/proof.
"""
from pathlib import Path
import argparse, hashlib, json, re, tempfile

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'bench/tritonbench_g/chunked_cumsum_fwd/Proofs'
TASK=Path(__file__).resolve().parent
EVIDENCE=ROOT/'.veritile-cache/kernel-evidence/chunked_cumsum_fwd'
REPORT=ROOT/'.veritile-cache/task-deliveries/chunked_cumsum_fwd/lowering/report.json'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def bits(x,n):
    if isinstance(x,int): return tuple(bool((x>>i)&1) for i in range(n))
    assert x[0]=='bits'; return (x[1]+(False,)*n)[:n]
def bor(a,b): return b if a is False else a if b is False or a==b else True if a is True or b is True else ('or',a,b)
def band(a,b): return b if a is True else a if b is True or a==b else False if a is False or b is False else ('and',a,b)
def leaves(t):
    if t[0]=='leaf': return [t[1]]
    if t[0]=='add': return leaves(t[1])+leaves(t[2])
    if t[0]=='fused': return [t[1]]+leaves(t[2])
    raise AssertionError(t)

def historical_rows():
    rows={}
    for i in (0,1,3,4):
        p=OLD/f'WholeNativePlan{i}.lean'; s=p.read_text(); nodes={}
        for line in s.splitlines():
            m=re.match(r'def (node\d+) : SumTree \d+ := \.(leaf|add|fused) (.*)',line)
            if not m: continue
            n,kind,args=m.groups()
            if kind=='leaf': val=(kind,int(re.search(r'⟨(\d+)',args)[1]))
            elif kind=='fused': val=(kind,int(re.search(r'⟨(\d+)',args)[1]),nodes[args.split()[-1]])
            else:
                a,b=args.split(); val=(kind,nodes[a],nodes[b])
            nodes[n]=val
        names=re.search(r'def row : .* := !\[(.*?)\]',s)[1].split(',')
        row=tuple(nodes[n] for n in names)
        assert all(sorted(leaves(t))==list(range(j+1)) for j,t in enumerate(row))
        rows[i]=(p,row)
    return rows

class Audit:
    def __init__(self,profile):
        self.profile=profile; self.h=profile['block_heads']; self.c=profile['block_chunk']
        self.reg=[{'%tid.x':tid,'%ctaid.x':0,'%ctaid.y':0,'%ctaid.z':0} for tid in range(128)]
        self.provenance=[{name:frozenset({('special',name)}) for name in
            ['%tid.x','%ctaid.x','%ctaid.y','%ctaid.z']} for _ in range(128)]
        self.params=[1000000,2000000,3000000,4000000,5000000,1,self.c,self.h,self.c,
            ('lower',),('upper',),self.c*self.h,self.h,1,1,1,self.h*self.c,self.c,self.c,1,
            self.h*self.c,self.c,self.c,1,0,0]
        self.shared={}; self.outputs={}; self.prepared={}; self.bias_orders=set(); self.scan_ops=[]
        self.exp_arguments=set()
        self.scan_started=False; self.loc=(0,0,0); self.instructions=[]
    def get(self,tid,x):
        x=x.strip().strip('{}').strip()
        if x.startswith('['):
            x=x[1:-1].strip(); bits=re.split(r'\s*\+\s*',x)
            return sum(self.get(tid,a) for a in bits)
        if x=='global_smem': return 0
        if x.startswith('%'): return self.reg[tid].get(x)
        if x.startswith('0f'): return ('word',x[2:].lower())
        return int(x)
    def set(self,tid,d,v): self.reg[tid][d]=v
    def deps(self,tid,args):
        names=re.findall(r'%[A-Za-z_][A-Za-z_0-9.]*',','.join(args))
        return frozenset().union(*(self.provenance[tid].get(name,frozenset()) for name in names))
    def execute(self,path):
        source=path.read_text().split('\tret;')[0]
        for number,line in enumerate(source.splitlines(),1):
            line=line.split('//')[0].strip()
            loc=re.match(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)',line)
            if loc: self.loc=tuple(map(int,loc.groups())); continue
            m=re.fullmatch(r'(?:(@!?%p\d+)\s+)?([a-z][a-z0-9_.]*)\s+(.+);',line)
            if not m: continue
            pred,op,args=m.groups(); args=[a.strip() for a in args.split(',')]
            self.instructions.append((number,self.loc,pred,op,args))
            if self.loc==(1,69,14): self.scan_started=True
            if op.startswith('shfl.'):
                assert op=='shfl.sync.up.b32'
                d,x,delta,control,member=args
                assert control=='0' and member=='-1'
                old=[self.get(t,x) for t in range(128)]; step=int(delta)
                for tid in range(128): self.set(tid,d,old[tid-step] if tid%32>=step else old[tid])
                self.scan_ops.append((number,op)); continue
            if op in ('bar.sync','bra'): continue
            for tid in range(128):
                r=self.reg[tid]
                enabled=True if pred is None else self.get(tid,pred.lstrip('@!'))
                if pred and pred.startswith('@!') and enabled is not None: enabled=not enabled
                if op=='st.global.b32':
                    assert isinstance(enabled,bool),(path,number,tid,'unknown store predicate')
                    addr=self.get(tid,args[0]); src=args[1].strip('{} '); val=self.get(tid,src)
                    if self.loc[0:2]==(1,67):
                        off=(addr-self.params[3])//4; h,j=divmod(off,self.c)
                        assert 0<=h<self.h and 0<=j<self.c
                        if enabled:
                            assert (h,j) not in self.prepared,(path,h,j,'duplicate prepared output')
                            assert val and val[0]=='minimum' and val[2]==('upper',), (path,number,tid,val)
                            assert val[1][0]=='maximum' and val[1][2]==('lower',)
                            core=val[1][1]
                            if self.profile['softplus']:
                                assert core[0]=='softplus'; core=core[1]
                            if self.profile['bias']:
                                assert core[0]=='biasAdd'
                                assert sorted(core[1:])==sorted((('dt',h,j),('bias',h)))
                            else: assert core==('dt',h,j)
                            self.prepared[h,j]=val
                        # Replicated lanes still carry the same word into scan shuffles.
                        self.set(tid,src,('prepared',h,j))
                    elif self.loc[0:2]==(1,71):
                        if enabled:
                            off=(addr-self.params[4])//4; h,j=divmod(off,self.c)
                            assert (h,j) not in self.outputs, (path,h,j,'duplicate output')
                            self.outputs[h,j]=val
                    else: raise AssertionError((path,number,'unclassified global store'))
                    continue
                if enabled is False: continue
                if op=='st.shared.b32':
                    assert enabled is True
                    addr=self.get(tid,args[0]); val=self.get(tid,args[1])
                    if addr in self.shared: assert self.shared[addr]==val
                    self.shared[addr]=val; continue
                d=args[0].strip('{} '); xs=[self.get(tid,a) for a in args[1:]] if not op.startswith('ld.param') else []
                deps=self.deps(tid,args[1:])
                value=None
                if op.startswith('ld.param'):
                    p=int(re.search(r'param_(\d+)',args[1])[1]); value=self.params[p]
                    deps=frozenset({('param',p)})
                elif op=='ld.global.b32':
                    addr=self.get(tid,args[1])
                    if self.loc[0:2]==(1,68): value=('rate',(addr-self.params[1])//4)
                    elif self.loc[0:2]==(1,57):
                        off=(addr-self.params[0])//4; j,h=divmod(off,self.h); value=('dt',h,j)
                    elif self.loc[0:2]==(1,59): value=('bias',(addr-self.params[2])//4)
                    else: raise AssertionError((path,number,'unclassified global load'))
                    deps=frozenset({value})
                elif op=='ld.shared.b32': value=self.shared[self.get(tid,args[1])]
                elif op.startswith('mov.') or (op.startswith('cvt.') and '.f32' not in op): value=xs[0]
                elif op in ('and.pred','or.pred'):
                    value=None if None in xs else (all(xs) if op=='and.pred' else any(xs))
                elif op.startswith('setp.'):
                    cmp,kind=op.split('.')[1:3]
                    if all(isinstance(x,int) for x in xs):
                        a,b=xs
                        if kind in ('b32','u32'): a%=2**32; b%=2**32
                        if kind=='u64': a%=2**64; b%=2**64
                        value={'eq':a==b,'ne':a!=b,'lt':a<b,'le':a<=b,'gt':a>b,'ge':a>=b}[cmp]
                    elif op=='setp.le.f32' and xs[1]==('word','41a00000'):
                        value=('le20',xs[0])
                    elif op=='setp.ne.b32' and isinstance(xs[0],tuple) and xs[0][0]=='bits' and xs[1]==0:
                        value=False
                        for bit in bits(xs[0],32): value=bor(value,bit)
                elif op.startswith('selp.'):
                    value=xs[0] if xs[2] is True else xs[1] if xs[2] is False else None
                    if op in ('selp.b16','selp.b32') and isinstance(xs[2],tuple):
                        n=int(op[-2:]); a,b,p=xs
                        assert b==0
                        value=('bits',tuple(p if bit else False for bit in bits(a,n)))
                    if self.loc==(1,62,58):
                        assert xs[2]==('le20',xs[1]),(path,number,tid,xs)
                        assert xs[1] in self.exp_arguments
                        # The saved log result depends on exactly one exp(x)+1
                        # word. Its scalar lowering (including FTZ sites) is a
                        # pinned unary compiler boundary, with no shape/scan data.
                        assert self.deps(tid,[args[1]])==frozenset({('logInput',xs[1])})
                        value=('softplus',xs[1])
                elif op in ('max.f32','min.f32'): value=('maximum' if op=='max.f32' else 'minimum',*xs)
                elif op.endswith('.f32'):
                    if self.scan_started:
                        assert op in ('mul.f32','add.f32','fma.rn.f32'), (path,number,op)
                        if op=='mul.f32':
                            p,a=xs; assert p[0]=='prepared' and a==('rate',p[1]),(path,number,xs)
                            value=('leaf',p[2])
                        elif op=='fma.rn.f32':
                            p,a,z=xs; assert p[0]=='prepared' and a==('rate',p[1])
                            value=('fused',p[2],z)
                        else: value=('add',*xs)
                    elif op=='add.f32' and all(isinstance(x,tuple) for x in xs) and {x[0] for x in xs}=={'dt','bias'}:
                        a,b=xs; assert {a[0],b[0]}=={'dt','bias'}
                        self.bias_orders.add(a[0]=='bias'); value=('biasAdd',*xs)
                    elif op=='mul.f32' and xs[1]==('word','3fb8aa3b'):
                        value=('expScale',xs[0])
                    elif op=='ex2.approx.f32':
                        assert xs[0][0]=='expScale'; value=('exp2',xs[0][1])
                    elif op=='add.f32' and isinstance(xs[0],tuple) and xs[0][0]=='exp2':
                        assert xs[1]==('word','3f800000'); self.exp_arguments.add(xs[0][1])
                        deps=frozenset({('logInput',xs[0][1])})
                elif any(isinstance(x,tuple) and x[0]=='bits' for x in xs):
                    n=int(op[-2:]); a=bits(xs[0],n)
                    if op.startswith('shl.'): value=('bits',((False,)*xs[1]+a)[:n])
                    elif op.startswith('shr.'): value=('bits',(a[xs[1]:]+(False,)*n)[:n])
                    elif op.startswith('or.'): value=('bits',tuple(bor(x,y) for x,y in zip(a,bits(xs[1],n))))
                    elif op.startswith('and.'): value=('bits',tuple(band(x,y) for x,y in zip(a,bits(xs[1],n))))
                    else: raise AssertionError((path,number,'unknown predicate packing',op,args))
                elif all(isinstance(x,int) for x in xs):
                    if op.startswith('add.'): value=xs[0]+xs[1]
                    elif op.startswith('sub.'): value=xs[0]-xs[1]
                    elif op.startswith('mul.'): value=xs[0]*xs[1]
                    elif op.startswith('mad.'): value=xs[0]*xs[1]+xs[2]
                    elif op.startswith('shl.'): value=xs[0]<<xs[1]
                    elif op.startswith('shr.'): value=xs[0]>>xs[1]
                    elif op.startswith('and.'): value=xs[0]&xs[1]
                    elif op.startswith('or.'): value=xs[0]|xs[1]
                    elif op.startswith('xor.'): value=xs[0]^xs[1]
                    elif op.startswith('bfe.'): value=(xs[0]>>xs[1])&((1<<xs[2])-1)
                    elif op.startswith('min.'): value=min(xs)
                    else: raise AssertionError((path,number,op,args))
                    value%=2**(64 if op.endswith('64') else 32)
                self.set(tid,d,value)
                self.provenance[tid][d]=deps
            if self.scan_started: self.scan_ops.append((number,op))
        assert set(self.outputs)==set(self.prepared)=={(h,j) for h in range(self.h) for j in range(self.c)}
        rows=tuple(tuple(self.outputs[h,j] for j in range(self.c)) for h in range(self.h))
        for h,row in enumerate(rows):
            for j,t in enumerate(row): assert sorted(leaves(t))==list(range(j+1)), (path,h,j)
        assert len(self.bias_orders)==int(self.profile['bias'])
        return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=EVIDENCE/'saved-scan.json'); args=ap.parse_args()
    saved=json.loads(REPORT.read_text()); old=historical_rows(); result=[]
    for profile in saved['profiles']:
        p=Path(profile['artifacts']['ptx']['path']); assert sha(p)==profile['artifacts']['ptx']['sha256']
        audit=Audit(profile); rows=audit.execute(p); matches=[]
        for row in rows:
            found=[i for i,(_,r) in old.items() if r==row]
            assert len(found)==1,(p,'unmatched row graph'); matches.append(found[0])
        result.append({'profile':p.stem,'ptx':str(p.relative_to(ROOT)),'ptx_sha256':sha(p),
            'block_heads':audit.h,'block_chunk':audit.c,'bias':profile['bias'],'softplus':profile['softplus'],
            'bias_reverse':next(iter(audit.bias_orders),False),'row_plan_ids':matches,
            'output_coordinates':audit.h*audit.c,'ordered_rows_sha256':digest(rows),
            'scan_instructions':audit.scan_ops})
        print(p.stem, 'PASS', f'{audit.h*audit.c} outputs',sorted(set(matches)))
    profile=saved['profiles'][3]; p=Path(profile['artifacts']['ptx']['path']); source=p.read_text()
    scan=source.index('\t.loc\t1 69 14')
    add=re.search(r'add.f32\s+(%r\d+), (%r\d+), (%r\d+);',source[scan:])
    start,end=scan+add.start(),scan+add.end()
    d,a,b=add.groups()
    minimum=re.search(r'min.f32\s+(%r\d+), (%r\d+), (%r\d+);',source)
    maximum=re.search(r'max.f32\s+(%r\d+), (%r\d+), (%r\d+);',source)
    softplus=re.search(r'(\.loc\s+1 62 58[^\n]*\n\s*selp.f32\s+%r\d+, )(%r\d+)',source)
    mutations={
      'scan_add_operand_swap':source[:start]+f'add.f32 {d}, {b}, {a};'+source[end:],
      'scan_fma_ftz':source[:scan]+source[scan:].replace('fma.rn.f32','fma.rn.ftz.f32',1),
      'upper_replaced_by_lower':source[:minimum.start()]+f'min.f32 {minimum[1]}, {minimum[2]}, {maximum[3]};'+source[minimum.end():],
      'exp_scale_changed':source.replace('0f3FB8AA3B','0f3F800000',1),
      'exp_plus_one_changed':source.replace('0f3F800000','0f40000000',1),
      'log_result_from_bound':source[:softplus.start(2)]+maximum[3]+source[softplus.end(2):],
    }
    controls=[]
    with tempfile.TemporaryDirectory(prefix='chunked-scan-controls-') as tmp:
        for name,content in mutations.items():
            changed=Path(tmp)/f'{name}.ptx'; changed.write_text(content)
            try:
                rows=Audit(profile).execute(changed)
                assert all(any(row==r for _,r in old.values()) for row in rows)
            except (AssertionError,TypeError,KeyError): controls.append({'mutation':name,'rejected':True})
            else: raise AssertionError(('mutation accepted',name))
    packet={'status':'passed','kind':'saved-artifact-symbolic-scan-audit','script_sha256':sha(Path(__file__)),
        'source_sha256':saved['source_sha256'],'lowering_report_sha256':sha(REPORT),
        'scope':'42 saved Triton 3.7.1 SM80 profiles; ordered scalar scan DAGs and runtime clamp operands. No GPU execution or universal compiler theorem.',
        'historical_plans':{i:{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'ordered_row_sha256':digest(row)} for i,(p,row) in old.items()},
        'profiles':result,'mutation_controls':controls}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(packet,indent=2)+'\n')
    print(args.output)
if __name__=='__main__': main()
