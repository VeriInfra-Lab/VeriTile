"""Successor saved-PTX binding: preserve row-half identity and exact WGMMA literals.
Decodes both literal B-layout transpose settings; does not evaluate floating-point instructions.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys
P=Path(__file__).resolve().parent
class Atom:
    def __init__(self,kind,*args): self.kind,self.args=kind,args
class Bits(tuple): pass

def bits(x,width):
    if isinstance(x,int): return [(x>>i)&1 for i in range(width)]
    if isinstance(x,Atom): return [(x,i) for i in range(width)]
    return list(x[:width])+[0]*max(0,width-len(x))
def norm(b):
    if all(isinstance(x,int) for x in b):return sum(x<<i for i,x in enumerate(b))
    if len(b) in (16,32) and isinstance(b[0],tuple) and all(x==(b[0][0],i) for i,x in enumerate(b)):return b[0][0]
    return Bits(b)
def bop(op,a,b,width):
    aa,bb=bits(a,width),bits(b,width)
    out=[]
    for x,y in zip(aa,bb):
        if op=='and':z=0 if x==0 or y==0 else y if x==1 else x if y==1 or x==y else None
        elif op=='or':z=1 if x==1 or y==1 else y if x==0 else x if y==0 or x==y else None
        else:z=y if x==0 else x if y==0 else 0 if x==y else None
        assert z is not None,(op,x,y)
        out.append(z)
    return norm(out)
def splitargs(s):return [x.strip() for x in re.split(r',\s*(?![^{}]*\})',s)]
def vector(s):return [x.strip() for x in s.strip('{} ').split(',')]
def signed(x,w):
    x=x&((1<<w)-1)
    return x-(1<<w) if x&(1<<(w-1)) else x

def parse(path):
    code=[];labels={}
    for raw in path.read_text().splitlines():
        line=raw.split('//')[0].strip()
        if not line or line.startswith(('.', '{','}')):continue
        if line.endswith(':'):labels[line[:-1]]=len(code);continue
        if not line.endswith(';'):continue
        pred=None
        if line.startswith('@'):pred,line=line.split(None,1);pred=pred[1:]
        parts=line[:-1].split(None,1)
        code.append((pred,parts[0],splitargs(parts[1]) if len(parts)>1 else [],line))
    return code,labels

def run(path,profile,pid,fault=None):
    s=profile['shape'];N,C,O,H,W,OH,OW,KH,KW,SH,SW,PH,PW,G=[s[k] for k in ['N','C','O','H','W','OH','OW','KH','KW','SH','SW','PH','PW','groups']]
    cg,og=C//G,O//G;gx,gy=(N*OH*OW+127)//128,(O+31)//32
    strides=profile.get('strides',{'input':[C*H*W,H*W,W,1],
        'weight':[cg*KH*KW,KH*KW,KW,1],'output':[O*OH*OW,OH*OW,OW,1]})
    sn,sc,sx,sy=strides['input'];so,tc,tx,ty=strides['weight']
    on,oc,oh,ow=strides['output']
    offsets=profile.get('storage_offsets',{'input':0,'weight':0,'output':0})
    base={'a':(1<<40)+4*offsets['input'],'b':(2<<40)+4*offsets['weight'],
        'o':(3<<40)+4*offsets['output']}
    spans={'a':1+sum((d-1)*s for d,s in zip((N,C,H,W),strides['input'])),
        'b':1+sum((d-1)*s for d,s in zip((O,cg,KH,KW),strides['weight']))}
    arguments={'input_pointer':base['a'],'weight_pointer':base['b'],'output_pointer':base['o'],
      'batch_dim':N,'in_feat_dim':C,'in_height':H,'in_width':W,'out_feat_dim':O,'out_height':OH,'out_width':OW,
      'input_batch_stride':sn,'input_in_feat_stride':sc,'input_height_stride':sx,'input_width_stride':sy,
      'weight_out_feat_stride':so,'weight_in_feat_stride':tc,'weight_height_stride':tx,'weight_width_stride':ty,
      'output_batch_stride':on,'output_out_feat_stride':oc,'output_height_stride':oh,'output_width_stride':ow}
    params=[arguments[k] for k,v in profile['signature'].items() if v!='constexpr']
    mma_count=0;bindings=0
    regs=[{} for _ in range(128)];shared={};atoms={};outputs={};count=0
    code,labels=parse(path)
    def val(t,s):
        s=s.strip()
        if s in regs[t]:return regs[t][s]
        if s=='global_smem':return 0
        if s=='%tid.x':return t
        if s.startswith('%ctaid.'):return pid['xyz'.index(s[-1])]
        if s.startswith('%nctaid.'):return (gx,gy,G)['xyz'.index(s[-1])]
        if s.startswith('0f'):return int(s[2:],16)
        return int(s,0)
    def addr(t,s):
        s=s.strip('[] ')
        m=re.fullmatch(r'([^+\s]+)\s*\+\s*(-?\d+)',s)
        return val(t,m[1])+int(m[2]) if m else val(t,s)
    def put(t,d,v,w=32):regs[t][d]=v&((1<<w)-1) if isinstance(v,int) else v
    def getmem(a,width):return norm([z for i in range(width//8) for z in shared[a+i]])
    def setmem(a,value,width):
        bb=bits(value,width)
        for i in range(width//8):shared[a+i]=bb[8*i:8*i+8]
    def inputword(a):
        if a>=base['b']:r='b'
        else:r='a'
        n=spans[r]
        index=(a-base[r])//4
        assert (a-base[r])%4==0 and 0<=index<n,(r,index,n,pid)
        return atoms.setdefault((r,index),Atom('input',r,index))
    def expected_a(r,h,w,c):
        n=r//(OH*OW);y=r//OW%OH*SH-PH+h;x=r%OW*SW-PW+w
        if not(n<N and c<cg and 0<=y<H and 0<=x<W):return 0
        index=n*sn+(pid[2]*cg+c)*sc+y*sx+x*sy
        return atoms.setdefault(('a',index),Atom('input','a',index))
    def expected_b(f,h,w,c):
        if not(c<cg and f<og):return 0
        index=(pid[2]*og+f)*so+c*tc+h*tx+w*ty
        return atoms.setdefault(('b',index),Atom('input','b',index))
    def cast(x):return Atom('halfRN',x)
    def equal(x,y):
        if isinstance(x,Atom) and isinstance(y,Atom):return x.kind==y.kind and len(x.args)==len(y.args) and all(equal(a,b) for a,b in zip(x.args,y.args))
        return x==y
    def desc_address(d,row,col,size):
        start=(d&16383)<<4;mode=d>>62;sw={1:128,2:64}[mode]
        assert ((d>>32)&16383)<<4==sw*8
        phase=((row*sw)>>7)&(sw//16-1)
        return (start//sw)*sw+row*sw+((start%sw+col*size)^(phase*16))
    pc=0
    while pc<len(code):
        pred,op,args,line=code[pc];pc+=1
        active=[t for t in range(128) if pred is None or bool(val(t,pred.lstrip('!')))^pred.startswith('!')]
        if op in ('bra','bra.uni'):
            assert len(active) in (0,128),('divergent branch',line)
            if active:pc=labels[args[0]]
            continue
        if op=='ret':break
        if op.startswith(('bar.','cp.async.commit_group','cp.async.wait_group','fence.','wgmma.fence.','wgmma.commit_group.','wgmma.wait_group.')):continue
        if op.startswith('ldmatrix.'):
            snapshots=[addr(t,args[1]) for t in range(128)]
            for t in active:
                row=(t%32)//4;col=(t%4)*4;warp=t//32*32
                for j,d in enumerate(vector(args[0])):put(t,d,getmem(snapshots[warp+j*8+row]+col,32))
            continue
        if op.startswith('wgmma.mma_async.'):
            assert len(active)==128
            half=profile['fp16'];kdim=16 if half else 8;perrow=32//kdim
            expected_opcode='wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16' if half else 'wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32'
            assert op==expected_opcode,('literal WGMMA opcode',op,expected_opcode)
            assert args[4:6]==['1','1'],('literal WGMMA scales',args[4:])
            if half:assert args[6:] in (['0'],['1']),('literal B transpose',args[6:])
            else:assert args[6:]==[],('TF32 options',args[6:])
            block=mma_count//(2*perrow);local=mma_count%(2*perrow);rhalf=local//perrow;chunk=local%perrow
            h=block//(KW*((cg+31)//32));w=block//((cg+31)//32)%KW;cb=block%((cg+31)//32)
            assert h<KH
            A={};B={};dest=vector(args[0]);assert len(dest)==16
            if half:
                av=vector(args[1]);assert len(av)==4
                for t in range(128):
                    lane=t%32;warp=t//32
                    for reg,v in enumerate(av):
                        packed=bits(val(t,v),32)
                        for part in range(2):
                            i=warp*16+lane//4+(reg%2)*8;k=(lane%4)*2+(reg//2)*8+part
                            A[i,k]=norm(packed[part*16:(part+1)*16])
                db=val(0,args[2]);assert all(val(t,args[2])==db for t in active)
                for k in range(16):
                    for j in range(32):
                        row,col=(k,j) if args[6]=='1' else (j,k)
                        B[k,j]=getmem(desc_address(db,row,col,2),16)
            else:
                da,db=val(0,args[1]),val(0,args[2]);assert all(val(t,args[1])==da and val(t,args[2])==db for t in active)
                for i in range(64):
                    for k in range(8):A[i,k]=getmem(desc_address(da,i,k,4),32)
                for k in range(8):
                    for j in range(32):B[k,j]=getmem(desc_address(db,j,k,4),32)
            for (i,k),a in A.items():
                expected=expected_a(pid[0]*128+rhalf*64+i,h,w,cb*32+chunk*kdim+k)
                if half:expected=cast(expected)
                assert equal(a,expected),('WGMMA A binding',mma_count,i,k,a.args if isinstance(a,Atom) else a,expected.args if isinstance(expected,Atom) else expected)
                bindings+=1
            for (k,j),b in B.items():
                expected=expected_b(pid[1]*32+j,h,w,cb*32+chunk*kdim+k)
                if half:expected=cast(expected)
                assert equal(b,expected),('WGMMA B binding',mma_count,k,j,b.args if isinstance(b,Atom) else b,expected.args if isinstance(expected,Atom) else expected)
                bindings+=1
            for t in active:
                assert val(t,args[3])!=0,('literal WGMMA live carry enable',val(t,args[3]))
                lane=t%32;warp=t//32
                for reg,d in enumerate(dest):
                    i=warp*16+lane//4+((reg%4)//2)*8;j=(lane%4)*2+reg%2+(reg//4)*8
                    put(t,d,Atom('mma',block,chunk,rhalf,i,j,val(t,d)))
            mma_count+=1
            continue
        if op.startswith('shfl.'):
            old=[val(t,args[1]) for t in range(128)]
            for t in active:put(t,args[0],old[t//32*32+(val(t,args[2])&31)])
            continue
        for t in active:
            if op.startswith('ld.param.'):
                put(t,args[0],params[int(re.search(r'param_(\d+)',args[1])[1])],64 if '.b64' in op else 32)
            elif op.startswith('mov.'):
                put(t,args[0],val(t,args[1]),64 if '64' in op else 16 if '16' in op else 32)
            elif op.startswith('cp.async.'):
                dest,src=addr(t,args[0]),addr(t,args[1]);size=val(t,args[2]);actual=val(t,args[3]) if len(args)>3 else size
                for j in range(0,size,4):
                    if j<actual and src+j<base['a']:
                        raise AssertionError(('bad copy',line,t,src,actual,params))
                    setmem(dest+j,inputword(src+j) if j<actual else 0,32)
            elif op.startswith('ld.shared.'):
                ds=vector(args[0]) if args[0].startswith('{') else [args[0]]
                for j,d in enumerate(ds):put(t,d,getmem(addr(t,args[1])+4*j,32))
            elif op.startswith('st.shared.'):
                vs=vector(args[1]) if args[1].startswith('{') else [args[1]]
                for j,v in enumerate(vs):
                    width=16 if '.b16' in op else 32
                    setmem(addr(t,args[0])+width//8*j,val(t,v),width)
            elif op.startswith('st.global.'):
                vs=vector(args[1]) if args[1].startswith('{') else [args[1]]
                for j,v in enumerate(vs):
                    a=addr(t,args[0])+4*j-base['o'];assert a%4==0
                    assert a//4 not in outputs,('duplicate store',a//4)
                    outputs[a//4]=val(t,v)
            elif op=='fma.rn.f32':
                put(t,args[0],Atom('fma',val(t,args[1]),val(t,args[2]),val(t,args[3])));count+=1
            elif op.startswith('setp.'):
                comparison=op.split('.')[1];a,b=val(t,args[1]),val(t,args[2])
                if '.s32' in op:a,b=signed(a,32),signed(b,32)
                regs[t][args[0]]=int({'lt':a<b,'gt':a>b,'ne':a!=b,'eq':a==b}[comparison])
            elif op.startswith('selp.'):
                put(t,args[0],val(t,args[1] if val(t,args[3]) else args[2]),16 if '.b16' in op else 32)
            elif op=='cvt.rn.f16x2.f32':
                put(t,args[0],Bits(bits(cast(val(t,args[2])),16)+bits(cast(val(t,args[1])),16)))
            elif op=='cvt.rn.f16.f32':
                put(t,args[0],cast(val(t,args[1])),16)
            elif op.startswith('cvt.'):

                dst,src=op.split('.')[1:3];v=val(t,args[1]);w=int(dst[1:])
                if src.startswith('s'):v=signed(v,int(src[1:]))
                put(t,args[0],v,w)
            elif op.startswith(('and.','or.','xor.')):
                w=16 if '16' in op else 64 if '64' in op else 1 if '.pred' in op else 32
                put(t,args[0],bop(op.split('.')[0],val(t,args[1]),val(t,args[2]),w),w)
            elif op.startswith(('shl.','shr.')):
                w=16 if '16' in op else 64 if '64' in op else 32;v=val(t,args[1]);shift=val(t,args[2]);kind=op.split('.')[0]
                if isinstance(v,int):
                    if op.startswith('shr.s'):v=signed(v,w)
                    result=v<<shift if kind=='shl' else v>>shift
                else:
                    bb=bits(v,w);result=norm(([0]*shift+bb)[:w] if kind=='shl' else bb[shift:]+[0]*shift)
                put(t,args[0],result,w)
            elif op in ('bfe.u32','bfe.s32'):
                v=val(t,args[1]);lo=val(t,args[2]);width=val(t,args[3]);bb=bits(v,32)[lo:lo+width];put(t,args[0],norm(bb+[bb[-1] if op=='bfe.s32' else 0]*(32-width)))
            elif op.startswith(('add.','sub.','mul.','mad.','div.','neg.')):
                kind=op.split('.')[0];w=64 if '64' in op or '.wide.' in op else 32
                vs=[val(t,a) for a in args[1:]]
                assert all(isinstance(v,int) for v in vs),(line,vs)
                if '.s32' in op:vs=[signed(v,64 if '.wide.' in op and i==2 else 32) for i,v in enumerate(vs)]
                if '.s64' in op:vs=[signed(v,64) for v in vs]
                if kind=='add':v=vs[0]+vs[1]
                elif kind=='sub':v=vs[0]-vs[1]
                elif kind=='mul':v=vs[0]*vs[1]
                elif kind=='mad':v=vs[0]*vs[1]+vs[2]
                elif kind=='neg':v=-vs[0]
                else:v=(abs(vs[0])//abs(vs[1]))*(-1 if (vs[0]<0)^(vs[1]<0) else 1)
                put(t,args[0],v,w)
            else:raise AssertionError(('unsupported opcode',line))
    expected=set()
    for r in range(pid[0]*128,min((pid[0]+1)*128,N*OH*OW)):
        for f in range(pid[1]*32,min((pid[1]+1)*32,og)):
            expected.add((r//(OH*OW)*O+pid[2]*og+f)*OH*OW+r%(OH*OW))
    assert set(outputs)==expected,('store coverage',len(outputs),len(expected),pid)
    if fault=='swap_fragment_store_sources':
        assert profile['fp16'] or profile['tf32']
        def offset_of(row,feature):
            return (row//(OH*OW)*O+pid[2]*og+feature)*OH*OW+row%(OH*OW)
        pairs=[]
        for row in range(pid[0]*128,pid[0]*128+64):
            for feature in range(pid[1]*32,min((pid[1]+1)*32,og)):
                left,right=offset_of(row,feature),offset_of(row+64,feature)
                if left in outputs and right in outputs:pairs.append((left,right))
        assert pairs,('negative control has no two-fragment stores',pid)
        for left,right in pairs:outputs[left],outputs[right]=outputs[right],outputs[left]
    else:assert fault is None
    observed_fragments=set()
    for offset,word in outputs.items():
        n=offset//(O*OH*OW);o=offset//(OH*OW)%O;sp=offset%(OH*OW);r=n*OH*OW+sp;f=o%og
        events=[]
        if profile['fp16'] or profile['tf32']:
            perrow=2 if profile['fp16'] else 4
            while isinstance(word,Atom) and word.kind=='mma':
                block,chunk,rhalf,i,j,word=word.args;events.append((block,chunk,rhalf,i,j))
                observed_fragments.add(rhalf)
            assert word==0
            events.reverse()
            assert events==[(block,k,(r-pid[0]*128)//64,r%64,f%32) for block in range(KH*KW*((cg+31)//32)) for k in range(perrow)],('MMA carry chain',offset,events)
        else:
            while isinstance(word,Atom) and word.kind=='fma':
                a,b,word=word.args;events.append((a,b))
            assert word==0
            events.reverse();assert len(events)==KH*KW*((cg+31)//32)*32
            for z,(a,b) in enumerate(events):
                c=z%(((cg+31)//32)*32);w=z//(((cg+31)//32)*32)%KW;h=z//(((cg+31)//32)*32)//KW
                assert equal(a,expected_a(r,h,w,c)),('FMA A',offset,h,w,c)
                assert equal(b,expected_b(f,h,w,c)),('FMA B',offset,h,w,c)
    return {'pid':pid,'stored_words':len(outputs),'fma_instructions':count,'wgmma_instructions':mma_count,'wgmma_operand_bindings':bindings,'complete_ordered_carry':True,'masked_zero_operands':True,'store_coverage':True,'fragment_anchors':sorted(observed_fragments),'fragment_identity_in_every_carry_event':True,'literal_opcode_scales_transpose':True}

parser=argparse.ArgumentParser()
parser.add_argument('--report',type=Path,default=Path('/tmp/conv2d-checked-lowering/report.json'))
parser.add_argument('--output',type=Path,default=P/'ptx-provenance.json')
options=parser.parse_args()
report=options.report;data=json.loads(report.read_text());rows=[]
for profile in data['profiles']:
 path=Path(profile['artifacts']['ptx']['path']);gx,gy,gz=profile['grid']
 pids=list(dict.fromkeys([(0,0,0),(gx-1,gy-1,gz-1),(gx-1,max(0,(profile['shape']['O']//gz-1)//32),gz-1)]))
 cases=[run(path,profile,pid) for pid in pids]
 rows.append({'label':profile['label'],'ptx':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':cases})
 print(profile['label'],'PASS',sum(c['stored_words'] for c in cases),flush=True)
negative=[]
for profile in data['profiles']:
 if profile['shape']['N']*profile['shape']['OH']*profile['shape']['OW']>128 and (profile['fp16'] or profile['tf32']):
  try:run(Path(profile['artifacts']['ptx']['path']),profile,(0,0,0),'swap_fragment_store_sources')
  except AssertionError as error:
   assert error.args[0][0]=='MMA carry chain',error
   negative.append({'profile':profile['label'],'fault':'swap_fragment_store_sources','rejected':True})
  else:raise AssertionError('fragment swap escaped provenance check')
result={'negative_controls':negative,'source_sha256':data['source_sha256'],'status' :'PASS','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'bounded symbolic PTX def-use/shared-memory/WGMMA descriptor and register-layout provenance; arithmetic instructions uninterpreted','GPU_execution':False,'native_numerical_conformance':False,'profiles':rows}
options.output.write_text(json.dumps(result,indent=2)+'\n')
