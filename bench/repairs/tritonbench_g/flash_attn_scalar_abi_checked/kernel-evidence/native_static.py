"""Static symbolic PTX topology audit. No floating values or GPU execution."""
from pathlib import Path
import re,json,hashlib,functools,sys
ROOT=Path.cwd(); BASE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def leaf(row,col,owner):return ('leaf',row,col,owner)
def tree(op,n,row,owner):
 def local(o):
  xs=[leaf(row,c,o) for b in range(0,n,8) for c in (b+2*(o%4),b+2*(o%4)+1)]
  return functools.reduce(lambda a,b:(op,a,b),xs)
 return (op,(op,local(owner),local(owner^2)),(op,local(owner^1),local(owner^3)))
def rowof(x):
 if x[0]=='leaf':return x[1]
 return rowof(x[1])
def parse(text):
 loc=0;out=[];seen_pv=False;skip_label=None
 for number,line in enumerate(text.splitlines(),1):
  if skip_label:
   if line.strip().startswith(skip_label+':'):skip_label=None
   else:continue
  m=re.match(r'\s*\.loc\s+1\s+(\d+)',line)
  if m:loc=int(m[1])
  if line.strip().startswith('.loc'):
   origin=re.search(r'flash_attn_scalar_abi_checked.py:(\d+):',line)
   if origin:loc=int(origin[1])-2
  l=line.strip()
  if not l or l.startswith(('.', '/', '{','}','$','@')):continue
  if ';' not in l:continue
  op,_,arg=l.partition(' ');arg=arg.strip().split(';')[0]
  if not arg:continue
  args=[a.strip() for a in re.split(r',\s*(?![^{}]*\})',arg)]
  if loc==163 and op.startswith('wgmma.mma_async'):seen_pv=True
  if seen_pv and op=='bra.uni':skip_label=args[0];continue
  out.append((number,loc,op,args))
 return out

def audit(case):
 bm,d=128,case['width'];one=case['raw_scale_kind']=='constexpr';threads=case['options']['num_warps']*32
 path=BASE/'native'/f"{case['label']}.ptx"; ins=parse(path.read_text()); regs={};qk=[];pv=[]
 max_checks=sum_checks=route_checks=division_checks=0; qk_mul=0;normalizers=[];rescale=[];fmas=[];predicates={};query_scales=set();outscale=[];cast_counts={}
 qkw=64
 pvw=d//2 if bm==64 and d>64 else d
 def val(a,t):
  if a=='%tid.x':return t
  if a in regs:return regs[a][t]
  try:return int(a,0)
  except ValueError:return None
 def assign(dst,fn):regs[dst]=[fn(t) for t in range(threads)]
 for number,loc,op,a in ins:
  if op.startswith('wgmma.mma_async'):
   width=int(re.search(r'm64n(\d+)k8\.f32\.tf32\.tf32$',op)[1])
   dst=tuple(re.findall(r'%\w+',a[0]));seed=a[3]
   if loc==156:
    assert width==qkw,(case['label'],width,qkw)
    if dst not in [x[0] for x in qk]:
     assert all(val(seed,t) in (-1,1) for t in range(threads)),(case['label'],'QK first live',seed)
     group=len(qk);qk.append([dst,0])
    group=next(i for i,x in enumerate(qk) if x[0]==dst)
    if qk[group][1]:assert all(val(seed,t) in (-1,1) for t in range(threads))
    qk[group][1]+=1
    for j,r in enumerate(dst):
     def qkleaf(t,j=j,g=group):
      row=(t%128//32)*16+(t%32)//4+(j%4//2)*8
      col=j//4*8+t%4*2+j%2
      if bm==64 and d>64:col+=g*qkw
      elif threads==128:row+=g*64
      else:row+=t//128*64
      return leaf(row,col,t)
     assign(r,qkleaf)
   elif loc==163:
    assert width==pvw
    if dst not in [x[0] for x in pv]:pv.append([dst,0])
    group=next(i for i,x in enumerate(pv) if x[0]==dst);kpart=pv[group][1];pv[group][1]+=1
    assert all(val(seed,t) in (-1,1) for t in range(threads)),('PV live',seed)
    src=re.findall(r'%\w+',a[1]);assert len(src)==4
    for j,r in enumerate(src):
     for t in range(threads):
      row=(t%128//32)*16+t%32//4+(j%2)*8
      if threads==128:row+=group*64
      elif bm==128:row+=t//128*64
      col=kpart*8+t%4+j//2*4
      owner=row%64//16*32+row%8*4+col%8//2
      if d>64:owner+=(row//64*128 if bm==128 else t//128*128)
      got=val(r,t);want=leaf(row,col,owner)
      assert got==want,('PV route',case['label'],number,j,t,got,want)
      route_checks+=1
    for j,r in enumerate(dst):
     def accleaf(t,j=j,g=group):
      row=(t%128//32)*16+t%32//4+(j%4//2)*8
      col=j//4*8+t%4*2+j%2
      if threads==128:row+=g*64
      elif bm==128:row+=t//128*64
      else:col+=t//128*pvw
      owner=row%64//16*32+row%8*4+col%8//2
      if d>64:owner+=(row//64*128 if bm==128 else col//(d//2)*128)
      assert owner==t,('storedOwner',case['label'],row,col,owner,t)
      return ('acc',row,col,t)
     assign(r,accleaf)
   else:raise AssertionError(('unknown dot',loc))
   continue
  if not a:continue
  dst=a[0]
  if op in ('mov.b32','mov.u32','mov.b64','mov.u64','mov.pred'):
   assign(dst,lambda t:val(a[1],t));continue
  if op in ('and.b32','or.b32','xor.b32','shl.b32','shr.u32','neg.s32','add.s32'):
   def integer(t):
    x=val(a[1],t);y=val(a[2],t) if len(a)>2 else 0
    if not isinstance(x,int) or not isinstance(y,int):return None
    return {'and.b32':lambda:x&y,'or.b32':lambda:x|y,'xor.b32':lambda:x^y,'shl.b32':lambda:x<<y,'shr.u32':lambda:(x&0xffffffff)>>y,'neg.s32':lambda:-x,'add.s32':lambda:x+y}[op]()
   assign(dst,integer);continue
  if op=='bfe.u32':
   assign(dst,lambda t:((val(a[1],t)>>int(a[2]))&((1<<int(a[3]))-1)) if isinstance(val(a[1],t),int) else None);continue
  if op=='setp.eq.b32':
   assign(dst,lambda t:int(val(a[1],t)==val(a[2],t)) if val(a[1],t) is not None and val(a[2],t) is not None else None);continue
  if op=='mul.f32' and loc==143:
   assert a[2]=='0f3FB8AA3B';query_scales.add(dst);continue
  if op=='mul.f32' and loc==145:
   assert (a[2]=='0f3FB8AA3B' if one else a[1] in query_scales),('query operand order',case['label'],number)
   qk_mul+=1;assign(dst,lambda t:val(a[2],t));continue
  if op=='selp.f32' and loc==-1:
   assert a[1]=='0fCCBEBC20';assign(dst,lambda t:val(a[2],t));continue
  if op=='max.f32' and loc==158 and isinstance(val(a[1],0),tuple) and val(a[1],0)[0] in ('leaf','max') or op=='add.f32' and loc==164:
   kind='max' if op=='max.f32' else 'sum'
   assign(dst,lambda t:(kind,val(a[1],t),val(a[2],t)));continue
  if op=='shfl.sync.bfly.b32' and loc in (158,164):
   mask=int(a[2]);assert mask in (1,2)
   assign(dst,lambda t:val(a[1],t^mask));continue
  if op=='max.f32' and loc==158:
   for t in range(threads):
    x=val(a[2],t);row=rowof(x);assert x==tree('max',64,row,t),('max tree',case['label'],t)
    max_checks+=1
   assign(dst,lambda t:('center',rowof(val(a[2],t)),t));continue
  if op=='sub.f32' and loc==160:
   assign(dst,lambda t:val(a[1],t));continue
  if op=='ex2.approx.ftz.f32' and loc==160:
   assign(dst,lambda t:val(a[1],t));continue
  if op=='sub.f32' and loc==159:
   assign(dst,lambda t:val(a[2],t));continue
  if op=='ex2.approx.ftz.f32' and loc==159:
   assign(dst,lambda t:('alpha',val(a[1],t)[1],t));continue
  if op=='fma.rn.f32' and loc==164:
   assert dst==a[1]
   for t in range(threads):
    alpha=val(a[2],t);assert alpha[0]=='alpha'
    assert val(a[3],t)==tree('sum',64,alpha[1],t),('sum tree',case['label'],t)
    sum_checks+=1
   fmas.append({'ptx_line':number,'operands':a});assign(dst,lambda t:('normalizer',val(a[2],t)[1],t));continue
  if op=='fma.rn.f32' and loc==161:
   assert a[2]=='0f00000000' and all(val(a[3],t)[0]=='alpha' for t in range(threads))
   outscale.append({'ptx_line':number,'operands':a});assign(dst,lambda t:('outscale',val(a[3],t)[1],t));continue
  if op in ('cvt.rn.f16.f32','cvt.f32.f16') and loc in (145,156,163,180):
   cast_counts[(loc,op)]=cast_counts.get((loc,op),0)+1
   assign(dst,lambda t:val(a[1],t));continue
  if op=='mul.f32' and loc==162:
   assert dst==a[1] and all(val(a[2],t)[0]=='outscale' for t in range(threads))
   rescale.append({'ptx_line':number,'operands':a});continue
  if op=='div.full.f32' and loc==169:
   for t in range(threads):
    acc=val(a[1],t);den=val(a[2],t)
    assert acc[0]=='acc' and den==('normalizer',acc[1],t),('division',case['label'],number,t,acc,den)
    division_checks+=1
   continue
  if op=='selp.f32' and loc==163:
   assign(dst,lambda t:val(a[1] if val(a[3],t) else a[2],t) if val(a[3],t) is not None else None);continue
  if op=='shfl.sync.idx.b32' and loc==163:
   assign(dst,lambda t:val(a[1],t//32*32+(val(a[2],t)&31)) if isinstance(val(a[2],t),int) else None);continue
 assert max_checks and sum_checks and route_checks and division_checks and outscale and (query_scales or one)
 assert '__nv_log2f.exit' in path.read_text()
 conversion={'i32':'cvt.rn.f32.s32','i64':'cvt.rn.f32.s64','u64':'cvt.rn.f32.u64'}.get(case['raw_scale_kind'])
 if conversion:assert conversion in path.read_text()
 assert all(n==d//8 for _,n in qk),(case['label'],'QK K8 coverage',qk)
 assert all(n==64//8 for _,n in pv),(case['label'],'PV K8 coverage')
 assert len(rescale)==sum(len(dst) for dst,_ in pv)
 return {'label':case['label'],'ptx_sha256':sha(path),'qk_n':qkw,'pv_n':pvw,'qk_groups':len(qk),'pv_groups':len(pv),'qk_k8_per_group':d//8,'pv_k8_per_group':64//8,'max_tree_thread_checks':max_checks,'sum_tree_thread_checks':sum_checks,'pv_route_word_checks':route_checks,'output_division_owner_checks':division_checks,'qk_scale_swapped_instructions':qk_mul,'normalizer_fmas':fmas,'outscale_fmas':outscale,'casts':{str(k):v for k,v in cast_counts.items()},'log2':'libdevice __nv_log2f','raw_scale_kind':case['raw_scale_kind'],'constexpr_one_coefficient_fold':one,'raw_integer_conversion':conversion,'old_accumulator_rescale_count':len(rescale),'shared_bytes':case['metadata']['shared'],'SM90_233472_byte_SM_capacity_exceeded':case['metadata']['shared']>233472,'status':'pass'}
report={'kind':'static symbolic PTX coordinate/tree/operand audit; no numerical experiment','GPU_execution':False,'cases':[]}
for case in json.loads((BASE/'native-probe.json').read_text())['profiles']:
 result=audit(case);report['cases'].append(result);print(result['label'],'PASS',flush=True)
(BASE/'native-static.json').write_text(json.dumps(report,indent=2)+'\n')
