import sys;sys.dont_write_bytecode=True
from pathlib import Path
import functools,hashlib,json,random,re
W=Path(__file__).resolve().parent
C=Path('/tmp/veritile-standardization-20261015/quant-transpose-finite-acceptance-revision1-wip')
P=C/'peer-review'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((P/'native-review-receipt.json').read_text())
assert sha(P/'native-review-receipt.json')=='30218217496d7351a19c53c30ed3032874863f6c6c896b441c99011f1686b61e'
for r in receipt['records']:
 for a in r.get('artifacts',{}).values():assert sha(P/Path(a['path']).name)==a['sha256']

def wrap(x,w):return (x+2**(w-1))%2**w-2**(w-1)
class Undefined(Exception):pass
def trunc(a,b):
 if b==0:raise Undefined('division_zero')
 return (-1 if (a<0)!=(b<0) else 1)*(abs(a)//abs(b))

# Independent executable reading of the source expression tree. Kind0 is a
# constexpr, with finite signed32/64 values wrapped at each individual node.
def meta(x):return (0,x) if x==1 else (32 if x<2**31 else 64,x)
def k(x):return (0,x)
def op(name,a,b):
 w=max(a[0],b[0]);x,y=a[1],b[1]
 if name in ['div','rem'] and (y==0 or (w and x==-2**(w-1) and y==-1)):raise Undefined('integer_division')
 z={'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'min':lambda:min(x,y),'div':lambda:x//y if w==0 else trunc(x,y),'rem':lambda:x%y if w==0 else x-y*trunc(x,y)}[name]()
 return (w,wrap(z,w) if w else z)
def source(M,N,sam,san,pid,lane,base=2**31,width=4,outbase=2**34):
 m,n,p=meta(M),meta(N),(32,wrap(pid,32))
 gm=op('div',op('sub',op('add',m,k(128)),k(1)),k(128))
 gn=op('div',op('sub',op('add',n,k(128)),k(1)),k(128))
 wd=op('mul',k(8),gn);group=op('div',p,wd);start=op('mul',group,k(8));size=op('min',op('sub',gm,start),k(8))
 row=op('add',start,op('rem',p,size));col=op('div',op('rem',p,wd),size)
 rm=op('add',op('mul',row,k(128)),(32,lane//128));rn=op('add',op('mul',col,k(128)),(32,lane%128))
 a=op('add',op('mul',rm,meta(sam)),op('mul',rn,meta(san)))
 b=op('add',op('mul',rm,meta(1)),op('mul',rn,meta(M)))
 return {'mask':rm[1]<M and rn[1]<N,'input':(base+a[1]*width)%2**64,'output':(outbase+b[1])%2**64,'rm':rm[1],'rn':rn[1],'input_offset':a[1],'input_kind':a[0],'output_kind':b[0]}

def shape(s):
 m=re.search(r'tensor<((?:\d+x)+)',s)
 return tuple(map(int,m[1][:-1].split('x'))) if m else ()
def widthof(s):
 # Integer result type, not an attribute type before the final colon.
 ws=re.findall(r'(?:^|[<x ])i(1|8|16|32|64)(?=[ >]|$)',s)
 return int(ws[-1]) if ws else None
def parse(path):
 nodes={};loads=[];store=None
 for line in path.read_text().splitlines():
  line=line.strip().split(' loc(')[0]
  if line.startswith('tt.store '):
   args=re.findall(r'%[\w]+',line.split(':')[0]);store=(args[0],args[2]);continue
  m=re.match(r'(%[\w]+) = ([\w.]+) (.*)',line)
  if not m:continue
  name,kind,body=m.groups();args=re.findall(r'%[\w]+',body.split(':')[0]);ty=body.rsplit(' : ',1)[-1]
  if kind in ['arith.extsi','arith.trunci']:ty=ty.split(' to ')[-1]
  if kind.startswith('tt.') and ' -> ' in ty:ty=ty.split(' -> ')[-1]
  data={'op':kind,'args':args,'shape':shape(ty),'width':widthof(ty),'body':body}
  if kind=='arith.constant':
   literal=body.split(' : ')[0];literal=literal.removeprefix('dense<').removesuffix('>')
   try:data['literal']=int(literal)
   except ValueError:data['literal']=None
  elif kind=='tt.expand_dims':data['axis']=int(re.search(r'axis = (\d+)',body)[1])
  elif kind=='tt.addptr':
   ptr=re.search(r'!tt.ptr<([fi])(\d+)>',body);data['element_bytes']=int(ptr[2])//8
  elif kind=='tt.load':loads.append((args[0],args[1] if len(args)>1 else None))
  nodes[name]=data
 assert len(loads)==2 and store is not None
 return nodes,loads[0],store

def ir_eval(parsed,arguments,pid,lane):
 nodes,load,store=parsed
 @functools.lru_cache(None)
 def ev(name,index):
  if name in arguments:return arguments[name]
  d=nodes[name];o=d['op'];args=d['args'];sh=d['shape'];w=d['width']
  if o=='arith.constant':assert d['literal'] is not None;return d['literal']
  if o=='tt.get_program_id':return wrap(pid,32)
  if o=='tt.make_range':return index[0]
  if o=='tt.splat':return ev(args[0],())
  if o=='tt.expand_dims':return ev(args[0],index[:d['axis']]+index[d['axis']+1:])
  if o=='tt.broadcast':
   src=nodes[args[0]]['shape'];return ev(args[0],tuple(0 if n==1 else i for n,i in zip(src,index)))
  if o=='arith.extsi':return ev(args[0],index)
  values=[ev(a,index) for a in args]
  if o=='tt.addptr':return (values[0]+values[1]*d['element_bytes'])%2**64
  if o=='arith.cmpi':assert d['body'].startswith('slt,');return int(values[0]<values[1])
  if o=='arith.andi':return values[0]&values[1]
  a,b=values
  if o in ['arith.divsi','arith.remsi'] and (b==0 or (a==-2**(w-1) and b==-1)):raise Undefined('integer_division')
  z={'arith.addi':lambda:a+b,'arith.subi':lambda:a-b,'arith.muli':lambda:a*b,'arith.minsi':lambda:min(a,b),'arith.divsi':lambda:trunc(a,b),'arith.remsi':lambda:a-b*trunc(a,b)}[o]()
  return wrap(z,w)
 ix=(lane//128,lane%128)
 return {'mask':bool(ev(load[1],ix)),'input':ev(load[0],ix),'output':ev(store[0],ix),'store_mask':bool(ev(store[1],ix))}

checks=[];count=0;rng=random.Random(732104)
for record in receipt['records']:
 if record['status']!='compiled' or not record['name'].startswith('singleton_'):continue
 view=next(x for x in receipt['views'] if x['name']==record['name']);args=view['runtime_arguments'];path=P/Path(record['artifacts']['ttir']['path']).name
 parsed=parse(path);runtime={'%A':2**31,'%B':2**34,'%absmax_inv_ptr':2**35,**{'%'+k:v for k,v in args.items()}}
 active=[]
 for lane in range(128*128):
  s=source(args['M'],args['N'],args['stride_am'],args['stride_an'],0,lane);i=ir_eval(parsed,runtime,0,lane)
  assert all(i[k]==s[k] for k in ['mask','input','output']) and i['store_mask']==s['mask'],(record['name'],lane,s,i)
  if i['mask']:active.append((lane,i['input'],i['output']))
  count+=1
 assert active==[(0,2**31,2**34)]
 checks.append({'profile':record['name'],'config':record['config_index'],'lanes':16384,'sole_active':active,'ttir_sha256':sha(path)})

# Generic probe signatures are used as expression-tree checks, then coupled to
# reachable output metadata. Integer1 is evaluated with the source's constexpr
# rule only in dedicated runtime-specialized singleton IR above.
generic=[]
for name in ['mixed_M_i64','mixed_N_i64','mixed_stride_am_i64','mixed_stride_an_i64']:
 records=[r for r in receipt['records'] if r['name']==name]
 for record in records:
  signature=record['signature'];path=P/Path(record['artifacts']['ttir']['path']).name;parsed=parse(path)
  for trial in range(36):
   vals={k:(rng.choice([2**31,2**31+1,2**32-129,2**63-1]) if t=='i64' else rng.choice([2,127,128,129,1024,2**30,2**31-129,2**31-1])) for k,t in signature.items() if t in ['i32','i64']}
   M,N=vals['M'],vals['N'];vals['stride_bm']=1;vals['stride_bn']=M
   # Retained mixed_M generic IR gives stride_bn i32, unlike the wrapper when
   # M is i64. Skip that unreachable coupled signature; other three remain
   # useful reachable width patterns with unit-stride input.
   if name=='mixed_M_i64':continue
   if name=='mixed_stride_am_i64':vals['stride_an']=1
   else:vals['stride_am']=1
   pid=rng.choice([0,1,7,8,2**21-1,2**31-2]);lane=rng.choice([0,1,127,128,16383,256])
   args={'%A':2**31,'%B':2**34,'%absmax_inv_ptr':2**35,**{'%'+k:v for k,v in vals.items()}}
   try:s=source(M,N,vals['stride_am'],vals['stride_an'],pid,lane)
   except Undefined:
    try:ir_eval(parsed,args,pid,lane)
    except Undefined:pass
    else:raise AssertionError('source undefined but IR defined')
   else:
    i=ir_eval(parsed,args,pid,lane);assert all(i[k]==s[k] for k in ['mask','input','output']) and i['store_mask']==s['mask'],(name,vals,pid,lane,s,i)
   count+=1
  generic.append({'profile':name,'config':record['config_index'],'ttir_sha256':sha(path)})

safe=[source(5,1,2**30,1,0,l,base=2**31,width=1) for l in [0,128,256,384,512]]
assert [v['input'] for v in safe]==[2**31,3*2**30,0,2**30,2**31]
assert all(v['mask'] and 0<=v['input']<6*2**30+1 for v in safe)
assert len({v['output'] for v in safe})==5
unsafe=source(5,1,2**30,1,0,256,base=0,width=1);assert unsafe['input']==2**64-2**31
negative_mask=source(2**31-127,1,1,1,2**24-1,0,width=1)
assert negative_mask['rm']<0 and negative_mask['mask']
report={'status':'passed','reviewer':'peer:/root/matrix_transpose_acceptance','independent_review_not_acceptance':True,'native_gpu_executions':0,'method':'Evaluate preserved compiled TTIR integer SSA with explicit tensor index transforms and signed finite arithmetic; compare each address/mask to an independent source-expression evaluator. Only numerical arithmetic after loads is excluded.','peer_native_receipt_sha256':sha(P/'native-review-receipt.json'),'singleton_checks':checks,'generic_checks':generic,'evaluated_lanes':count,'safe_wrapped_example':safe,'unsafe_wrapped_address':unsafe,'limitations':'Generic mixed_M probe output-stride signature is not wrapper-reachable and is reviewed structurally rather than compared dynamically; LLVM optimization and native scalar execution are not validated.'}
report['negative_coordinate_still_masked_active']=negative_mask
(W/'address-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','evaluated_lanes':count,'singleton_profiles':len(checks),'generic_profiles':len(generic)}))
