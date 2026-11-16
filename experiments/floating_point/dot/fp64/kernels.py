"""FP64 dot/add source pair with a double-double reference observer."""
import triton
import triton.language as tl

@triton.jit
def separate_dot(A,B,T):
 g=tl.program_id(0);i=tl.arange(0,32)
 a=tl.load(A+g*1024+i[:,None]*32+i[None,:])
 b=tl.load(B+g*1024+i[:,None]*32+i[None,:])
 p=tl.dot(a,b,input_precision='ieee',out_dtype=tl.float64)
 tl.store(T+g*1024+i[:,None]*32+i[None,:],p)

@triton.jit
def separate_add(T,C,P):
 x=tl.program_id(0)*1024+tl.arange(0,1024)
 tl.store(P+x,tl.load(T+x)+tl.load(C+x))

@triton.jit
def seeded_dot(A,B,C,Q):
 g=tl.program_id(0);i=tl.arange(0,32)
 a=tl.load(A+g*1024+i[:,None]*32+i[None,:])
 b=tl.load(B+g*1024+i[:,None]*32+i[None,:])
 c=tl.load(C+g*1024+i[:,None]*32+i[None,:])
 q=tl.dot(a,b,c,input_precision='ieee',out_dtype=tl.float64)
 tl.store(Q+g*1024+i[:,None]*32+i[None,:],q)

@triton.jit
def observer(A,B,C,P,Q,OHI,OLO,U,R,SAVE:tl.constexpr):
 g=tl.program_id(0);x=tl.arange(0,1024);i=x//32;j=x%32
 hi=tl.full((1024,),0.,tl.float64);lo=tl.full((1024,),0.,tl.float64)
 for k in range(32):
  a=tl.load(A+g*1024+i*32+k);b=tl.load(B+g*1024+k*32+j)
  ph=a*b;pl=tl.fma(a,b,-ph)
  s=hi+ph;v=s-hi;e=(hi-(s-v))+(ph-v)
  l=(lo+pl)+e
  hi=s+l;lo=l-(hi-s)
 c=tl.load(C+g*1024+x)
 s=hi+c;v=s-hi;e=(hi-(s-v))+(c-v)
 l=lo+e
 hi=s+l;lo=l-(hi-s)
 p=tl.load(P+g*1024+x);q=tl.load(Q+g*1024+x)
 exp=((tl.abs(hi).to(tl.uint64,bitcast=True)>>52)&2047).to(tl.int32)
 ub=tl.where(exp>52,(exp-52).to(tl.uint64)<<52,tl.full((1024,),1,tl.uint64)<<tl.maximum(exp-1,0))
 u=ub.to(tl.float64,bitcast=True)
 delta=(q-p)/u
 ep=tl.abs((p-hi)-lo);eq=tl.abs((q-hi)-lo)
 valid=tl.sum(((tl.abs(p)<float('inf'))&(tl.abs(q)<float('inf'))&(tl.abs(hi)<float('inf'))).to(tl.int32),0)==1024
 tl.store(R+g*4,tl.where(valid,tl.sum(delta,0)/1024,0.))
 tl.store(R+g*4+1,tl.where(valid,tl.max(ep,0),0.));tl.store(R+g*4+2,tl.where(valid,tl.max(eq,0),0.));tl.store(R+g*4+3,valid.to(tl.float64))
 if SAVE:
  tl.store(OHI+g*1024+x,hi);tl.store(OLO+g*1024+x,lo);tl.store(U+g*1024+x,u)
