"""Independent exact-rational oracle for the frozen finite two-block corpus.

This is deliberately a finite-input characterization oracle, not an IEEE
implementation for every NaN/Inf payload or a model of tensor-core ordering.
"""
from fractions import Fraction as Q
from functools import lru_cache

def need(ok,msg):
    if not ok:raise ValueError(msg)
def pow2(e):return Q(1<<e) if e>=0 else Q(1,1<<(-e))
@lru_cache(maxsize=8192)
def decode(bits,ebits,fbits):
    need(isinstance(bits,int) and 0<=bits<1<<(1+ebits+fbits),'invalid word')
    sign=-1 if bits>>(ebits+fbits) else 1
    exp=(bits>>fbits)&((1<<ebits)-1);frac=bits&((1<<fbits)-1);bias=(1<<(ebits-1))-1
    need(exp!=(1<<ebits)-1,'finite-corpus oracle does not accept NaN/Inf')
    return sign*Q(frac if exp==0 else (1<<fbits)+frac)*pow2((1-bias if exp==0 else exp-bias)-fbits)
def round_even(x):
    need(x>=0,'positive rounding input expected');n,r=divmod(x.numerator,x.denominator)
    return n+(2*r>x.denominator or (2*r==x.denominator and n%2==1))
@lru_cache(maxsize=8192)
def rounded(value,ebits,fbits):
    x=Q(value);sign=int(x<0);x=abs(x);bias=(1<<(ebits-1))-1
    if x==0:return 0
    emin=1-bias;emax=bias
    e=x.numerator.bit_length()-x.denominator.bit_length()
    if x<pow2(e):e-=1
    if e<emin:
        sig=round_even(x/pow2(emin-fbits))
        result=sig if sig<(1<<fbits) else 1<<fbits
    else:
        sig=round_even(x/pow2(e-fbits))
        if sig==1<<(fbits+1):sig>>=1;e+=1
        result=(((1<<ebits)-1)<<fbits) if e>emax else ((e+bias)<<fbits)+(sig-(1<<fbits))
    return (sign<<(ebits+fbits))|result
def half(bits):return decode(bits,5,10)
def single(bits):return decode(bits,8,23)
def rhalf(x):return rounded(x,5,10)
def r32(x):return rounded(x,8,23)
def i32(x):return (x&0xffffffff)-(1<<32) if x&0x80000000 else x&0xffffffff

PROFILES=('source_separate_dot_add','compiled_separate_ascending_fma','compiled_fused_ascending_fma')

@lru_cache(maxsize=65536)
def _dequant(weight,zero,scale,fused):
    z=rhalf(half(rhalf(zero))*half(scale))
    return rhalf(half(rhalf(weight))*half(scale)-half(z)) if fused else rhalf(half(rhalf(half(rhalf(weight))*half(scale)))-half(z))

@lru_cache(maxsize=65536)
def _dot_output(a,b,profile):
    carry=0
    for step in range(2):
        if profile=='source_separate_dot_add':
            block=0
            for k in range(32):
                off=step*32+k
                block=r32(single(block)+single(r32(half(a[off])*half(b[off]))))
            carry=r32(single(carry)+single(block))
        else:
            for k in range(32):
                off=step*32+k;carry=r32(half(a[off])*half(b[off])+single(carry))
    return rhalf(single(carry))

def expected(case,profile):
    need(profile in PROFILES,'unknown software graph')
    cfg=case['launch']['constexprs'];args=case['runtime_args'];alloc=case['allocations']
    need((args['M'],args['N'],args['K'])==(64,128,64),'outside frozen legal case')
    need((cfg['BLOCK_SIZE_M'],cfg['BLOCK_SIZE_N'],cfg['BLOCK_SIZE_K'],cfg['GROUP_SIZE_M'])==(64,128,32,8),'wrong original tile')
    need(args['groupsize']==(64 if cfg['NO_GROUPS'] else 32),'NO_GROUPS mismatch')
    def read(name,offset):
        a=alloc[name];idx=a['logical_origin_elements']+offset
        need(0<=idx<len(a['raw_words']),'read outside allocation')
        return a['raw_words'][idx]
    columns=[]
    for col in range(128):
        values=[]
        for step in range(2):
            group=0 if cfg['NO_GROUPS'] else step//(args['groupsize']//32)
            scale=read('scales_ptr',col*args['stride_scales_n']+group*args['stride_scales_g'])
            zero=(i32(read('zeros_ptr',(col//8)*args['stride_zeros_n']+group*args['stride_zeros_g']))>>((col%8)*4))&15
            for k in range(32):
                offset=(k//8)*args['stride_bk']+col*args['stride_bn']+step*(32//8*args['stride_bk'])
                weight=(i32(read('b_ptr',offset))>>((k%8)*4))&15
                values.append(_dequant(weight,zero,scale,profile=='compiled_fused_ascending_fma'))
        columns.append(tuple(values))
    out=list(alloc['c_ptr']['raw_words']);origin=alloc['c_ptr']['logical_origin_elements']
    for row in range(64):
        a=tuple(read('a_ptr',row*args['stride_am']+k*args['stride_ak']+step*32*args['stride_ak']) for step in range(2) for k in range(32))
        for col in range(128):out[origin+row*args['stride_cm']+col*args['stride_cn']]=_dot_output(a,columns[col],profile)
    return out

def compare(case,before,after,references,selected):
    need(selected in PROFILES and set(references)==set(PROFILES),'a preselected supported profile is required')
    need(set(before)==set(after)==set(case['allocations']),'incomplete allocation capture')
    for name,a in case['allocations'].items():
        need(before[name]==a['raw_words'],'before differs from bound initial memory')
        need(len(after[name])==len(before[name]),'allocation length changed')
        need(all(isinstance(x,int) and 0<=x<1<<a['word_bits'] for x in after[name]),'invalid raw word')
    changed=[{'allocation':name,'offset':i,'before':x,'after':y} for name in before if name!='c_ptr'
      for i,(x,y) in enumerate(zip(before[name],after[name])) if x!=y]
    origin=case['allocations']['c_ptr']['logical_origin_elements'];end=origin+8192
    frame=[{'offset':i,'before':x,'after':y} for i,(x,y) in enumerate(zip(before['c_ptr'],after['c_ptr'])) if not(origin<=i<end) and x!=y]
    matches={name:after['c_ptr']==v and not changed for name,v in references.items()}
    return {'status':'CHARACTERIZATION','profile_matches_on_fixed_case':matches,'input_mutations':changed,
      'output_frame_violations':frame,'compared_C_words':len(after['c_ptr']),'active_C_words':8192,
      'selected_profile':selected,'selected_profile_result':'MATCH_ON_FIXED_CASE' if matches[selected] else 'MISMATCH_ON_FIXED_CASE',
      'universal_native_equality_proved':False,'numeric_acceptance':False,
      'limitation':'This finite case discriminates scalar half fusion; it does not distinguish the two unfused dot graphs or prove hardware reduction order.'}
