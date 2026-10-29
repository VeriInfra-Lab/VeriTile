"""Independent exact-rational oracle for the frozen finite two-block corpus.

This is deliberately a finite-input characterization oracle, not an IEEE
implementation for every NaN/Inf payload or a model of tensor-core ordering.
"""
from fractions import Fraction as Q

def need(ok,msg):
    if not ok:raise ValueError(msg)
def pow2(e):return Q(1<<e) if e>=0 else Q(1,1<<(-e))
def decode(bits,ebits,fbits):
    need(isinstance(bits,int) and 0<=bits<1<<(1+ebits+fbits),'invalid word')
    sign=-1 if bits>>(ebits+fbits) else 1
    exp=(bits>>fbits)&((1<<ebits)-1);frac=bits&((1<<fbits)-1);bias=(1<<(ebits-1))-1
    need(exp!=(1<<ebits)-1,'finite-corpus oracle does not accept NaN/Inf')
    return sign*Q(frac if exp==0 else (1<<fbits)+frac)*pow2((1-bias if exp==0 else exp-bias)-fbits)
def round_even(x):
    need(x>=0,'positive rounding input expected');n,r=divmod(x.numerator,x.denominator)
    return n+(2*r>x.denominator or (2*r==x.denominator and n%2==1))
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

def expected(case,profile):
    need(profile in ['ascending_fma','descending_fma','source_separate_dot_add'],'unknown graph')
    cfg=case['launch']['constexprs'];args=case['runtime_args'];alloc=case['allocations']
    need(args['M']==args['N']==1 and args['K']==64 and cfg['BLOCK_SIZE_K']==32 and cfg['SPLIT_K']==1,'outside bound finite corpus')
    def read(name,offset):
        a=alloc[name];idx=a['logical_origin_elements']+offset
        need(0<=idx<len(a['raw_words']),'out of allocation')
        return a['raw_words'][idx]
    carry=0
    for step in range(2):
        products=[]
        for k in range(32):
            ak=k*args['stride_ak']+step*32*args['stride_ak']
            bk=(k//8)*args['stride_bk']+step*(32*args['stride_bk']//8)
            group=(k+step*32)//args['group_size']
            word=read('b_ptr',bk);zero=read('bzp_ptr',group*args['stride_bzpk'])
            nibble=(i32(word)>>(k%8*4))&15;zp=i32(zero)&15
            diff=nibble-zp
            b=rhalf(half(rhalf(diff))*half(read('bs_ptr',group*args['stride_bsk'])))
            products.append((half(read('a_ptr',ak)),half(b)))
        if profile=='source_separate_dot_add':
            block=0
            for a,b in products:block=r32(single(block)+single(r32(a*b)))
            carry=r32(single(carry)+single(block))
        else:
            if profile=='descending_fma':products.reverse()
            for a,b in products:carry=r32(a*b+single(carry))
    out=list(alloc['c_ptr']['raw_words']);out[alloc['c_ptr']['logical_origin_elements']]=rhalf(single(carry));return out

def compare(case,before,after,references,selected=None):
    need(set(before)==set(after)==set(case['allocations']),'incomplete allocation capture')
    for name,a in case['allocations'].items():
        need(before[name]==a['raw_words'],'before allocation differs from bound input')
        need(len(after[name])==len(before[name]),'allocation length changed')
    origin=case['allocations']['c_ptr']['logical_origin_elements']
    changed_inputs=[{'allocation':name,'offset':i,'before':x,'after':y}
        for name in before if name!='c_ptr' for i,(x,y) in enumerate(zip(before[name],after[name])) if x!=y]
    frame=[{'offset':i,'before':x,'after':y} for i,(x,y) in enumerate(zip(before['c_ptr'],after['c_ptr'])) if i!=origin and x!=y]
    matches={name:after['c_ptr']==ref and not changed_inputs for name,ref in references.items()}
    need(selected is None or selected in matches,'unknown selected profile')
    return {'status':'CHARACTERIZATION','output_word':after['c_ptr'][origin],
      'profile_matches_on_fixed_case':matches,'input_mutations':changed_inputs,'output_frame_violations':frame,
      'selected_profile':selected,'selected_profile_result':None if selected is None else ('MATCH_ON_FIXED_CASE' if matches[selected] else 'MISMATCH_ON_FIXED_CASE'),
      'universal_native_equality_proved':False,'numeric_acceptance':False}
