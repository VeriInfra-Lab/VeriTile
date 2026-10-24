"""Finite experiment contracts. No numerical approximation replaces pow/sin/cos."""
PROFILES={
 'wrapper_default':{'THETA':10000.,'BLOCK_SIZE_M':2,'BLOCK_SIZE_K':1024},
 'old_backend_witness':{'THETA':10000.,'BLOCK_SIZE_M':1,'BLOCK_SIZE_K':2},
 'nonperiod_tile':{'THETA':10000.,'BLOCK_SIZE_M':2,'BLOCK_SIZE_K':64},
 'period_tile':{'THETA':10000.,'BLOCK_SIZE_M':4,'BLOCK_SIZE_K':128}}
SPECIAL_WORDS=[0,0x8000,0x3c00,0xbc00,1,0x8001,0x7bff,0xfbff,0x7c00,0xfc00,0x7e55,0xfe01,0x7d01,0x3555]
def need(ok,msg):
    if not ok:raise ValueError(msg)
def ceildiv(a,b):return (a+b-1)//b

def footprint(c):
    a=c['runtime_args'];bm=c['constexprs']['BLOCK_SIZE_M'];bk=c['constexprs']['BLOCK_SIZE_K'];batch=c['batch']
    parts=ceildiv(a['K'],bk);grid=[batch,ceildiv(a['M'],bm)*parts,1]
    need(c['grid']==grid,'grid differs from original host launch')
    accesses=[];writes={};masked={'real':0,'imag':0,'real_only_pairs':0}
    for b in range(batch):
      for pid in range(grid[1]):
       pm,pn=divmod(pid,parts)
       for i in range(bm):
        row=pm*bm+i
        for j in range(bk//2):
         col=pn*bk+2*j
         real=row<a['M'] and col<a['K'];imag=row<a['M'] and col+1<a['K']
         if real and not imag:masked['real_only_pairs']+=1
         for component,live in [('real',real),('imag',imag)]:
          delta=int(component=='imag')
          if not live:masked[component]+=1;continue
          # The original imaginary pointer is +1, not +stride_*_n.
          xi=b*a['stride_x_batch']+row*a['stride_x_m']+col*a['stride_x_n']+delta
          yi=b*a['stride_out_batch']+row*a['stride_out_m']+col*a['stride_out_n']+delta
          xp=c['pointers']['x_ptr'];yp=c['pointers']['out_ptr'];x=xp['origin_elements']+xi;y=yp['origin_elements']+yi
          need(0<=x<len(c['allocations'][xp['allocation']]['raw_words']),'active input out of allocation')
          need(0<=y<len(c['allocations'][yp['allocation']]['raw_words']),'active output out of allocation')
          key=(yp['allocation'],y);need(key not in writes,'sample has inter/intra-program output collision')
          item={'program':[b,pid],'lane':[i,j],'row':row,'column':col,'component':component,'input_allocation':xp['allocation'],'input_word':x,'output_allocation':yp['allocation'],'output_word':y}
          accesses.append(item);writes[key]=item
    return {'grid':grid,'active_components':len(accesses),'active_pairs_real':sum(x['component']=='real' for x in accesses),
      'active_pairs_imag':sum(x['component']=='imag' for x in accesses),'masked_lanes':masked,'accesses':accesses,
      'write_offsets':{n:sorted(o for name,o in writes if name==n) for n in c['allocations']},
      'scope':'Finite launch footprint only; old per-program theorem and arbitrary Nat/Real parameters remain unchanged.'}

def one_case(cid,profile,batch,rows,cols,pos,pattern='finite',strided=False,inplace=False):
    k=PROFILES[profile];xs=3 if strided else 1;ys=5 if strided else 1
    xm=cols*xs+(7 if strided else 0);ym=cols*ys+(11 if strided else 0)
    xb=rows*xm+(13 if strided else 0);yb=rows*ym+(17 if strided else 0)
    args={'M':rows,'K':cols,'stride_x_batch':xb,'stride_x_m':xm,'stride_x_n':xs,
      'stride_out_batch':yb,'stride_out_m':ym,'stride_out_n':ys,'start_token_position':pos}
    need(not inplace or (xs,xm,xb)==(ys,ym,yb),'in-place sample layout mismatch')
    size_x=batch*xb+16;size_y=batch*yb+16
    def word(i):
        if pattern=='all_payload_classes':return SPECIAL_WORDS[i%len(SPECIAL_WORDS)]
        return [0x3c00,0x3800,0xbc00,0x3555,0x8000,0x0001,0x7bff][i%7]
    allocations={'x':{'word_bits':16,'raw_words':[word(i) for i in range(size_x)]}}
    if not inplace:allocations['out']={'word_bits':16,'raw_words':[0x7e55]*size_y}
    c={'case_id':cid,'profile_id':profile,'batch':batch,'runtime_args':args,'constexprs':dict(k),
      'grid':[batch,ceildiv(rows,k['BLOCK_SIZE_M'])*ceildiv(cols,k['BLOCK_SIZE_K']),1],
      'allocations':allocations,'pointers':{'x_ptr':{'allocation':'x','origin_elements':8},'out_ptr':{'allocation':'x' if inplace else 'out','origin_elements':8}},
      'input_pattern':pattern,'native_numeric_reference':'NOT_ESTABLISHED','finite_input_assumption_added_to_formal_theorem':False}
    c['footprint']=footprint(c);return c

def cases():
    out=[one_case('wrapper_pos'+str(pos),'wrapper_default',2,4,1024,pos) for pos in range(4)]
    out += [one_case('odd_tail_all_payloads','wrapper_default',2,3,1025,1,'all_payload_classes'),
      one_case('strided_dual_mask','nonperiod_tile',2,3,129,1,'all_payload_classes',True),
      one_case('period_large_position','period_tile',1,5,129,16777217),
      one_case('inplace_same_storage','nonperiod_tile',1,3,129,3,'all_payload_classes',inplace=True),
      one_case('signed_position_wrap','nonperiod_tile',1,3,3,2147483647),
      one_case('old_backend_witness','old_backend_witness',1,1,2,1)]
    return {c['case_id']:c for c in out}

def check_capture(c,before,after):
    need(set(before)==set(after)==set(c['allocations']),'incomplete allocations')
    fp=footprint(c);need(fp==c['footprint'],'footprint drift')
    frame=[];changed=[]
    for name,a in c['allocations'].items():
        need(before[name]==a['raw_words'],'initial raw words differ')
        need(len(after[name])==len(before[name]) and all(type(x) is int and 0<=x<65536 for x in after[name]),'invalid/incomplete raw words')
        write=set(fp['write_offsets'][name])
        for i,(x,y) in enumerate(zip(before[name],after[name])):
            if x!=y:
                changed.append({'allocation':name,'word':i,'before':x,'after':y})
                if i not in write:frame.append({'allocation':name,'word':i,'before':x,'after':y})
    outputs=[{'program':x['program'],'lane':x['lane'],'component':x['component'],'allocation':x['output_allocation'],
      'word':x['output_word'],'before':before[x['output_allocation']][x['output_word']],'after':after[x['output_allocation']][x['output_word']]} for x in fp['accesses']]
    return {'status':'CHARACTERIZATION_NUMERICS_UNBOUND','numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED',
      'active_output_words':outputs,'all_allocations_words':sum(len(x) for x in after.values()),'changed_words':changed,
      'frame_violations':frame,'frame_check':'UNCHANGED_OUTSIDE_FOOTPRINT' if not frame else 'VIOLATION',
      'limitation':'No closed native pow/sin/cos/division/fusion profile is supplied. Capturing all active outputs and a matching frame does not establish output correctness.'}
