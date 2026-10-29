"""Finite full-launch contracts; unspecified block padding is never filled by zero."""
import struct
POINTERS=('q','k','v','o','do','dq','dk','dv')
def need(x,msg):
    if not x:raise ValueError(msg)
def ceildiv(a,b):return (a+b-1)//b
def f32bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def constants(case,stage):
    return {'BTL':128 if stage=='forward' else 64,'BTS':32,'BK':case['BK'],'BV':case['BV'],'DK':case['DK'],'DV':case['DV']}
def cases():
    specs=[('original_test_shape',2,4,128,64,64,False,False),('full_two_forward_chunks',1,2,256,64,64,False,False),
      ('ragged_time_special',1,2,129,64,64,False,True),('ragged_split_heads',1,2,65,160,96,False,True),('strided',1,2,96,64,64,True,False)]
    out={}
    for name,B,H,T,DK,DV,strided,special in specs:
        BK=min(128,1<<(DK-1).bit_length());BV=min(128,1<<(DV-1).bit_length());q_t=DK+(8 if strided else 0);v_t=DV+(12 if strided else 0);q_h=T*q_t+(16 if strided else 0);v_h=T*v_t+(16 if strided else 0)
        nk=ceildiv(DK,BK);nv=ceildiv(DV,BV);pattern=[0,0x80000000,0x3e800000,0xbe000000,0x3f000000,1,0x80000001]
        if special:pattern += [0x7f800000,0xff800000,0x7fc01234,0xffc05678,0x7f800001]
        a={'s_qk_h':q_h,'s_qk_t':q_t,'s_qk_d':1,'s_vo_h':v_h,'s_vo_t':v_t,'s_vo_d':1,'B':B,'H':H,'T':T,'scale':struct.unpack('<f',struct.pack('<f',DK**-0.5))[0]}
        allocations={}
        sizes={'q':B*H*q_h,'k':B*H*q_h,'v':B*H*v_h,'do':B*H*v_h,'o':nk*B*H*v_h,'dq':nv*B*H*q_h,'dk':nv*B*H*q_h,'dv':nk*B*H*v_h}
        for j,(region,size) in enumerate(sizes.items()):allocations[region]={'word_bits':32,'origin_elements':8,'raw_words':[pattern[(i+j)%len(pattern)] if region in ('q','k','v','do') else 0x7fc0abcd for i in range(size+16)]}
        out[name]={'case_id':name,'B':B,'H':H,'T':T,'DK':DK,'DV':DV,'BK':BK,'BV':BV,'NK':nk,'NV':nv,'runtime_args':a,'scale_bits':f32bits(a['scale']),
          'options':{'num_warps':4,'num_stages':3 if DK<=64 else 2},'allocations':allocations,'special_payloads':special,
          'padding_semantics':'unspecified; no replacement values are supplied by this harness','numeric_comparison':'NOT_ESTABLISHED'}
    return out

def launch_plan(case,stage):
    need(stage in ('forward','backward_helper','backward_shell'),'unknown stage');k=constants(case,stage);a=case['runtime_args'];grid=[case['NK']*case['NV'],ceildiv(case['T'],k['BTL']),case['B']*case['H']]
    if stage=='forward':inputs=['q','k','v'];outputs=['o']
    elif stage=='backward_helper':inputs=['q','k','v','do'];outputs=['dk','dv']
    else:inputs=['q','k','v','do'];outputs=['dq','dk','dv']
    calls=[];programs=[];writes={n:set() for n in outputs};loads=[]
    def load(region,bh,rows,cols,sr,sc,r0,c0,br,bc,reason,program):
        validr=max(0,min(br,rows-r0));validc=max(0,min(bc,cols-c0));invalid=br*bc-validr*validc
        loads.append({'region':region,'program':program,'base':bh*(a['s_qk_h'] if region in ['q','k'] else a['s_vo_h']),
          'parent_shape':[rows,cols],'strides':[sr,sc],'offsets':[r0,c0],'block_shape':[br,bc],'invalid_lanes':invalid,
          'padding':'unspecified','phase':reason})
    for bh in range(grid[2]):
      for ic in range(grid[1]):
       for ikv in range(grid[0]):
        ik,iv=divmod(ikv,case['NV']);program=[ikv,ic,bh];start=ic*k['BTL']
        before=list(range(0,start,k['BTS'])) if stage=='forward' else list(range(ceildiv(case['T'],k['BTS'])*k['BTS']-k['BTS'],(ic+1)*k['BTL']-k['BTS'],-k['BTS']))
        diagonal=list(range(start,(ic+1)*k['BTL'],k['BTS']))
        programs.append({'program':program,'i_k':ik,'i_v':iv,'i_h':bh%case['H'],'offdiagonal_order':before,'diagonal_order':diagonal})
        if stage=='backward_helper':calls.append({'i_bh':bh,'i_c':ic,'i_k':ik,'i_v':iv,'i_h':bh%case['H']})
        if stage=='forward':
            load('q',bh,case['T'],case['DK'],a['s_qk_t'],1,start,ik*k['BK'],k['BTL'],k['BK'],'query',program)
            for phase,starts in [('offdiagonal',before),('diagonal',diagonal)]:
                for j in starts:
                    load('k',bh,case['DK'],case['T'],1,a['s_qk_t'],ik*k['BK'],j,k['BK'],k['BTS'],phase,program)
                    load('v',bh,case['T'],case['DV'],a['s_vo_t'],1,j,iv*k['BV'],k['BTS'],k['BV'],phase,program)
        else:
            load('k',bh,case['T'],case['DK'],a['s_qk_t'],1,start,ik*k['BK'],k['BTL'],k['BK'],'local_key',program)
            load('v',bh,case['T'],case['DV'],a['s_vo_t'],1,start,iv*k['BV'],k['BTL'],k['BV'],'local_value',program)
            for phase,starts in [('descending',before),('diagonal',diagonal)]:
                for j in starts:
                    load('q',bh,case['DK'],case['T'],1,a['s_qk_t'],ik*k['BK'],j,k['BK'],k['BTS'],phase,program)
                    load('do',bh,case['DV'],case['T'],1,a['s_vo_t'],iv*k['BV'],j,k['BV'],k['BTS'],phase,program)
        for out in outputs:
            qtype=out in ('dq','dk');cols=case['DK'] if qtype else case['DV'];block=k['BK'] if qtype else k['BV'];column=(ik*k['BK']) if qtype else (iv*k['BV']);partial=iv if qtype else ik
            h=a['s_qk_h'] if qtype else a['s_vo_h'];t=a['s_qk_t'] if qtype else a['s_vo_t'];base=(bh+case['B']*case['H']*partial)*h+8
            for i in range(k['BTL']):
                for j in range(block):
                    if start+i<case['T'] and column+j<cols:
                        off=base+(start+i)*t+column+j;need(off not in writes[out],'overlapping sample output');need(off<len(case['allocations'][out]['raw_words']),'output out of allocation');writes[out].add(off)
    return {'grid':grid,'helper_calls':calls,'programs':programs,'input_regions':inputs,'output_regions':outputs,'write_offsets':{n:sorted(o) for n,o in writes.items()},
      'load_descriptors':loads,'undefined_load_lanes':sum(x['invalid_lanes'] for x in loads),
      'load_descriptor_scope':'forward or accepted DK/DV helper. Shell has extra DQ reads which are outside accepted proof; shell store footprints include DQ.',
      'host_reduction_performed':False,'accepted_scope':stage!='backward_shell'}
def check_capture(case,stage,before,after):
    plan=launch_plan(case,stage);names=set(plan['input_regions']+plan['output_regions']);need(set(before)==set(after)==names,'incomplete full stage allocation capture')
    frame=[];changed_counts={};active={}
    for n in names:
        need(before[n]==case['allocations'][n]['raw_words'],'initial bits differ');need(len(after[n])==len(before[n]) and all(type(x) is int and 0<=x<(1<<32) for x in after[n]),'invalid fp32 raw capture')
        written=set(plan['write_offsets'].get(n,[]));changed_counts[n]=0
        for i,(a,b) in enumerate(zip(before[n],after[n])):
            if a!=b:
                changed_counts[n]+=1
                if i not in written:frame.append({'region':n,'word':i,'before':a,'after':b})
        if n in plan['output_regions']:active[n]={'word_offsets':plan['write_offsets'][n],'raw_words':[after[n][i] for i in plan['write_offsets'][n]]}
    return {'status':'CHARACTERIZATION_NUMERICS_UNBOUND','numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED','accepted_stage':plan['accepted_scope'],
      'active_outputs':active,'changed_word_counts':changed_counts,'frame_violations':frame,'undefined_load_lanes':plan['undefined_load_lanes'],
      'undefined_values_measured':False,'host_partial_reduction_performed':False,
      'limitation':'All original loop contributions and tail arithmetic execute. Masked block-pointer padding is unspecified, not zero or captured global memory. This record cannot assign a deterministic numeric reference to undefined lanes.'}
