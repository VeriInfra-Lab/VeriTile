"""Integer-only independent e2m1 -> bf16 reference; high nibble first."""
MAGNITUDE_BITS = (0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0)

def uint(value, width):
    if type(value) is not int or not 0 <= value < 2**width:
        raise ValueError(f'Expected unsigned {width}-bit word')
    return value

def decode_nibble(n):
    uint(n, 4)
    return MAGNITUDE_BITS[n & 7] | ((n & 8) << 12)

def decode_byte(b):
    uint(b, 8)
    return [decode_nibble(b >> 4), decode_nibble(b & 15)]

def expected(case):
    count, block, grid = case['count'], case['block'], case['grid']
    if any(type(x) is not int for x in (count, block, grid)):
        raise ValueError('Integer launch parameters required')
    if count < 0 or block not in (16, 512) or grid != (count + block - 1)//block + 1:
        raise ValueError('Profile requires block16/512 and one extra fully masked program')
    if max(2*count, 2*grid*block) >= 2**31:
        raise ValueError('Profile excludes signed-i32 address overflow')
    ip, op = case['input_origin'], case['output_origin']
    if any(type(x) is not int or x < 1 for x in (ip, op)):
        raise ValueError('Nonzero independent allocation origins required')
    src, initial = case['input_words'], case['output_initial_words']
    for b in src: uint(b, 8)
    for b in initial: uint(b, 16)
    if len(src) < ip+count+1 or len(initial) < op+2*grid*block+1:
        raise ValueError('Missing allocation guards')
    out = list(initial)
    for i in range(count): out[op+2*i:op+2*i+2] = decode_byte(src[ip+i])
    return out

def cases():
    result=[]
    for block in (16,512):
        counts=sorted({0,1,block-1,block,block+1,255,256,257,511,512,513,3*block+7})
        for count in counts:
            grid=(count+block-1)//block+1
            row={'id':f'b{block}-n{count}', 'count':count, 'block':block, 'grid':grid,
                 'input_origin':17,'output_origin':19,
                 'allocation_relation':'separate_disjoint_input_and_output',
                 'input_words':[(i*73+19)%256 for i in range(17+count+23)],
                 'output_initial_words':[(0xd000+i*29)%65536 for i in range(19+2*grid*block+23)]}
            # Every byte occurs in order when count >= 256; signs of zero stay distinct.
            row['input_words'][17:17+count]=[i%256 for i in range(count)]
            row['expected_output_words']=expected(row)
            result.append(row)
    return result

def compare(case, input_after, output_after):
    want=expected(case)
    if len(input_after)!=len(case['input_words']) or len(output_after)!=len(want):
        raise ValueError('Native allocation length differs')
    for b in input_after:uint(b,8)
    for b in output_after:uint(b,16)
    changes=[]
    for i,(a,b) in enumerate(zip(case['input_words'],input_after)):
        if a!=b:changes.append({'region':'input','offset':i,'expected':a,'actual':b,'kind':'frame'})
    lo,hi=case['output_origin'],case['output_origin']+2*case['count']
    for i,(a,b) in enumerate(zip(want,output_after)):
        if a!=b:changes.append({'region':'output','offset':i,'expected':a,'actual':b,
                                'kind':'value' if lo<=i<hi else 'frame'})
    return changes
