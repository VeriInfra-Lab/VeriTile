#!/usr/bin/env python3
"""Source-specific CUDA FP32 sum selector; no tensor values or numerical oracle.

Input contains the actual ordered, 32-bit TensorIterator subiterators that visit
one logical output cell. Device/config/address fields are ordinary runtime ABI
metadata. This program does not claim to capture them from an unobserved run.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from math import prod
from pathlib import Path

PINNED = {
 '/usr/local/lib/python3.12/site-packages/torch/include/ATen/native/cuda/Reduce.cuh': 'ef9bea8fdfd03cec2d7e7421cbe75d98fddfdaafddb4f25a1ed655c5861dfb65',
 '/usr/local/lib/python3.12/site-packages/torch/include/ATen/native/SharedReduceOps.h': '75a6bad9a3c7f47572ab8264fd66487697e2a77bb9fba1b0d555d967aac18b5e',
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def divup(n, d):
    return (n+d-1)//d

def floorpow2(n):
    return 1 << (max(1,n).bit_length()-1)

@dataclass(frozen=True)
class Launch:
    first: int
    inputs: int
    width: int
    height: int
    reduceX: bool
    reduceY: bool
    ctas: int
    vectorInput: bool
    alignmentShift: int
    accumulate: bool

    @property
    def cta_stride(self):
        return (self.width if self.reduceX else 1)*(self.height if self.reduceY else 1)

    @property
    def step(self):
        return self.cta_stride*self.ctas

    def start(self,x,y,c):
        return (x if self.reduceX else 0)+(y*(self.width if self.reduceX else 1) if self.reduceY else 0)+c*self.cta_stride

    def sane(self):
        return self.inputs>0 and self.width==floorpow2(self.width) and self.width>0 and self.height==floorpow2(self.height) and self.height>0 and self.width*self.height<=512 and self.ctas>0 and self.step<=self.inputs and 0<=self.alignmentShift<4 and (self.ctas==1 or self.reduceY) and (not self.vectorInput or (self.reduceX and self.inputs>=128))


def select_config(meta):
    """Literal integer selector from pinned setReduceConfig<float,float,4,4>."""
    shape=meta['shape']; strides=meta['input_strides_bytes']; r=meta['num_reduce_dims']
    require(len(shape)==len(strides) and all(type(n) is int and n>0 for n in shape), 'nonempty TensorIterator shape')
    require(all(type(n) is int and n>=0 and n%4==0 for n in strides), 'FP32 byte strides')
    require((r==1 and len(shape)>=1) or r==0, 'at most one source reduction dimension after singleton coalescing')
    n=prod(shape[:r]); outputs=prod(shape[r:])
    require(prod(shape)<2**31, 'actual subiterator numel must fit int32')
    require(sum((n-1)*st for n,st in zip(shape,strides))<2**31, 'actual subiterator input byte offsets must fit int32')
    base=meta['input_base_word']; row=meta['output_linear']
    require(type(base) is int and base>=0 and type(row) is int and 0<=row<outputs, 'actual input pointer/output coordinate')
    rem=row; row_offset=0
    for dim,st in zip(shape[r:],strides[r:]):
        row_offset+=(rem%dim)*(st//4); rem//=dim
    fastest=not shape or r==len(shape) or strides[0]<strides[r]
    if shape:
        dim0,dim1=(n,outputs) if fastest else (outputs,n)
        fast_stride=strides[0 if fastest else r]
    else:
        dim0=dim1=1; fast_stride=4
    vector=False; outvec=1
    if fast_stride==4:
        if fastest and dim0>=128 and r==1:
            vector=True; dim0//=4
        elif not fastest:
            outvec=4
            alignment=[base,shape[r]]+[st//4 for j,st in enumerate(strides) if j!=r]
            for value in alignment:
                while value%outvec:
                    outvec//=2
            dim0//=outvec
    max_threads=512//outvec
    p0=min(floorpow2(dim0),max_threads); p1=min(floorpow2(dim1),max_threads)
    width=min(p0,32); height=min(p1,max_threads//width); width=min(p0,max_threads//height)
    step_in=width if fastest else 1; step_out=1 if fastest else width
    reduce_y=divup(n,step_in)>=min(height*16,256)
    if reduce_y: step_in*=height
    else: step_out*=height
    device=meta['device']; mp=device['multiprocessors']; mt=device['max_threads_per_mp']
    require(type(mp) is int and mp>0 and type(mt) is int and mt>=width*height, 'ordinary CUDA device properties')
    target=mp*(mt//(width*height)); grid=divup(outputs//outvec,step_out)
    ctas=1
    if reduce_y and divup(n,step_in)>=256 and grid<=target:
        values=divup(n,step_in)
        ctas=max(min(divup(target,grid),divup(values,16)),divup(values,256))
    selected={
        'block_width':width,'block_height':height,'num_threads':width*height,
        'num_inputs':n,'num_outputs':outputs,'vectorize_input':vector,'output_vec_size':outvec,
        'ctas_per_output':ctas,'step_input':step_in*ctas,'step_output':step_out,
        'input_mult':[1 if fastest else 0,(width if fastest else 1) if reduce_y else 0,step_in if ctas>1 else 0],
    }
    if 'config' in meta:
        require(meta['config']==selected,'captured ReduceConfig differs from source selector')
    d=Launch(meta['first'],n,width,height,fastest,reduce_y,ctas,vector,(base+row_offset)%4,meta['accumulate'])
    require(type(d.first) is int and d.first>=0 and type(d.accumulate) is bool and d.sane(),'invalid selected launch metadata')
    return d,selected

ZERO=('zero',)
def add(a,b): return ('add',a,b)
def inp(i): return ('input',i)
def chain(seed,terms):
    for term in terms: seed=add(seed,term)
    return seed

def thread_plan(d,x,y,c):
    start=d.start(x,y,c); step=d.step
    if not d.vectorInput:
        accum=[chain(ZERO,(inp(d.first+j) for j in range(start+i*step,d.inputs,4*step))) for i in range(4)]
    else:
        owns=(not d.reduceY or y==0) and c==0
        shift=d.alignmentShift; head=4-shift if shift else 0; remaining=d.inputs-head
        header=add(ZERO,inp(d.first+x-shift)) if shift>0 and shift<=x<4 and owns else ZERO
        accum=[chain(header if i==0 else ZERO,(inp(d.first+head+4*j+i) for j in range(start,remaining//4,step))) for i in range(4)]
        tail=remaining-remaining%4+x
        if owns and tail<remaining: accum[0]=add(accum[0],inp(d.first+head+tail))
    return chain(accum[0],accum[1:])

def halve(values):
    # Only dependencies of owner zero are materialized; no extra off-lane reads.
    while len(values)>1:
        half=len(values)//2
        values=[add(values[i],values[i+half]) for i in range(half)]
    return values[0]

def block_plan(d,c):
    rows=[halve([thread_plan(d,x,y,c) for x in range(d.width)]) if d.reduceX else thread_plan(d,0,y,c) for y in range(d.height if d.reduceY else 1)]
    return halve(rows) if d.reduceY else rows[0]

def launch_plan(d):
    if d.ctas==1: return block_plan(d,0)
    blocks=[block_plan(d,c) for c in range(d.ctas)]
    def staged(x,y):
        start=x+y*d.width if d.reduceX else y
        step=d.width*d.height if d.reduceX else d.height
        return chain(ZERO,(blocks[c] for c in range(start,d.ctas,step)))
    columns=[halve([staged(x,y) for y in range(d.height)]) for x in range(d.width if d.reduceX else 1)]
    return halve(columns) if d.reduceX else columns[0]

def select_plan(launches):
    previous=ZERO
    for d in launches:
        require(d.sane(),'invalid launch')
        value=launch_plan(d)
        previous=add(previous,value) if d.accumulate else value
    return previous

def leaves(tree):
    todo=[tree]; result=[]
    while todo:
        node=todo.pop()
        if node[0]=='input': result.append(node[1])
        elif node[0]=='add': todo.extend((node[2],node[1]))
    return result

def tree_digest(tree):
    # Flat prefix serialization retains every zero and ordered addition.
    tokens=[]; todo=[tree]
    while todo:
        node=todo.pop(); tokens.append(node[0])
        if node[0]=='input': tokens.append(str(node[1]))
        elif node[0]=='add': todo.extend((node[2],node[1]))
    return hashlib.sha256(' '.join(tokens).encode()).hexdigest()

def admit(data):
    for path,pin in PINNED.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,'pinned CUDA reduction source mismatch')
    launches=[]; configs=[]
    for meta in data['subiterators']:
        d,c=select_config(meta); launches.append(d); configs.append(c)
    tree=select_plan(launches); ls=leaves(tree)
    require(sorted(ls)==list(range(data['reduced_inputs'])),'source schedule fails finite input coverage')
    return {'status':'STRUCTURAL_ADMISSION','source_sha256':PINNED,'launches':[asdict(d) for d in launches],
        'selected_configs':configs,'reduced_inputs':data['reduced_inputs'],'tree_sha256':tree_digest(tree),
        'numerical_oracle':False,'GPU_execution':False,
        'boundary':'Actual ordered TensorIterator metadata, input/output mapping and ordinary scalar-add compilation must match the pinned source. This checker does not observe a GPU run.'}

def self_test():
    device={'multiprocessors':132,'max_threads_per_mp':2048}
    def meta(n,outputs,axis='last',base=0,row=0,first=0,acc=False):
        return {'shape':[n,outputs],'input_strides_bytes':[4,4*n] if axis=='last' else [4*outputs,4],
            'num_reduce_dims':1,'input_base_word':base,'output_linear':row,'first':first,'accumulate':acc,'device':device}
    cases={
      'scalar_19':{'reduced_inputs':19,'subiterators':[meta(19,8)]},
      'vector_aligned_131':{'reduced_inputs':131,'subiterators':[meta(131,8)]},
      'vector_row_alignment_131':{'reduced_inputs':131,'subiterators':[meta(131,8,row=1)]},
      'global_262144':{'reduced_inputs':262144,'subiterators':[meta(262144,1)]},
      'scratch_2':{'reduced_inputs':2,'subiterators':[meta(2,131,axis='first')]},
      'scratch_257':{'reduced_inputs':257,'subiterators':[meta(257,128,axis='first')]},
      'coalesced_single_shard':{'reduced_inputs':1,'subiterators':[dict(meta(1,131,axis='first'),shape=[131],input_strides_bytes=[4],num_reduce_dims=0)]},
      'reduction_axis_split':{'reduced_inputs':3,'subiterators':[
          dict(meta(1,131,axis='first'),shape=[131],input_strides_bytes=[4],num_reduce_dims=0),
          dict(meta(1,131,axis='first',base=131,first=1,acc=True),shape=[131],input_strides_bytes=[4],num_reduce_dims=0),
          dict(meta(1,131,axis='first',base=262,first=2,acc=True),shape=[131],input_strides_bytes=[4],num_reduce_dims=0)]},
      'ordered_split':{'reduced_inputs':263,'subiterators':[meta(128,1),meta(135,1,base=128,first=128,acc=True)]},
    }
    results={name:admit(data) for name,data in cases.items()}
    require(results['vector_aligned_131']['tree_sha256']!=results['vector_row_alignment_131']['tree_sha256'],'row alignment must affect tree')
    require(results['global_262144']['launches'][0]['ctas']>1,'global control must use multiple CTAs')
    import copy
    mutants=[]
    for name,change in [
      ('lost_split',lambda d:d['subiterators'].pop()),
      ('reset_accumulate',lambda d:d['subiterators'][1].update(accumulate=False)),
      ('duplicated_interval',lambda d:d['subiterators'][1].update(first=0)),
      ('wrong_config',lambda d:d['subiterators'][0].update(config={})),
      ('wrong_dtype_stride',lambda d:d['subiterators'][0].update(input_strides_bytes=[2,512])),
    ]:
        d=copy.deepcopy(cases['ordered_split']); change(d)
        try: admit(d)
        except ValueError as e: mutants.append({'name':name,'rejected':True,'reason':str(e)})
        else: raise AssertionError(name)
    return {'status':'PASSED','positive_cases':results,'negative_controls':mutants,'source_runtime_capture':False,'native_numerical_conformance':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path); parser.add_argument('--output',required=True,type=Path); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    result=self_test() if args.self_test else admit(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'output':str(args.output)}))
