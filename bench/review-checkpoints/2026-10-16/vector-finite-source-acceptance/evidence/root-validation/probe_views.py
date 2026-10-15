"""CPU metadata and independent physical-address examples; no CUDA execution."""
import hashlib,json,random
from pathlib import Path
import torch
W=Path(__file__).resolve().parent
rows=[]
base=torch.arange(256,dtype=torch.int32)
cases=[('contiguous',base[:12],base[40:52]),
 ('offset',base[7:19],base[60:72]),
 ('strided',base[3:27:2],base[43:67:2]),
 ('transpose',base[:12].reshape(3,4).t(),base[40:52].reshape(3,4).t()),
 ('different_layouts',base[:12].reshape(3,4).t(),base[40:52].reshape(4,3)),
 ('overlapping_inputs',base[5:17],base[10:22]),
 ('short_y_backing_safe',base[:12],base[40:41]),
 ('identical_inputs',base[4:16],base[4:16]),
 ('empty',base[3:3],base[7:7]),
 ('expanded_unsafe',base[:1].clone().expand(12),base[40:52])]
for name,x,y in cases:
 out=torch.empty_like(x);n=out.numel();width=x.element_size()
 valid=lambda t:(t.storage_offset()+n)*t.element_size()<=t.untyped_storage().nbytes()
 safe=valid(x) and valid(y) and valid(out)
 r={'case':name,'shape':list(x.shape),'x_strides':list(x.stride()),'y_strides':list(y.stride()),
  'output_strides':list(out.stride()),'x_storage_offset':x.storage_offset(),'y_storage_offset':y.storage_offset(),
  'output_storage_offset':out.storage_offset(),'numel':n,'y_numel':y.numel(),'flat_span_safe':safe,
  'same_input_allocation':x.untyped_storage().data_ptr()==y.untyped_storage().data_ptr(),
  'fresh_output':out.untyped_storage().data_ptr() not in [x.untyped_storage().data_ptr(),y.untyped_storage().data_ptr()],
  'cpu_metadata_only':True}
 if safe:
  xf=torch.as_strided(x,(n,),(1,),x.storage_offset());yf=torch.as_strided(y,(n,),(1,),y.storage_offset())
  of=torch.as_strided(out,(n,),(1,),out.storage_offset());of.copy_(xf+yf)
  r['source_flat_output']=(xf+yf).tolist();r['returned_tensor_flat_values']=out.flatten().tolist()
  r['equals_torch_elementwise']=torch.equal(out,x+y)
 rows.append(r)
assert not next(x for x in rows if x['case']=='strided')['equals_torch_elementwise']
assert not next(x for x in rows if x['case']=='different_layouts')['equals_torch_elementwise']
assert not next(x for x in rows if x['case']=='expanded_unsafe')['flat_span_safe']
assert next(x for x in rows if x['case']=='short_y_backing_safe')['flat_span_safe']
assert next(x for x in rows if x['case']=='short_y_backing_safe')['y_numel']<12
# Independent ordered-source interpreter and byte frame over finite small examples.
rng=random.Random(7281);memory_cases=[]
for n in [0,1,2,31,1023,1024,1025,2049]:
 for wx in [1,2,4,8]:
  for wy in [1,2,4,8]:
   bx=0x1000;by=0x100000;bo=0x200000;ox=3;oy=5;oo=0
   initial={bx+(ox+i)*wx:100+i for i in range(n)}|{by+(oy+i)*wy:200+2*i for i in range(n)}
   initial.update({bo+i*wx:-11 for i in range(n+3)})
   def source(schedule):
    state=dict(initial)
    for pid in schedule:
     active=[pid*1024+lane for lane in range(1024) if pid*1024+lane<n]
     # Source order: all x loads, then all y loads, then compute, then store.
     xs=[state[bx+(ox+i)*wx] for i in active];ys=[state[by+(oy+i)*wy] for i in active]
     for i,v in zip(active,[a+b for a,b in zip(xs,ys)]):state[bo+i*wx]=v
    return state
   schedule=list(range((n+1023)//1024));a=source(schedule);rng.shuffle(schedule);b=source(schedule)
   expected=dict(initial);expected.update({bo+i*wx:300+3*i for i in range(n)})
   assert a==b==expected
   memory_cases.append({'n':n,'x_bytes':wx,'y_bytes':wy,'output_bytes':wx,'x_offset':ox,'y_offset':oy,'all_values_and_frame_match':True})
# Boundary checks avoid enumerating large grids.
boundaries=[]
for n in [0,1,1023,1024,1025,2**31-1024,2**31-1,2**31,2**31+1]:
 count=(n+1023)//1024;last=count*1024-1 if count else None
 signed_safe=n<2**31 and (last is None or last<2**31)
 boundaries.append({'n':n,'grid':count,'last_launched_index':last,'in_explicit_i32_api':signed_safe,
  'not_a_fabricated_native_rejection':not signed_safe})
metadata=[]
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz']
for name in names:
 storage=torch.empty(128,dtype=getattr(torch,name))
 for layout,x in [('contiguous',storage[:12]),('offset',storage[3:15]),('stride2',storage[3:27:2]),('transpose',storage[3:15].reshape(3,4).t())]:
  out=torch.empty_like(x);width=x.element_size()
  assert out.storage_offset()==0 and out.dtype==x.dtype and out.shape==x.shape
  assert x.data_ptr()==x.untyped_storage().data_ptr()+x.storage_offset()*width
  assert out.data_ptr()==out.untyped_storage().data_ptr()
  assert x.data_ptr()%width==out.data_ptr()%width==0
  metadata.append({'dtype':name,'layout':layout,'shape':list(x.shape),'input_strides':list(x.stride()),'output_strides':list(out.stride()),
   'input_storage_offset':x.storage_offset(),'output_storage_offset':out.storage_offset(),'element_bytes':width,
   'input_storage_bytes':x.untyped_storage().nbytes(),'output_storage_bytes':out.untyped_storage().nbytes(),
   'data_pointer_equals_storage_plus_offset':True,'natural_alignment':True,'fresh_output_allocation':out.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr(),
   'input_stride_copied':x.stride()==out.stride(),'cpu_metadata_only':True})
result={'torch':torch.__version__,'view_cases':rows,'dtype_metadata_cases':metadata,'memory_schedule_cases':memory_cases,'index_boundaries':boundaries,
 'counts':{'actual_cpu_views':len(rows),'dtype_layout_metadata_cases':len(metadata),'memory_cases':len(memory_cases),'index_boundaries':len(boundaries)},
 'native_cuda_runs':0,'numerical_gpu_validation':False,
 'interpretation':'Independent integer examples validate source layout/index semantics; Lean theorems quantify over all admitted dtypes, values and finite safe extents.'}
(W/'evidence/view-and-schedule-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['counts']))
