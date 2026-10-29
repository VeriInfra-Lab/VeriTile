"""CPU metadata observations of the exact empty_like call, not GPU execution."""
import hashlib
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
inputs={
  'scalar':torch.empty(()),
  'empty':torch.empty((0,3)),
  'contiguous':torch.empty((2,3)),
  'transpose_dense':torch.empty((2,3)).t(),
  'offset_slice':torch.empty(10)[3:7],
  'strided_slice':torch.empty(8)[::2],
  'transpose_slice':torch.empty((3,4)).t()[::2],
  'expanded_unsafe_flat_span':torch.empty(1).expand(4),
  'empty_arbitrary_strides':torch.empty_strided((0,3),(5,2)),
  'singleton_arbitrary_stride':torch.empty_strided((1,3),(99,1)),
}
rows=[]
for name,x in inputs.items():
  out=torch.empty_like(x,dtype=torch.float32,device=x.device)
  safe=x.storage_offset()*x.element_size()+x.numel()*x.element_size()<=x.untyped_storage().nbytes()
  assert out.shape==x.shape and out.dtype==torch.float32 and out.device==x.device
  assert out.storage_offset()==0 and out.numel()==x.numel()
  assert out.untyped_storage().nbytes()>=out.numel()*out.element_size()
  assert not x.numel() or x.untyped_storage().data_ptr()!=out.untyped_storage().data_ptr()
  rows.append({'name':name,'shape':list(x.shape),'input_strides':list(x.stride()),
    'input_storage_offset':x.storage_offset(),'input_storage_bytes':x.untyped_storage().nbytes(),
    'input_flat_span_safe':safe,'output_strides':list(out.stride()),'output_storage_offset':out.storage_offset(),
    'output_dtype':str(out.dtype),'output_device':str(out.device),'output_storage_bytes':out.untyped_storage().nbytes()})
assert not next(r for r in rows if r['name']=='expanded_unsafe_flat_span')['input_flat_span_safe']
data={'kind':'trusted_Torch_metadata_observation','torch_version':torch.__version__,
  'source_call':'torch.empty_like(x, dtype=torch.float32, device=x.device)',
  'device_tested':'cpu','cuda_allocation_tested':False,'kernel_executed':False,
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':rows,
  'boundary':'These observations support the named metadata primitive contract. They are not a proof of Torch/CUDA allocator implementation or all layouts.'}
(root/'metadata-observation.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'metadata_cases':len(rows),'unsafe_expanded_view_preserved':True}))
