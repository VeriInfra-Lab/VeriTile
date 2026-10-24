from pathlib import Path
import itertools,json
import torch
W=Path(__file__).resolve().parent
rows=[]
for shape in [(1,),(1,5),(2,1,3),(2,3,1),(1,2,1,3),(2,1,1,3)]:
 strides=[];total=1
 for n in reversed(shape):strides.append(total);total*=n
 canonical=tuple(reversed(strides))
 for replacement in [0,1,17,99]:
  unusual=tuple(replacement if n==1 else stride for n,stride in zip(shape,canonical))
  x=torch.empty_strided(shape,unusual,device='meta');assert x.is_contiguous()
  contiguous=x.contiguous();out=torch.empty_like(contiguous);grad=torch.empty_like(out)
  assert contiguous is x and out.is_contiguous() and grad.is_contiguous()
  assert all(n==1 or st==want for n,st,want in zip(shape,out.stride(),canonical))
  rows.append({'shape':shape,'input_strides':unusual,'contiguous_strides':contiguous.stride(),'output_strides':out.stride(),'gradient_strides':grad.stride(),'contiguous_identity':True,'canonical_output':out.stride()==canonical,'canonical_strides':canonical})
result={'status':'passed','torch_version':torch.__version__,'cases':rows,'scope':'Positive-shape singleton-stride boundary: contiguous may preserve arbitrary singleton strides, empty_like may preserve noncanonical singleton strides. The exact primitive output metadata is retained; only non-singleton coordinate strides are fixed. Ordinary primitive metadata evidence, no GPU values.','gpu_execution':False}
(W/'evidence/layout-probe-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','cases':len(rows),'noncanonical_outputs':sum(not r['canonical_output'] for r in rows)}))
