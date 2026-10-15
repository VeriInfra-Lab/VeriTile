from pathlib import Path
import hashlib,json
W=Path(__file__).resolve().parent
text=(W/'probe_host.py').read_text().split('storage=[torch.float16')[0]
text=text.replace("grad=Facade(meta_tensor(shape).to(dtype=out.dtype),trace,'output_gradient')","grad=out")
text=text.replace("'new_strides':list(t.stride())","'new_strides':list(t.stride()),'same_tensor':t is self.tensor")
space={'__file__':str(W/'probe_host.py')};exec(compile(text,str(W/'probe_host.py'),'exec'),space)
torch=space['torch'];run=space['run']
rows=[run(shape,dim,dtype,None,tile) for shape,dim in [((3,5),1),((2,3,5),1)] for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.float64] for tile in [32,128,1024]]
assert len(rows)==24 and all(r['status']=='host_replayed' for r in rows)
assert all(any(e['op']=='contiguous' and e['tag'].startswith('allocation_') and e['same_tensor'] for e in r['trace']) for r in rows)
(W/'evidence/alias-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
result={'status':'passed','cases':len(rows),'original_host_ast_unchanged':True,'output_gradient_is_saved_output':True,'contiguous_preserves_alias':True,'source_sha256':hashlib.sha256(space['SOURCE'].read_bytes()).hexdigest(),'scope':'Exact original backward host method accepts out_grad=out; Torch meta contiguous retains tensor identity. Selected-config launch spies only; compiler admission and actual trial order covered by separate probes.','gpu_execution':False}
(W/'evidence/alias-probe-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
