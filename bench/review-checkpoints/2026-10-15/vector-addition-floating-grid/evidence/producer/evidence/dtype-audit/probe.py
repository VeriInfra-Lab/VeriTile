#!/usr/bin/env python3
"""Source-bound CPU/frontend audit. No original module import, CUDA, or device execution."""
import ast, hashlib, importlib.util, inspect, json, os, sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'bench/tritonbench_g/vector_addition/vector_addition.py'
import torch,triton,triton.language as tl
import triton.language.semantic as semantic
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz']
triton_names=['int1','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8e4nv','float8e5','float8e4b8','float8e5b16']
rows=[]
sem=semantic.TritonSemantic(None)
for xn,xt in zip(names,triton_names):
 for yn,yt in zip(names,triton_names):
  row={'x':xn,'y':yn,'output':xn,'store_element_type':'int8' if xn=='bool' else xt}
  try:row.update(status='promoted',add_dtype=str(sem.computation_type_impl(getattr(tl,xt),False,getattr(tl,yt),False,False)))
  except Exception as e:row.update(status='promotion_error',error_type=type(e).__name__,error=str(e))
  rows.append(row)
(OUT/'promotion-matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='add_kernel')
lines=SOURCE.read_text().splitlines(keepends=True)
text='import triton\nimport triton.language as tl\n'+''.join(lines[min([node.lineno]+[x.lineno for x in node.decorator_list])-1:node.end_lineno])
extract=OUT/'extracted_add_kernel.py';extract.write_text(text)
extracted_node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef))
assert ast.dump(node,include_attributes=False)==ast.dump(extracted_node,include_attributes=False)
spec=importlib.util.spec_from_file_location('vector_addition_dtype_audit_source',extract)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);opts=backend.parse_options({'num_warps':4})
context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
probes=[('float32','float32'),('float64','float32'),('float16','bfloat16'),('bfloat16','int32'),('int8','int8'),('int8','uint8'),('int64','uint64'),('bool','bool'),('bool','float32'),('float8_e4m3fn','float8_e5m2'),('float8_e4m3fn','int8'),('float64','float8_e4m3fn'),('float8_e4m3fnuz','float8_e4m3fnuz')]
results=[]
for xn,yn in probes:
 sig={'x_ptr':'*'+jit.type_canonicalisation_dict[xn],'y_ptr':'*'+jit.type_canonicalisation_dict[yn],'output_ptr':'*'+jit.type_canonicalisation_dict[xn],'n_elements':'i32'}
 row={'x':xn,'y':yn,'signature':sig}
 try:
  src=ASTSource(mod.add_kernel,sig,constexprs={'BLOCK_SIZE':1024})
  module=src.make_ir(target,opts,backend.get_codegen_implementation(opts),backend.get_module_map(),context)
  irtext=str(module);path=OUT/(xn+'__'+yn+'.raw.ttir');path.write_text(irtext)
  row.update(status='frontend_generated',ttir_path=path.name,ttir_sha256=sha(path),arithmetic_and_cast_lines=[l.strip() for l in irtext.splitlines() if any(s in l for s in ['arith.addf','arith.addi','arith.ext','arith.trunc','arith.fpto','arith.sitofp','arith.uitofp','tt.fp_to_fp','tt.store'])])
 except Exception as e:row.update(status='frontend_error',error_type=type(e).__name__,error=str(e))
 results.append(row)
(OUT/'frontend-probes.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:print(r['x'],r['y'],r['status'])
methods=['computation_type_impl','integer_promote_impl','add','_load_legacy','_store_legacy','cast']
(OUT/'triton-semantic-excerpts.txt').write_text('\n'.join(inspect.getsource(getattr(semantic.TritonSemantic,m)) for m in methods))
dtype_inventory={str(v):jit.type_canonicalisation_dict.get(str(v).split('.')[-1]) for v in vars(torch).values() if isinstance(v,torch.dtype)}
receipt={'task':'vector_addition original dtype/promotion/store independent audit','source':str(SOURCE),'source_sha256':sha(SOURCE),'extracted_kernel_sha256':sha(extract),'kernel_ast_identical':True,'environment':{'python':sys.version,'torch':torch.__version__,'triton':triton.__version__,'semantic_path':semantic.__file__,'semantic_sha256':sha(semantic.__file__),'jit_sha256':sha(jit.__file__)},'scope':{'representative_torch_dtype_count':len(names), 'distinct_canonical_scalar_signatures':len(names),'promotion_pairs':len(rows),'promotion_success':sum(r['status']=='promoted' for r in rows),'promotion_errors':sum(r['status']=='promotion_error' for r in rows),'frontend_profile_count':len(results),'frontend_success':sum(r['status']=='frontend_generated' for r in results),'offline_target':{'backend':'cuda','arch':90,'warp_size':32},'constexprs':{'BLOCK_SIZE':1024},'frontend_options':vars(opts),'original_module_imported':False,'gpu_executed':False,'native_backend_compiled':False,'full_dtype_native_matrix_complete':False,'whole_program_acceptance':False},'torch_dtype_canonicalization_inventory':dtype_inventory,'bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'bench/tritonbench_g/vector_addition/Kernels.lean',ROOT/'VeriTile/Triton/Core/Types.lean',ROOT/'VeriTile/Triton/Core/Ast.lean',ROOT/'VeriTile/Triton/DSL/Typing.lean',ROOT/'VeriTile/Triton/Float/Structural.lean',ROOT/'VeriTile/Triton/Float/ScalarBackend.lean',ROOT/'VeriTile/Triton/Float/FixedInteger.lean']},'artifacts':{p.name:sha(p) for p in [Path(__file__),OUT/'promotion-matrix.json',OUT/'frontend-probes.json',OUT/'triton-semantic-excerpts.txt']},'conclusions':['Original wrapper only asserts CUDA device membership; it has no dtype, matching shape/length, contiguous-stride, or same-device assertion. output is empty_like(x), and n_elements is output.numel().','The 17 representative Torch dtypes map to 17 distinct canonical Triton scalar signatures: bool, signed/unsigned 8/16/32/64, fp16/bf16/fp32/fp64, and four fp8 formats. Runtime table additionally maps shell torch.int1/torch.uint1 to the same u1 signature as bool; their allocator/storage domain is uncharacterized. Recognition and frontend generation do not establish CUDA allocator or native backend support.','Mixed tensor promotion: fp64 dominates; then fp32; then fp16. bf16+bf16 stays bf16, remaining bf16 mixtures use fp32. Identical fp8 stays that format; different fp8 uses fp16. fp8+integer/bool directly raises TypeError.','Integer promotion does not widen small equal-width operands to int32: same signedness uses wider width; mixed signedness selects unsigned at equal/greater unsigned width, otherwise signed wider width. The final addition is arith.addi at that width and requires modular bitvector semantics in ordinary non-debug execution. Raw frontend IR also includes overflow-sanitizer calculations/assertions; this probe does not lower them or prove the debug/error domain.','tl.store casts to x pointer element dtype. Bool pointer is first reinterpreted as int8, then value cast to int8; mixed bool output is not modeled by a general nonzero-to-bool store. Bool+bool uses i1 addition before zero-extending to stored i8.','Float downcast defaults RTNE. Float/integer casts use signed/unsigned conversion and need explicit out-of-range/NaN semantics. Integer truncation and signed extension are visible in frontend IR.','Current vector_addition Kernels.lean uses RegionName (real), no source dtype parameter or promotion/store dispatch. Existing per-program fp32 theorem cannot cover all native branches.','DSL has typed fp16/bf16/fp32/f8e4/f8e5 channels and an explicit ComputeDType.fp64 annotation/raw Float64Bits representation, so float64 is not wholly absent. There is no distinct fp64 TileDType, and current kernel does not bind fp64 provenance; native binary64 arithmetic/cast correspondence remains to establish.','DSL int/nat channels use unbounded Int/Nat addition, not per-dtype modular arithmetic. FixedInteger and unsigned/signed cast helpers can express explicit conversions, but current kernel has no width-preserving automatic integer arithmetic or full signed/unsigned promotion matrix.','The fp8e4b8 (Torch float8_e4m3fnuz) same-type frontend probe is rejected at sm90 with unsupported-architecture dtype; this is a checked frontend failure, not a claim about all architectures. Only two fp8 storage channels exist in TileDType. Distinct fnuz fp8 formats need separate semantics or explicit unsupported-runtime characterization; canonical Torch fp8 names alone do not prove native support.','Unknown canonical Torch shell/complex/quantized dtypes remain explicitly uncharacterized here; absence from the runtime canonicalization table is recorded, not claimed as a full host error proof.','Masked tl.load has no other argument, so inactive lanes contain unspecified values. Arithmetic may evaluate those lanes, but stores are masked; any finite-input premise over every lane must account for this source behavior.'],'remaining_work':['Compile the complete source-bound 17x17 native backend matrix at each intended architecture, separating frontend promotion errors and lowering failures (especially fp8/fp64 and fnuz).','Prove dtype dispatch/promotion/storage casts and integer wrapping, or characterize exact error branches. These are structural scope gaps, not deferrable numerical assumptions.','Bind native Torch allocation/layout/device/error behavior; prove valid physical address bounds for both inputs and output despite missing shape/stride checks.','For newly represented FP arithmetic/casts, register narrowly scoped numerical assumptions and preserve exceptional-value domain.']}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('receipt',sha(OUT/'receipt.json'))
