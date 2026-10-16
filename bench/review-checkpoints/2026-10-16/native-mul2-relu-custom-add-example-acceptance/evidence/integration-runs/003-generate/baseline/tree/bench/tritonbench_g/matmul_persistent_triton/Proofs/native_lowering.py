import ast, hashlib, importlib.util, json, os, sys, tempfile
from pathlib import Path
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/veritile-persistent-native-cache")
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
source=Path(__file__).resolve().with_name("source_reference.py")
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ("_matmul_launch_metadata","matmul_kernel_persistent")]
results=[]
with tempfile.TemporaryDirectory(prefix="veritile-persistent-source-") as directory:
    target=Path(directory)/"persistent_source.py"
    target.write_text("import triton\nimport triton.language as tl\n\n"+"\n\n".join(ast.unparse(n) for n in nodes)+"\n")
    spec=importlib.util.spec_from_file_location("persistent_source",target)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    function=module.matmul_kernel_persistent
    for dtype,bk,stages in [("fp16",64,3),("fp8e4nv",128,4)]:
        constants={"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":bk,"GROUP_SIZE_M":8,"NUM_SMS":132}
        signature={name:"i32" for name in function.arg_names if name not in constants}
        signature.update(a_ptr="*"+dtype,b_ptr="*"+dtype,c_ptr="*"+dtype)
        row={"dtype":dtype,"constexprs":constants,"signature":signature,"num_warps":8,"num_stages":stages}
        try:
            compiled=triton.compile(ASTSource(function,signature,constexprs=constants),target=GPUTarget("cuda",90,32),options={"num_warps":8,"num_stages":stages})
            ttir=compiled.asm["ttir"]
            Path("/tmp/veritile-persistent-"+dtype+".ttir").write_text(ttir)
            row.update(status="compiled",ttir_sha256=hashlib.sha256(ttir.encode()).hexdigest(),ptx_bytes=len(compiled.asm["ptx"]),dot_lines=[line.strip() for line in ttir.splitlines() if "tt.dot" in line],signed_minus_one="-1 : i32" in ttir,loop_present="scf.for" in ttir)
        except Exception as error:
            row.update(status="compile_failure",error_type=type(error).__name__,error=str(error))
        results.append(row)
        print(dtype,row["status"],flush=True)
receipt={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"triton_version":triton.__version__,"target":{"backend":"cuda","arch":90,"warp_size":32},"scope":"Unchanged AST of both wrapper storage/block profiles. Offline compile only, no GPU execution, global race proof or IEEE numerical claim. Runtime dimensions and strides remain signed i32; NUM_SMS=132 is one native specialization.","results":results}
Path(__file__).resolve().with_name("native_lowering_sm90_triton_3_7_1.json").write_text(json.dumps(receipt,indent=2)+"\n")
raise SystemExit(any(row["status"]!="compiled" for row in results))
