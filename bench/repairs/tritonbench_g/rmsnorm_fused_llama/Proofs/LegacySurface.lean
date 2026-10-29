import bench.tritonbench_g.rmsnorm_fused_llama.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Task-local external-scalar frontend. It seeds only eps's FP32 erased
register type and emits no placeholder statement or scalar value. -/
open Lean VeriTile.Triton VeriTile.Triton.DSL
syntax "llama_source" "[" ident "]" "{" tritonStmt* "}" : term
macro_rules
  | `(llama_source[$epsilon:ident] { $stmts:tritonStmt* }) => do
    let body := stmts.toList
    let pinned := Inference.collectNatPinned body
    let intPinned := Inference.collectIntPinned body
    let regions := Inference.collectRegionDTypes body
    let pointers := Inference.collectPtrElems regions body
    let context : FPSourceContext := { regions := regions, legacyPointerCasts := Bool.false }
    let environment : Env := [(epsilon.getId.toString,.real,SInfo.scalar,none)]
    let (_,code,_,_) ← expandStmts environment pinned intPinned regions pointers context [] [] body
    let (inputs,outputs) := Metadata.blockRegions body
    let inputs := inputs.toArray
    let outputs := outputs.toArray
    let result ← `(ComputeKernel.mk [$inputs,*] [$outputs,*] [$code,*])
    normalizePrecisionScopes none result.raw

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.LegacySurface
open VeriTile Triton FP.Structural

/-- One authoritative complete quotation. Mathematical and runtime bindings
change only epsilon's scalar leaf. All casts, loops and stores are shared. -/
syntax "llama_kernel%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(llama_kernel%[$binding:ident]) => do
    let literal := binding.getId.eraseMacroScopes == `literal
    unless literal || binding.getId.eraseMacroScopes == `runtime do
      Macro.throwError "expected literal or runtime epsilon binding"
    let epsilon : TSyntax `tritonExpr ← if literal then
      `(tritonExpr| $((eps : ℝ))) else `(tritonExpr| eps)
    let source ← `(ComputeKernel.mk [X,W] [Y] (llama_source[eps] {
    tl.region X = tl.float32, W = tl.float32, Y = tl.float32
    row = tl.program_id(0)
    Y += row * $(N)
    X += row * $(N)
    _var = tl.zeros([$(B)], dtype=tl.float32)
    for off in range(0, $(N), $(B)) {
      cols = off + tl.arange(0, $(B))
      x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
      _var += x * x
    }
    var = tl.sum(_var, axis=0) / $(N)
    rstd = 1 / tl.sqrt(var + $epsilon)
    for off in range(0, $(N), $(B)) {
      cols = off + tl.arange(0, $(B))
      mask = cols < $(N)
      w = tl.load(W + cols, mask=mask).to(tl.float32)
      x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
      x_hat = x * rstd
      y = x_hat * w
      tl.store(Y + cols, (y.to(tl.float16)).to(tl.float32), mask=mask)
    }
  }).surfaceBody )
    if literal then
      `(fun (X Y W : RegionName) (N B : Nat) (eps : ℝ) => $source)
    else
      `(fun (X Y W : RegionName) (N B : Nat) => $source)

def kernel := llama_kernel%[literal]
def runtimeKernel := llama_kernel%[runtime]

/-- A genuine alternative full kernel, commuting only an integer pointer product. -/
def indexCommuted (source : ComputeKernel) (Y : RegionName) (N : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs
    (source.surfaceBody.take 1 ++
      [.assign .ptr [] "Y" (.alg (.ptrAdd .nil (.ptrBase Y)
        (.mul .nat .nil (.constNat N) (.ref .nat [] "row"))))] ++ source.surfaceBody.drop 2)

def indexCommutedKernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  indexCommuted (kernel X Y W N B eps) Y N

def runtimeCandidate (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  indexCommuted (runtimeKernel X Y W N B) Y N

end VeriTile.Bench.Repairs.RmsnormFusedLlama.LegacySurface
