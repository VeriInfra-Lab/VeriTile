import bench.tritonbench_g.quantize_kv_copy.Kernels
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Kernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit true
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Scalar primitive interpretation, including an explicit total policy for
otherwise undefined INT8 conversion. This does not assume any program result. -/
structure Ops (α : Type) where
  zero : α
  one : α
  levels : α
  abs : α→α
  divide : α→α→α
  equal : α→α→Bool
  maximum : {n : Nat}→(Fin n→α)→α
  convert : α→Int

structure Parameters where
  groups : Nat
  block : Nat
  heads : Nat

def kernel (p : Parameters) : ComputeKernel := KernelSyntax.source p.groups p.block p.heads

def reduction (op : Ops α) {shape : TileShape} (ax : Fin shape.length) :
    (keep : Bool)→Values α .real shape→Values α .real (TileShape.reduceShape shape ax keep)
  | Bool.false,x => fun out=>op.maximum (fun i=>x (TileShape.insertAxisIndex shape ax out i))
  | Bool.true,x => fun out=>op.maximum (fun i=>x (TileShape.replaceAxisIndex shape ax out i))

def backend (op : Ops α) : Algebra α where
  literal := fun _ _ x=>if x=127 then op.levels else if x=1 then op.one else op.zero
  negInf := op.zero
  binary := fun _ _ _=>op.divide
  unary := fun _ _=>op.abs
  compareEq := fun _ _=>some op.equal
  floatToInt := fun _ _ _=>some (fun x=>some (op.convert x))
  cast := fun _ _ _=>id
  fromNat := fun _ _=>op.zero
  fromInt := fun _ _=>op.zero
  fp32Bits := fun _=>op.zero
  fp32Load := id
  reduceMax := fun _=>reduction op
  reduceSum := fun _=>reduction op

/-- Every statement and both stores execute, including zero/NaN/Inf and
padded lanes. Conversion is totalized explicitly, not silently restricted. -/
def execution [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (backend op) (kernel p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [kernel, KernelSyntax.source, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run,
    FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    backend,ComputeDType.eraseDType,FP.Structural.ofFloat,FP.Structural.toFloat,Region.cast,Option.bind,
    TileShape.axisDim]
  try rfl

#axiomsClean execution
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.Kernel
