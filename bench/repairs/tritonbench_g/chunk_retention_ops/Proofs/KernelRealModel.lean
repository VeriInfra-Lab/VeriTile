import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardModel
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.RealScalar
import bench.repairs.tritonbench_g.chunk_delta_fwd.RealModel
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural
open KernelBackward (Parameters PID Matrix)
noncomputable section
abbrev Memory := KernelBackward.Memory

def parameters (c : Shape) : Parameters := ⟨c.B,c.H,c.T,c.K,c.V,64,c.BK,c.BV,c.chunks⟩
/-- The mathematical interpretation of precisely the scalar operations appearing
in the selected retention source. Native rounding has a separate interpretation. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x=>x
  negInf := 0
  binary := fun _ _ op a b=>match op with
    | .add=>a+b | .sub=>a-b | .mul=>a*b | .div=>a/b
    | .max=>max a b | .pow=>Real.rpow a b
  unary := fun _ op x=>match op with
    | .exp2=>Real.exp (x*Real.log 2)
    | .log2=>Real.log x/Real.log 2
    | _=>x
  cast := fun _ _ _ x=>x
  fromNat := fun _ n=>n
  fromInt := fun _ n=>n
  fp32Bits := fun _=>0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _=>0
  reduceSum := fun _ {_} _ _ _ _=>0
abbrev dotPrimitive := VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.dot

/-- Independent logical inputs read from the admitted buffers. Absent optional
state operands denote zero; no output or execution result occurs here. -/
def inputs (c : Shape) (m : Memory ℝ) (seed terminal : Bool) : Inputs ℝ where
  q := fun h t e=>(m "q" (h*(c.T*c.K)+t*c.K+e)).read .real
  k := fun h t e=>(m "k" (h*(c.T*c.K)+t*c.K+e)).read .real
  v := fun h t j=>(m "v" (h*(c.T*c.V)+t*c.V+j)).read .real
  seed := fun h e j=>if seed then (m "h0" (h*c.K*c.V+e*c.V+j)).read .real else 0
  incoming := fun h t j=>(m "incoming" (h*(c.T*c.V)+t*c.V+j)).read .real
  finalIncoming := fun h e j=>if terminal then (m "terminal" (h*c.K*c.V+e*c.V+j)).read .real else 0

theorem logRate_eq (c : Shape) (pid : PID) :
    KernelBackward.logRate algebra (parameters c) pid=logRate realOps c pid.z := by
  rw [RealScalar.log_rate]
  simp [KernelBackward.logRate,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
    parameters,VeriTile.Bench.Repairs.ChunkLinearAttn.scalar,VeriTile.Bench.Repairs.ChunkLinearAttn.mul,
    MathTarget.headRate]

theorem weight_eq (c : Shape) (pid : PID) (i : Nat) :
    KernelBackward.weight algebra (parameters c) pid i=decay realOps c pid.z ((i:Int)+1) := by
  unfold KernelBackward.weight
  rw [logRate_eq]
  simp [algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
    VeriTile.Bench.Repairs.ChunkLinearAttn.mul,decay,realOps]

#axiomsClean logRate_eq
#axiomsClean weight_eq
#specNonCircular inputs avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
