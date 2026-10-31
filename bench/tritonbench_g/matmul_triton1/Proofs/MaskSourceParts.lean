import bench.tritonbench_g.matmul_triton1.Kernels
import bench.tritonbench_g.matmul_triton1.Sources.DotProfile
import bench.tritonbench_g.matmul_triton1.Proofs.FiniteIndex
import bench.tritonbench_g.matmul_triton1.Proofs.FPReference
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MatmulTriton1.MaskRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
open VeriTile.Bench.Repairs.MatmulTriton1.Kernel VeriTile.Bench.Repairs.MatmulTriton1.DotProfile
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def prelude (p : Core.Parameters) : List ComputeStmt :=
  (VeriTile.Bench.Repairs.MatmulTriton1.Kernel.maskHoisted p).surfaceBody.take 31

def preludeExecution (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (s : State (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α)) :
    {after : State (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α) //
      run (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) (prelude p) s=some after} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.maskHoisted,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.arguments,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.arguments,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters,
    VeriTile.Bench.Repairs.MatmulTriton1.Kernel.maskHoistedBody,ComputeKernel.surfaceBody,
    ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.add]
  rfl


theorem prelude_rows (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (s : State (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α)) :
    (preludeExecution rest P p s).val.regs .nat [] "m_size" =
      some (fun _ => p.rows % 2^64) := by
  simp only [preludeExecution,State.setReg_ne_name,State.setReg_same]
  simp [preludeExecution,Int.toNat_emod (Int.natCast_nonneg p.rows)
    (by norm_num : (0 : Int) ≤ 18446744073709551616)]


theorem prelude_memory (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (s : State (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α)) :
    (preludeExecution rest P p s).val.mem = s.mem := by simp [preludeExecution]


theorem prelude_pids (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (s : State (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α)) :
    (preludeExecution rest P p s).val.pids = s.pids := by simp [preludeExecution]


theorem prelude_metadata (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (s : State (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α)) (name : RegName) (n : Nat)
    (member : (name,n) ∈ VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters p) :
    (preludeExecution rest P p s).val.regs .nat [] name = some (fun _ => n % 2^64) := by
  simp only [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with h | h | h | h | h | h | h | h | h
  all_goals cases h
  all_goals simp [preludeExecution,Int.toNat_emod]

#axiomsClean prelude_metadata

def loopStatement (p : Core.Parameters) : ComputeStmt :=
  ((VeriTile.Bench.Repairs.MatmulTriton1.Kernel.maskHoisted p).surfaceBody[31]?).getD (.ifThen (.constBool Bool.false) [])

def loopBody (p : Core.Parameters) : List ComputeStmt :=
  match loopStatement p with
  | .forRangeDyn _ _ _ _ body => body
  | _ => []

def epilogue (p : Core.Parameters) : List ComputeStmt :=
  (VeriTile.Bench.Repairs.MatmulTriton1.Kernel.maskHoisted p).surfaceBody.drop 32

theorem source_decomposition (p : Core.Parameters) :
    (VeriTile.Bench.Repairs.MatmulTriton1.Kernel.maskHoisted p).surfaceBody =
      prelude p ++ loopStatement p :: epilogue p := by rfl

#axiomsClean source_decomposition

#axiomsClean prelude_memory
#axiomsClean prelude_pids
#axiomsClean preludeExecution
#axiomsClean prelude_rows
end
end VeriTile.Bench.Repairs.MatmulTriton1.MaskRuntime
