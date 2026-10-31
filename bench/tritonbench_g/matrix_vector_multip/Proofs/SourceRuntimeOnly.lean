import bench.tritonbench_g.matrix_vector_multip.Kernels
import VeriTile.Triton.Float.AffineWhileTrace
import bench.tritonbench_g.matrix_vector_multip.Proofs.Mathematical_WholeSource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Parameters := Typed.Parameters

abbrev arguments := Kernels.arguments

def scalarParameters (p : Parameters) : List (RegName × Nat) :=
  [("N",p.rows),("M",p.columns),("stride_an",p.strideRow),("stride_am",p.strideColumn),
    ("stride_bm",p.strideVector),("stride_cn",p.strideOut)]

noncomputable abbrev source := Kernels.fromParameters

def prelude (p : Parameters) : List ComputeStmt := (source p).surfaceBody.take 20
def loopStatement (p : Parameters) : ComputeStmt :=
  ((source p).surfaceBody[20]?).getD (.ifThen (.constBool Bool.false) [])
def loopBody (p : Parameters) : List ComputeStmt := match loopStatement p with
  | .forRangeDyn _ _ _ _ code => code
  | _ => []
def epilogue (p : Parameters) : List ComputeStmt := (source p).surfaceBody.drop 21

theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody=prelude p++loopStatement p::epilogue p := by rfl

variable {α : Type} [Inhabited α]
def preludeExecution (A : Algebra α) (p : Parameters) (s : State α) :
    {after : State α // run A (prelude p) s=some after} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod]
  rfl

theorem prelude_memory (A : Algebra α) (p : Parameters) (s : State α) :
    (preludeExecution A p s).val.mem=s.mem := by simp [preludeExecution]

theorem prelude_pids (A : Algebra α) (p : Parameters) (s : State α) :
    (preludeExecution A p s).val.pids=s.pids := by simp [preludeExecution]

theorem prelude_metadata (A : Algebra α) (p : Parameters) (s : State α)
    (name : RegName) (n : Nat) (member : (name,n) ∈ scalarParameters p) :
    (preludeExecution A p s).val.regs .nat [] name=some (fun _ => n % 2^64) := by
  simp only [scalarParameters,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with h | h | h | h | h | h
  all_goals cases h
  all_goals simp [preludeExecution,Int.toNat_emod]

theorem prelude_counter (A : Algebra α) (p : Parameters) (s : State α) :
    AffineWhile.NormalizedCounter (preludeExecution A p s).val "m" 0 := by
  exact AffineWhile.normalized_set _ _ _

#axiomsClean source_decomposition
#axiomsClean preludeExecution
#axiomsClean prelude_memory
#axiomsClean prelude_pids
#axiomsClean prelude_metadata
#axiomsClean prelude_counter
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
