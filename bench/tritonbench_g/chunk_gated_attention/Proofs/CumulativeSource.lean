import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels
import VeriTile.Triton.Core.Integer
import VeriTile.Triton.Float.Integer
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Parameters where
  headStride : Nat
  timeStride : Nat
  featureStride : Nat
  T : Nat
  S : Nat
  BT : Nat
  BS : Nat
  heads : Nat
  base : Nat
  swapped : Bool := Bool.false

variable {α : Type} [Inhabited α]

def kernel (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) : ComputeKernel :=
  (if p.swapped then NativeSource.cumIndexFirst else NativeSource.cumOriginal)
    dtype out_dtype s o p.headStride p.timeStride p.featureStride p.T p.S p.BT p.BS p.heads p.base

def prelude (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) : List ComputeStmt :=
  (kernel p dtype out_dtype s o).surfaceBody.take 4

def guardStatement (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) : ComputeStmt :=
  ((kernel p dtype out_dtype s o).surfaceBody[4]?).getD (.ifThen (.constBool Bool.false) [])

def activeBody (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) : List ComputeStmt :=
  match guardStatement p dtype out_dtype s o with
  | .ifThen _ body => body
  | _ => []

theorem decomposition (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) :
    (kernel p dtype out_dtype s o).surfaceBody =
      prelude p dtype out_dtype s o ++ [guardStatement p dtype out_dtype s o] := by
  cases h:p.swapped <;> simp [kernel,h,NativeSource.cumIndexFirst,NativeSource.cumIndexFirstBody,NativeSource.cumOriginal,NativeSource.cumBody,
    ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,
    ScalarArgument.statement,prelude,guardStatement,activeBody]

theorem active_length (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) :
    (activeBody p dtype out_dtype s o).length=11 := by
  cases h:p.swapped <;> simp [kernel,h,NativeSource.cumIndexFirst,NativeSource.cumIndexFirstBody,NativeSource.cumOriginal,NativeSource.cumBody,
    ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,
    ScalarArgument.statement,prelude,guardStatement,activeBody]

def preludeExecution (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    {final : State α // run model (prelude p dtype out_dtype s o) initial=some final} := by
  refine ⟨?_,?_⟩
  swap
  cases h:p.swapped
  all_goals
    simp [prelude,kernel,NativeSource.cumOriginal,NativeSource.cumBody,ComputeKernel.surfaceBody,
      ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
      NativeSource.cumIndexFirst,NativeSource.cumIndexFirstBody,h,run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Region.cast,Option.bind,numeric,bop]
    try rfl


theorem prelude_memory (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    (preludeExecution model p dtype out_dtype s o initial).val.mem=initial.mem := by
  simp [preludeExecution]

theorem prelude_pids (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    (preludeExecution model p dtype out_dtype s o initial).val.pids=initial.pids := by
  simp [preludeExecution]

theorem prelude_linear (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    (preludeExecution model p dtype out_dtype s o initial).val.regs .int [] "linear" =
      some (fun _=>((initial.pids 0+p.base : Nat) : Int)) := by
  simp [preludeExecution,Int.natCast_add,bop]
  rfl

def activePrelude (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) : List ComputeStmt :=
  (activeBody p dtype out_dtype s o).take 8

theorem intBopComm (v w : TileIndex [] → Int) :
    bop Int.mul .nil v w=bop Int.mul .nil w v := by
  funext i
  exact Int.mul_comm _ _

def activePreludeExecution (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    {final : State α // run model (activePrelude p dtype out_dtype s o)
      (preludeExecution model p dtype out_dtype s o initial).val=some final} := by
  refine ⟨?_,?_⟩
  swap
  cases h:p.swapped
  all_goals
    simp [activePrelude,activeBody,guardStatement,kernel,NativeSource.cumOriginal,NativeSource.cumBody,
      ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,
      ScalarArgument.statement,preludeExecution,run,step,evalExpr,evalComputeOp,
      NativeSource.cumIndexFirst,NativeSource.cumIndexFirstBody,h,FP.Structural.eval_truncDiv,FP.Structural.eval_truncMod,evalOp_unfold,evalOffsets,
      Region.cast,Option.bind,numeric,bop,numericLt,natLt,natLe,numericLe,numericEq,integralFloorDiv,integralMod,
      ofFloat,toFloat,Int.mul_comm]
    try rfl
    all_goals repeat' first | exact intBopComm _ _ | congr! 1


theorem activePrelude_memory (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    (activePreludeExecution model p dtype out_dtype s o initial).val.mem=initial.mem := by
  simp [activePreludeExecution,preludeExecution]

#axiomsClean decomposition
#axiomsClean active_length
#axiomsClean preludeExecution
#axiomsClean prelude_memory
#axiomsClean prelude_linear
#axiomsClean activePreludeExecution
#axiomsClean activePrelude_memory
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
