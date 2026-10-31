import bench.tritonbench_g.triton_argmax.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITCall
open VeriTile.Triton FP.Structural
noncomputable section

def first (X V : RegionName) (I : Region .int) (dtype : SourceDType)
    (size block base : Nat) (index64 : Bool) : ComputeKernel :=
  (JITSource.stage1 X V I dtype block index64).withScalarArguments
    [ScalarArgument.integer "M" size,ScalarArgument.integer "program_base" base]

def second (V : RegionName) (I O : Region .int) (dtype : SourceDType)
    (size block : Nat) : ComputeKernel :=
  (JITSource.stage2 V I O dtype block).withScalarArguments [ScalarArgument.integer "mid_size" size]

def dimension (X : RegionName) (O : Region .int) (M N K BM BN base : Nat)
    (index64 : Bool) : ComputeKernel :=
  (JITSource.dimension X O BM BN index64).withScalarArguments
    [ScalarArgument.integer "M" M,ScalarArgument.integer "N" N,
     ScalarArgument.integer "K" K,ScalarArgument.integer "program_base" base]

/-- Only caller argument bindings precede the authoritative quotation. -/
theorem bind_first {α : Type} [Inhabited α] (A : Algebra α) (kernel : ComputeKernel)
    (s : State α) (size base : Nat) :
    exec A (kernel.withScalarArguments [ScalarArgument.integer "M" size,
      ScalarArgument.integer "program_base" base]) s=
    exec A kernel ((s.setReg "M" .int [] (fun _=>(size : Int))).setReg
      "program_base" .int [] (fun _=>(base : Int))) := by
  cases kernel
  simp [ComputeKernel.withScalarArguments,ComputeKernel.surfaceBody,ScalarArgument.integer,ScalarArgument.statement,
    exec,run,step,stepAlg,evalExpr,evalOp_unfold]

theorem bind_second {α : Type} [Inhabited α] (A : Algebra α) (kernel : ComputeKernel)
    (s : State α) (size : Nat) :
    exec A (kernel.withScalarArguments [ScalarArgument.integer "mid_size" size]) s=
    exec A kernel (s.setReg "mid_size" .int [] (fun _=>(size : Int))) := by
  cases kernel
  simp [ComputeKernel.withScalarArguments,ComputeKernel.surfaceBody,ScalarArgument.integer,ScalarArgument.statement,
    exec,run,step,stepAlg,evalExpr,evalOp_unfold]

theorem bind_dimension {α : Type} [Inhabited α] (A : Algebra α) (kernel : ComputeKernel)
    (s : State α) (M N K base : Nat) :
    exec A (kernel.withScalarArguments
      [ScalarArgument.integer "M" M,ScalarArgument.integer "N" N,
       ScalarArgument.integer "K" K,ScalarArgument.integer "program_base" base]) s=
    exec A kernel ((((s.setReg "M" .int [] (fun _=>(M : Int))).setReg
      "N" .int [] (fun _=>(N : Int))).setReg "K" .int [] (fun _=>(K : Int))).setReg
      "program_base" .int [] (fun _=>(base : Int))) := by
  cases kernel
  simp [ComputeKernel.withScalarArguments,ComputeKernel.surfaceBody,ScalarArgument.integer,ScalarArgument.statement,
    exec,run,step,stepAlg,evalExpr,evalOp_unfold]

#axiomsClean bind_first
#axiomsClean bind_second
#axiomsClean bind_dimension
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITCall
