import bench.tritonbench_g.matmul_triton2.Proofs.MaskPreludeInvariant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.MaskRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def bodyResult (p : Core.Parameters) (initial current : State (Word α)) (block : Nat) : State (Word α) :=
  let s1 := current.setReg "k" .nat [] (fun _ => block)
  let s2 := s1.setReg "a" .real [p.blockRows,p.blockInner]
    (Core.aValues (symbolicPrimitives α) p initial block)
  let s3 := s2.setReg "b" .real [p.blockInner,p.blockColumns]
    (Core.bValues (symbolicPrimitives α) p initial block)
  let s4 := s3.setReg "accumulator" .real [p.blockRows,p.blockColumns]
    (Core.accumulation (symbolicPrimitives α) p initial (block+1))
  let s5 := s4.setReg "a_ptrs" .ptr [p.blockRows,p.blockInner]
    (fun i => (p.input,Core.aOffset p initial block i+p.blockInner*p.inputInner))
  s5.setReg "b_ptrs" .ptr [p.blockInner,p.blockColumns]
    (fun i => (p.weight,Core.bOffset p initial block i+p.blockInner*p.weightInner))

def bodyExecution (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (initial current : State (Word α)) (block : Nat)
    (h : RuntimeInvariant p initial block current) :
    {after : State (Word α) //
      run (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {}) (loopBody p)
        (current.setReg "k" .nat [] (fun _ => block)) = some after} := by
  have hm := h.metadata "M" p.rows (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  have hn := h.metadata "N" p.columns (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  have hk := h.metadata "K" p.innerSize (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  have hsx := h.metadata "stride_ak" p.inputInner (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  have hsy := h.metadata "stride_bk" p.weightInner (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  refine ⟨bodyResult p initial current block,?_⟩
  simp [loopBody,loopStatement,VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.maskHoisted,VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.arguments,
    VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.maskHoistedBody,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,
    ScalarArgument.integer,ScalarArgument.statement,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,hm,hn,hk,hsx,hsy,h.aptrs,h.bptrs,h.offm,h.offn,h.rowMask,h.columnMask,h.offk,h.acc,h.memory,
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.add,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.tileDot,
    bodyResult,Core.accumulation,Core.aValues,Core.bValues,Core.matrixDot,symbolicPrimitives,selectedOptions,
    Broadcast.leftIndex,Broadcast.rightIndex,show (0.0 : ℝ)=0 by norm_num]
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Core.aValues,Core.bValues,Core.inner,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.add]
  rfl

#axiomsClean bodyExecution
end
end VeriTile.Bench.Repairs.MatmulTriton2.MaskRuntime
