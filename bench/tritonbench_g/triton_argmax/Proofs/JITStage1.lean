import bench.tritonbench_g.triton_argmax.Kernels
import bench.tritonbench_g.triton_argmax.Proofs.TypedScratch

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITStage1
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def floatingType : TypedScratch.Kind → SourceDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64

def arguments {α : Type} (s : State α) (size base : Nat) : State α :=
  (s.setReg "M" .int [] (fun _ => (size : Int))).setReg
    "program_base" .int [] (fun _ => (base : Int))

theorem floating_run (kind : TypedScratch.Kind) (r : TypedScratch.Rounding)
    (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0 < block) :
    ∃ t, exec (TypedScratch.backend r rest)
      (JITSource.stage1 X V I (floatingType kind) block index64) (arguments s size base) = some t ∧
      t.mem = (TypedScratch.firstWritten kind r s X V I size block base positive).mem := by
  cases kind <;> cases index64 <;>
    simp [floatingType, JITSource.stage1, arguments, ComputeKernel.surfaceBody,
      FP.Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, TypedScratch.backend, Typed.backend, TypedScratch.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex, TileShape.eraseAxis,
      ofFloat, toFloat, bop, numeric, numericLt, numericEq, natLt, TileShape.discreteMaximum,
      TileShape.discreteIntArgMax, TypedReduction.self_equal, TypedReduction.presence,
      TypedReduction.selected, flag, clean, Typed.globalLoaded, Typed.firstIndex,
      TypedScratch.firstWritten, Option.bind, store, TileShape.allIndices, Region.cast,
      -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, collectValues?, TypedScratch.channel, TypedScratch.Rounding.cast]

  all_goals simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast, State.write, State.setReg_mem]
  all_goals rfl


open TypedScratch in
theorem floating_second (kind : Kind) (r : Rounding) (rest : Algebra Word) (s : State Word)
    (V : RegionName) (I O : Region .int) (size block : Nat) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend r rest) (JITSource.stage2 V I O (floatingType kind) block) (s.setReg "mid_size" .int [] (fun _ => (size : Int)))=some t ∧
      t.mem=(secondWritten kind s V I O size block positive).mem := by
  have pad16 : r.fp16 (4286578688#32) = (4286578688#32) := r.fp16Padding
  have padbf : r.bf16 (4286578688#32) = (4286578688#32) := r.bf16Padding
  cases kind <;>
    simp [JITSource.stage2,floatingType,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,Typed.backend,indexed,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
      TypedReduction.selected,flag,clean,secondIndex,secondWritten,Option.bind,
      store,TileShape.allIndices,Region.cast,channel,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul,r.fp16Padding,r.bf16Padding]
  all_goals try simp only [pad16,padbf]
  all_goals simp only [State.write,State.setReg_mem,← apply_ite (fun n : Nat => (n : Int)),Int.toNat_natCast]
  all_goals rfl


#axiomsClean floating_second
#axiomsClean floating_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITStage1
