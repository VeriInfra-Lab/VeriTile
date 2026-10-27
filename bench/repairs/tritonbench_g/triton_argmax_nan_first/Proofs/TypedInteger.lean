import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.IntegerReduction
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedGlobal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedInteger
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Source's nonfloating stage-two branch on the shared exact decoded integer
channel. Physical signed/unsigned widths and bool byte normalization occur
only at the scalar load boundary, using Formats.decodeStored. -/
abbrev stage2 := TypedSource.stage2Integer

def selected (s : State Word) (V : Region .int) (size block : Nat) (padding : Int) : Nat :=
  TileShape.leftmostIntArgMax (fun lane : Fin block =>
    if lane.val<size then (s.mem (Region.cast V) lane.val).read .int else padding)

theorem stage2_run (rest : Algebra Word) (s : State Word) (V I O : Region .int)
    (size block : Nat) (padding : Int) :
    ∃ t, FP.Structural.exec rest (stage2 V I O size block padding) s=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int
        ((s.mem (Region.cast I) (selected s V size block padding)).read .int))).mem := by
  simp [stage2,TypedSource.stage2Integer,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    State.setReg,TileShape.axisDim,TileShape.insertAxisIndex,TileShape.eraseAxis,
    ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteIntArgMax,selected,
    Option.bind,store,TileShape.allIndices,Region.cast]
  all_goals rfl

theorem selected_tree (s : State Word) (V : Region .int) (size block : Nat) (padding : Int)
    (positive : 0<block) (tree : Tree block) (complete : tree.Complete) :
    selected s V size block padding=(tree.pick orderedGt orderedEq
      (fun lane => if lane.val<size then (s.mem (Region.cast V) lane.val).read .int else padding)).val :=
  IntegerReduction.shared_tree positive _ tree complete

/-- Actual shared quotation through selected-value reread, before the
source's scalar conversion macro and its two typed stores. -/
def firstPrefix (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) :
    List ComputeStmt := (TypedSource.stage1FP32 X V I size block base index64).surfaceBody.take 17

theorem first_prefix (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0<block) :
    ∃ t, run (Typed.backend rest) (firstPrefix X V I size block base index64) s=some t ∧
      t.mem=s.mem ∧
      t.regs .real [] "max_val"=some (fun _ => (s.mem X (Typed.firstIndex s X size block base positive)).read .real) ∧
      t.regs .int [] "max_index"=some (fun _ => (Typed.firstIndex s X size block base positive : Int)) := by
  cases index64 <;>
    simp [firstPrefix,TypedSource.stage1FP32,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,Typed.backend,State.setReg,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
      TypedReduction.selected,flag,clean,Typed.globalLoaded,Typed.firstIndex,Option.bind,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast]
  all_goals exact ⟨rfl,rfl⟩

#axiomsClean stage2_run
#axiomsClean selected_tree
#axiomsClean first_prefix
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedInteger
