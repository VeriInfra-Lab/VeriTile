import bench.tritonbench_g.triton_argmax.Proofs.TypedWhole
import bench.tritonbench_g.triton_argmax.Proofs.Global

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def globalLoaded (s : State Word) (X : RegionName) (size block pid : Nat) (lane : Fin block) : Word :=
  if pid*block+lane.val<size then (s.mem X (pid*block+lane.val)).read .real else 0xff800000

def firstIndex (s : State Word) (X : RegionName) (size block base : Nat) (positive : 0<block) : Nat :=
  TypedReduction.selected positive (globalLoaded s X size block (s.pids 0+base))+(s.pids 0+base)*block

def firstWritten (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (positive : 0<block) : State Word :=
  let selected := firstIndex s X size block base positive
  (s.write V (s.pids 0+base) (.mk .real ((s.mem X selected).read .real))).write
    (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat selected))

theorem stage1_run (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend rest) (TypedSource.stage1FP32 X V I size block base index64) s=some t ∧
      t.mem=(firstWritten s X V I size block base positive).mem := by
  cases index64 <;>
    simp [TypedSource.stage1FP32,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,backend,State.setReg,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
      TypedReduction.selected,flag,clean,globalLoaded,firstIndex,firstWritten,Option.bind,
      store,TileShape.allIndices,Region.cast]
  all_goals rfl

def secondIndex (s : State Word) (V : RegionName) (size block : Nat) (positive : 0<block) : Nat :=
  TypedReduction.selected positive (fun lane : Fin block =>
    if lane.val<size then (s.mem V lane.val).read .real else 0xff800000)

def secondWritten (s : State Word) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0<block) : State Word :=
  s.write (Region.cast O) 0 (.mk .int ((s.mem (Region.cast I) (secondIndex s V size block positive)).read .int))

theorem stage2_run (rest : Algebra Word) (s : State Word) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend rest) (TypedSource.stage2FP32 V I O size block) s=some t ∧
      t.mem=(secondWritten s V I O size block positive).mem := by
  simp [TypedSource.stage2FP32,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,backend,State.setReg,
    VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
    VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
    TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
    ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
    TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
    TypedReduction.selected,flag,clean,secondIndex,secondWritten,Option.bind,
    store,TileShape.allIndices,Region.cast]
  all_goals rfl

#axiomsClean stage1_run
#axiomsClean stage2_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
