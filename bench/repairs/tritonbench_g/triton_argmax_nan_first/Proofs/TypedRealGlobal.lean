import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedRealReturn
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedScratch

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def globalLoaded (s : State Value) (X : RegionName) (size block pid : Nat) (lane : Fin block) : Value :=
  if pid*block+lane.val<size then (s.mem X (pid*block+lane.val)).read .real else ⊥

def firstIndex (s : State Value) (X : RegionName) (size block base : Nat) (positive : 0<block) : Nat :=
  (best positive (globalLoaded s X size block (s.pids 0+base))).val+(s.pids 0+base)*block

def firstWritten (kind : TypedScratch.Kind) (s : State Value) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (positive : 0<block) : State Value :=
  let selected := firstIndex s X size block base positive
  (s.write V (s.pids 0+base)
    (.mk (TypedScratch.channel kind).toTileDType
      (ofFloat (TypedScratch.channel kind) ((s.mem X selected).read .real)))).write
    (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat selected))

theorem stage1_run (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend rest) (TypedScratch.firstKernel kind X V I size block base index64) s=some t ∧
      t.mem=(firstWritten kind s X V I size block base positive).mem := by
  cases kind <;> cases index64 <;>
    simp [TypedScratch.firstKernel,TypedSource.stage1FP32,TypedSource.stage1FP16,TypedSource.stage1BF16,TypedSource.stage1FP64,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,State.setReg,indexed,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,orderedEq,globalLoaded,firstIndex,firstWritten,Option.bind,
      store,TileShape.allIndices,Region.cast,TypedScratch.channel]
  all_goals rfl

def secondIndex (kind : TypedScratch.Kind) (s : State Value) (V : RegionName)
    (size block : Nat) (positive : 0<block) : Nat :=
  (best positive (fun lane : Fin block =>
    if lane.val<size then toFloat (TypedScratch.channel kind) ((s.mem V lane.val).read (TypedScratch.channel kind).toTileDType)
      else ⊥)).val

def secondWritten (kind : TypedScratch.Kind) (s : State Value) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0<block) : State Value :=
  s.write (Region.cast O) 0
    (.mk .int ((s.mem (Region.cast I) (secondIndex kind s V size block positive)).read .int))

theorem stage2_run (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (V : RegionName) (I O : Region .int) (size block : Nat) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend rest) (TypedScratch.secondKernel kind V I O size block) s=some t ∧
      t.mem=(secondWritten kind s V I O size block positive).mem := by
  cases kind <;>
    simp [TypedScratch.secondKernel,TypedSource.stage2FP32,TypedSource.stage2FP16,TypedSource.stage2BF16,TypedSource.stage2FP64,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,State.setReg,indexed,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,orderedEq,secondIndex,secondWritten,Option.bind,
      store,TileShape.allIndices,Region.cast,TypedScratch.channel]
  all_goals rfl

theorem best_tree {n : Nat} (positive : 0<n) (input : Fin n → Value)
    (tree : Tree n) (complete : tree.Complete) : best positive input=tree.pick orderedGt orderedEq input :=
  least_maximum_unique _ _ _
    (mathematical_maximum input _ (Math.enumeration_complete n positive))
    (mathematical_maximum input tree complete)

/-- The first quoted program's original-address reread matches the independent
Real stage and erased scalar conversion for every admitted active block. -/
theorem first_reference (s : State Value) (X : RegionName) (c : GlobalConfig)
    (coverage : Math.GlobalCoverage c) (base : Nat) (input : Nat → ℝ)
    (active : s.pids 0+base<c.blocks)
    (reads : ∀ i,i<c.size → (s.mem X i).read .real=(input i : Value)) :
    firstIndex s X c.size c.block base coverage.blockPositive=Math.firstIndex c input (s.pids 0+base) ∧
      (s.mem X (firstIndex s X c.size c.block base coverage.blockPositive)).read .real=
        RealScratch.firstValue adapter c input (s.pids 0+base) := by
  have loads : globalLoaded s X c.size c.block (s.pids 0+base)=Math.padded c.size ((s.pids 0+base)*c.block) c.block input := by
    funext lane
    by_cases valid : (s.pids 0+base)*c.block+lane.val<c.size
    · simp [globalLoaded,Math.padded,valid,reads _ valid]
    · simp [globalLoaded,Math.padded,valid]
  have selected : firstIndex s X c.size c.block base coverage.blockPositive=Math.firstIndex c input (s.pids 0+base) := by
    unfold firstIndex
    rw [loads,best_tree coverage.blockPositive _ _ (coverage.firstComplete _ active)]
    exact Nat.add_comm _ _
  refine ⟨selected,?_⟩
  rw [selected,RealScratch.first_value adapter c coverage input _ active]
  exact reads _ (Math.block_valid _ _ _ _ _ (coverage.firstComplete _ active)
    (coverage.eachActive _ active) coverage.blockPositive)

/-- Scalar load bindings to the actual first-grid stores suffice: successful
second-quotation execution gathers the independent Real first-maximum index. -/
theorem second_reference (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (V : RegionName) (I O : Region .int) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (input : Nat → ℝ)
    (values : ∀ i,i<c.blocks → toFloat (TypedScratch.channel kind)
      ((s.mem V i).read (TypedScratch.channel kind).toTileDType)=(input (Math.firstIndex c input i) : Value))
    (indices : ∀ i,i<c.blocks → (s.mem (Region.cast I) i).read .int=Int.ofNat (Math.firstIndex c input i)) :
    ∃ t, FP.Structural.exec (backend rest) (TypedScratch.secondKernel kind V I O c.blocks c.midBlock) s=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int (Int.ofNat
        (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val))).mem := by
  let positive := coverage.blocksPositive.trans_le coverage.midFits
  obtain ⟨t,run,mem⟩ := stage2_run kind rest s V I O c.blocks c.midBlock positive
  have loads : (fun lane : Fin c.midBlock => if lane.val<c.blocks then
      toFloat (TypedScratch.channel kind) ((s.mem V lane.val).read (TypedScratch.channel kind).toTileDType)
      else ⊥)=Math.midValues c input := by
    funext lane
    by_cases active : lane.val<c.blocks
    · simp [Math.midValues,active,values _ active]
    · simp [Math.midValues,active]
  have selected : secondIndex kind s V c.blocks c.midBlock positive=(Math.globalSelected c input).val := by
    unfold secondIndex
    rw [loads,best_tree positive _ c.secondTree coverage.secondComplete]
    rfl
  refine ⟨t,run,mem.trans ?_⟩
  unfold secondWritten
  rw [selected,indices _ (Math.mid_valid c coverage input)]
  have result := congrArg Fin.val (least_maximum_unique _ _ _ (Math.global_mathematical c coverage input)
    (Math.argmax_spec c.size coverage.sizePositive (fun i => input i.val)))
  rw [←result]
  rfl

#axiomsClean stage1_run
#axiomsClean stage2_run
#axiomsClean first_reference
#axiomsClean second_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
