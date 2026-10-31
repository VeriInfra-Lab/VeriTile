import bench.tritonbench_g.triton_argmax.Proofs.TypedRealReturn
import bench.tritonbench_g.triton_argmax.Proofs.TypedScratch

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

def secondIndex (kind : TypedScratch.Kind) (s : State Value) (V : RegionName)
    (size block : Nat) (positive : 0<block) : Nat :=
  (best positive (fun lane : Fin block =>
    if lane.val<size then toFloat (TypedScratch.channel kind) ((s.mem V lane.val).read (TypedScratch.channel kind).toTileDType)
      else ⊥)).val

def secondWritten (kind : TypedScratch.Kind) (s : State Value) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0<block) : State Value :=
  s.write (Region.cast O) 0
    (.mk .int ((s.mem (Region.cast I) (secondIndex kind s V size block positive)).read .int))

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

#axiomsClean first_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
