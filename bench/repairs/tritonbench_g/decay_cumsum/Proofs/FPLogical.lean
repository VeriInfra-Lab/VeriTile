import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPStorage
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.LogicalStorage

namespace VeriTile.Bench.Repairs.DecayCumsum.FPLogical
open VeriTile.Triton FP.Structural Geometry Grid PackedStorage Logical LogicalStorage
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

/-- Independent numerical graph on a single logical lane, with source order,
load callbacks and casts retained. No source result or program index is used. -/
def evaluate (M : Algebra α) (kind : Kind) (span : Nat) (scale : ℝ)
    (read : RegionName → Nat → α) (r : RegionName) (t : Nat) : Cell α :=
  match kind with
  | .forward => FPReference.forward M (read "g") t
  | .prepare => FPReference.stored M (if r="qg" then FPReference.prepareQ M scale (read "q" t) (read "g" t)
      else FPReference.prepareK M (read "k" t) (read "g" (span-1)) (read "g" t))
  | .backward => FPReference.backwardStored M r (if r="dqo" then FPReference.backwardQ M (read "dqi" t) (read "dqo" t) (read "g" t)
      else if r="dko" then FPReference.backwardK M (read "dki" t) (read "dko" t) (read "g" (span-1)) (read "g" t)
      else FPReference.backwardCarry M span (read "dqi") (read "dqo") (read "dki") (read "dko")
        (read "q") (read "k") (read "g") (span-t))

def reference (M : Algebra α) (h : Call) (kind : Kind) (scale : ℝ) (data : Data (Cell α))
    (r : RegionName) (c : Coordinate h) : Cell α :=
  evaluate M kind (chunkSpan h c.2.1.val) scale
    (fun s n => (data s c.1.val (chunkStart h c.2.1.val+n) c.2.2.val).read .real)
    r (c.2.1.val-chunkStart h c.2.1.val)

theorem tile_evaluate (M : Algebra α) (kind : Kind) (h : Call) (p : Phase) (positive : 0<p.span)
    (prog : Program h p) (scale : ℝ) (u : State α) (read : RegionName → Nat → α)
    (t : Fin p.span) (j : Fin h.BK)
    (reads : ∀ r∈Roles.inputs kind, ∀ n, n<p.span → FPOps.raw h.BK r (FPPrefix.off h p prog n) u (j,PUnit.unit)=read r n)
    (r : RegionName) :
    FPTile.expected M kind h p prog scale u r t.val j=evaluate M kind p.span scale read r t.val := by
  have endpoint : p.span-1<p.span := by omega
  cases kind
  · apply congrArg (FPReference.stored M)
    apply FPTileStable.forward_congr
    intro n hn
    exact reads "g" (by simp [Roles.inputs]) n (by omega)
  · unfold FPTile.expected FPPrepare.qValue FPPrepare.kValue evaluate
    rw [reads "q" (by simp [Roles.inputs]) t.val t.isLt,reads "k" (by simp [Roles.inputs]) t.val t.isLt,
      reads "g" (by simp [Roles.inputs]) t.val t.isLt,reads "g" (by simp [Roles.inputs]) (p.span-1) endpoint]
  · unfold FPTile.expected FPBackwardBase.output evaluate
    apply congrArg (FPReference.backwardStored M r)
    split
    · unfold FPBackwardBase.qValue
      rw [reads "dqi" (by simp [Roles.inputs]) t.val t.isLt,reads "dqo" (by simp [Roles.inputs]) t.val t.isLt,
        reads "g" (by simp [Roles.inputs]) t.val t.isLt]
    · split
      · unfold FPBackwardBase.kValue
        rw [reads "dki" (by simp [Roles.inputs]) t.val t.isLt,reads "dko" (by simp [Roles.inputs]) t.val t.isLt,
          reads "g" (by simp [Roles.inputs]) (p.span-1) endpoint,reads "g" (by simp [Roles.inputs]) t.val t.isLt]
      · apply FPTileStable.backward_congr M p.span (p.span-t.val) (by omega)
        all_goals intro n hn; exact reads _ (by simp [Roles.inputs]) n hn

theorem expected_reference (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (heap : FPStorage.Heap α) (data : Data (Cell α)) (snapshot : Snapshot h kind storage heap data)
    (c : Coordinate h) (o : Owner h c) (r : RegionName) (hr : r∈Roles.outputs kind) :
    FPStorage.expected M h kind storage scale heap o.work (⟨r,hr⟩,o.step,o.lane)=reference M h kind scale data r c := by
  obtain ⟨_,spanEq,timeEq⟩ := owner_chunk h c o
  unfold reference
  rw [spanEq,timeEq]
  apply tile_evaluate M kind h o.work.1 (phase_positive h o.work.1 (schedule_valid h o.work o.member)).2.1
  intro s hs n hn
  apply congrArg (Cell.read .real)
  exact owner_read h kind storage heap data snapshot c o s hs n hn

theorem grid_correct (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (heap : FPStorage.Heap α) (data : Data (Cell α)) (snapshot : Snapshot h kind storage heap data) :
    ∃ final, Run.run (FPStorage.execute M h kind storage scale) (schedule h) heap=some final ∧
      (∀ r∈Roles.outputs kind, ∀ i, crop h kind storage final r i=reference M h kind scale data r (coordinates h i)) ∧
      (∀ addr, (∀ w∈schedule h, ¬footprint h kind storage w addr) → final addr=heap addr) := by
  obtain ⟨final,run,output,frame⟩ := FPStorage.grid_complete M h kind storage scale heap
  refine ⟨final,run,?_,frame⟩
  intro r hr i
  let c := coordinates h i
  let o := owner h c
  have out := output o.work o.member (⟨r,hr⟩,o.step,o.lane)
  rw [expected_reference M h kind storage scale heap data snapshot c o r hr] at out
  simpa only [FPStorage.outputAddress,owner_offset h c o] using out

#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.DecayCumsum.FPLogical
