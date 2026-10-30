import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.DenseSafety
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.RealFlat

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1500000

/-- Ordinary allocation facts only: physical bases remain arbitrary. -/
structure DenseAllocation (c : Config) (A : FlatAlloc) : Prop where
  listed : A.regions = regions
  sizes : A.extent = extent c
  apart : A.Disjoint

theorem DenseAllocation.closed (c : Config) (A : FlatAlloc) (h : DenseAllocation c A) :
    ∀ r, r ∉ A.regions → A.extent r = 0 := by
  intro r hr
  rw [h.sizes]
  apply extent_unlisted
  simpa only [h.listed] using hr

theorem prepare_flatten (c : Config) (A : FlatAlloc) (s : BlockState) (a : CTA c) :
    A.flattenState (prepare c s a) = prepare c (A.flattenState s) a := rfl

def runFlat (c : Config) (A : FlatAlloc) : List (CTA c) → BlockState → Option BlockState
  | [], s => some s
  | a :: rest, s => do
      let t ← exec (A.flattenKernel (kernel c).eraseDType.toAlgKernel) (prepare c s a)
      runFlat c A rest t

/-- Flattening is compatible with the entire sequential execution. The flat
source also runs on its actual predecessor memory at every CTA. -/
theorem grid_flatten (c : Config) (A : FlatAlloc) (ha : DenseAllocation c A)
    (order : List (CTA c)) (s : BlockState) :
    runFlat c A order (A.flattenState s) = (runGrid c order s).map A.flattenState := by
  induction order generalizing s with
  | nil => rfl
  | cons a rest ih =>
    obtain ⟨t,he,_⟩ := step_exists c s a
    have hs : (kernel c).eraseDType.toAlgKernel.TraceSafe A.extent (prepare c s a) := by
      rw [ha.sizes]
      exact source_safe_dense c s a
    have hok : (kernel c).eraseDType.toAlgKernel.FlattenOk :=
      RealSafety.source_flatten_ok c.layout .fp32 .fp32 .fp32 .fp32
        "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
    have hf := A.exec_flatten ha.apart (ha.closed c A) _ (prepare c s a) hs hok rfl
    rw [he,Option.map_some,prepare_flatten] at hf
    simp only [runFlat,runGrid,he,hf,bind,Option.bind,ih]

def flatWrites (c : Config) (A : FlatAlloc) : WriteFootprint := fun addr =>
  addr.1 = A.flat ∧ ∃ region off, gridWrites c (allCTAs c) (region,off) ∧ A.addr region off = addr.2

theorem whole_flat_frame (c : Config) (A : FlatAlloc) (s t : BlockState)
    (hf : BlockState.WriteWithin (gridWrites c (allCTAs c)) s t) :
    BlockState.WriteWithin (flatWrites c A) (A.flattenState s) (A.flattenState t) := by
  intro region off hn
  by_cases hr : region=A.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨r,o⟩ := cell
      obtain ⟨_,he,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro hw
      exact hn ⟨rfl,r,o,hw,he.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

/-- Complete flattened execution, logical dQ/dK/per-key-tile dG, and exact
tagged-cell frame for every address outside all active output stores. -/
theorem whole_flat_result (c : Config) (A : FlatAlloc) (ha : DenseAllocation c A)
    (s : BlockState) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    ∃ final, runFlat c A order (A.flattenState s) = some final ∧
      (∀ h t e, final.readMem A.flat (A.addr "dq" ((h.val*c.T+t.val)*c.K+e.val)) = expectedQ c s h t e) ∧
      (∀ h t e, final.readMem A.flat (A.addr "dk" ((h.val*c.T+t.val)*c.K+e.val)) = expectedK c s h t e) ∧
      (∀ k h t, final.readMem A.flat (A.addr "dg" ((k.val*c.heads+h.val)*c.T+t.val)) = expectedG c s k h t) ∧
      BlockState.WriteWithin (flatWrites c A) (A.flattenState s) final := by
  obtain ⟨t,he,hq,hk,hg,hf⟩ := whole_result c s order ho
  refine ⟨A.flattenState t,?_,?_,?_,?_,whole_flat_frame c A s t hf⟩
  · rw [grid_flatten c A ha,he,Option.map_some]
  · intro h row e
    rw [A.flattenState_readMem ha.apart t (by rw [ha.listed]; simp [regions])
      (by rw [ha.sizes]; simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ h.isLt row.isLt) e.isLt)]
    exact hq h row e
  · intro h row e
    rw [A.flattenState_readMem ha.apart t (by rw [ha.listed]; simp [regions])
      (by rw [ha.sizes]; simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ h.isLt row.isLt) e.isLt)]
    exact hk h row e
  · intro k h row
    rw [A.flattenState_readMem ha.apart t (by rw [ha.listed]; simp [regions])
      (by rw [ha.sizes]; simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ k.isLt h.isLt) row.isLt)]
    exact hg k h row

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
