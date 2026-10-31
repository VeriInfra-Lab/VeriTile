import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHExecution

/-! Both complete H kernels in disjoint finite flat allocations. The explicit
zero-unspecified-value premise is retained from the region theorem. Active
history/final outputs match the independent closed form, and every other whole
tagged cell is preserved. Bounds and separation are on actual source accesses. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHFlat
open VeriTile.Triton RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
    (storeFinal : Bool) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, RealHExecution.footprint p l r storeFinal (rg,off) ∧ A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
    (storeFinal : Bool) (u v : BlockState)
    (hf : BlockState.WriteWithin (RealHExecution.footprint p l r storeFinal) u v) :
    BlockState.WriteWithin (footprint A p l r storeFinal) (A.flattenState u) (A.flattenState v) := by
  intro rg off hmiss
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨rg',off'⟩ := cell
      obtain ⟨_,heq,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro hhit
      exact hmiss ⟨rfl,rg',off',hhit,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

structure Outputs (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealHStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))) =
      closed p l a gateK hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val))) =
      closed p l a gateK hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHFlat
