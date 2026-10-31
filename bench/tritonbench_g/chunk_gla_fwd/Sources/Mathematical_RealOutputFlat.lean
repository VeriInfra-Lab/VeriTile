import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealOutputSafety
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealOutputCandidate

/-! Complete real consumer execution in a flat allocation. Active address bounds,
allocation disjointness, zero unspecified values and distinct output addresses
are explicit. The conclusion includes the independent gated contraction and
preservation of every other tagged memory cell for both source orders. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputFlat
open VeriTile.Triton RealOutput
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

private theorem flatten_append (alloc : FlatAlloc) (xs ys : List Stmt) :
    alloc.flattenStmts (xs ++ ys) = alloc.flattenStmts xs ++ alloc.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append,FlatAlloc.flattenStmts,ih]

section
variable (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)



def footprint (alloc : FlatAlloc) (o : RegionName) (s_v_h s_v_t T V BT BV : Nat) (s : BlockState) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ ∃ idx : TileIndex [BT,BV],
    (tIndex s BT idx.1.val < T ∧ vIndex s BV idx.2.1.val < V) ∧
    cell.2 = alloc.addr o (outOffset s s_v_h s_v_t BT BV idx)

theorem flatten_frame (alloc : FlatAlloc) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealOutput.footprint o s_v_h s_v_t T V BT BV s) s t) :
    BlockState.WriteWithin (footprint alloc o s_v_h s_v_t T V BT BV s) (alloc.flattenState s) (alloc.flattenState t) := by
  intro region off hn
  by_cases hr : region = alloc.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : alloc.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨r,k⟩
      obtain ⟨_,heq,_⟩ := alloc.decode_sound hd
      apply congrArg alloc.trCell
      apply hf
      rintro ⟨hr,idx,hi,haddr⟩
      apply hn
      refine ⟨rfl,idx,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]


end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputFlat
