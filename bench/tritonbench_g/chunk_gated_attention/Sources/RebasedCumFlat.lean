import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedCumSafety
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealCumFlat

/-! Full repaired cumulative source in finite flat allocations, including
inactive dummy programs. The region representative's oracle is internal. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumFlat
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RebasedCumExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def footprint (A : FlatAlloc) (p : Layout) (heads base : Nat) (o : RegionName) (u : BlockState) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ rg off, RebasedCumExecution.footprint p heads base o u (rg,off) ∧ A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (heads base : Nat) (o : RegionName)
    (u v : BlockState) (hf : BlockState.WriteWithin (RebasedCumExecution.footprint p heads base o u) u v) :
    BlockState.WriteWithin (footprint A p heads base o u) (A.flattenState u) (A.flattenState v) := by
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

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool) (dtype out_dtype : FloatDType)
    (u : BlockState)
    (hinj : enabled p heads base u →
      RealCumExecution.NoCollision (q p base u) RebasedCumAccess.localLaunch 0)
    (hb : enabled p heads base u →
      RebasedCumSafety.Bounds p (decoded p base u) (chunk p base u) s o A.extent) :
    ∃ v, exec (A.flattenKernel (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧
      (enabled p heads base u → ∀ i : TileIndex [p.BT,p.BS],
        v.readMem A.flat (A.addr o (RealCumExecution.offset (q p base u) RebasedCumAccess.localLaunch 0 i)) =
          RealCumExecution.expected (q p base u) RebasedCumAccess.localLaunch 0 (u.readMem s) i) ∧
      BlockState.WriteWithin (footprint A p heads base o u) (A.flattenState u) v := by
  let u0 : BlockState := {u with undef := fun _ _ => 0}
  obtain ⟨v,he,ho,hf⟩ := RebasedCumExecution.executes p s o heads base swapped dtype out_dtype u0 hinj
  have hs := RebasedCumSafety.source_safe p s o heads base swapped dtype out_dtype A.extent u0 hb
  have hflat := A.exec_flatten hd hcov _ u0 hs
    (RebasedCumSafety.source_flatten_ok p s o heads base swapped dtype out_dtype) rfl
  rw [he,Option.map_some] at hflat
  refine ⟨A.flattenState v,hflat,?_,flatten_frame A p heads base o u0 v hf⟩
  intro ha i
  have hbound := (hb ha).output i.1 i.2.1 (by simpa using And.intro i.1.isLt i.2.1.isLt)
  have hoff : RealCumExecution.offset (q p base u) RebasedCumAccess.localLaunch 0 i < A.extent o := by
    simpa [RealCumExecution.offset,VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealStep.address,
      q,RebasedCumAccess.tileLayout,RebasedCumAccess.localLaunch] using hbound
  have hr : o ∈ A.regions := by
    by_contra hn
    rw [hcov o hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [A.flattenState_readMem hd v hr hoff]
  exact ho ha i

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumFlat
