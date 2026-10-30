import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Batch
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.FlatBatch

/-! Complete regional source execution. Read-only physical aliasing is handled
by direct input projection and fresh-output commit, without input copies. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.DirectRealBatch
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def writes (p : Layout) (r : Regions) (base size : Nat) (storeFinal : Bool) (u : BlockState) : WriteFootprint :=
  fun cell => ∃ idx : GridIndex (Batch.grid size), RebasedExecution.footprint p
    (RebasedInit.decoded p base (u.withGridIndex idx)) r storeFinal cell

theorem executes (p : Layout) (heads : Nat) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base size : Nat) (gateK hasSeed storeFinal swapped : Bool)
    (a : Inputs) (u : BlockState)
    (hp : StagedLayout.Canonical p)
    (hsize : 0 < size) (hgrid : size ≤ Grid.limit)
    (hbatch : base+size ≤ heads*(p.K/p.BK)*(p.V/p.BV))
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off)
    (hgate : ∀ off, u.readMem r.gate off = a.gate off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off) :
    ∃ v, Nonempty (Kernel.GridLaunchedOrdinary
      (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel
      (Batch.grid size) u v) ∧
      (∀ idx : GridIndex (Batch.grid size),
        RealHExecution.Outputs p (RebasedInit.decoded p base (u.withGridIndex idx)) r a gateK hasSeed storeFinal v) ∧
      BlockState.WriteWithin (writes p r base size storeFinal u) u v ∧ v.undef=u.undef := by
  classical
  have hpid (idx : GridIndex (Batch.grid size)) :
      (u.withGridIndex idx).pids 0+base < heads*(p.K/p.BK)*(p.V/p.BV) := by
    rw [Batch.pid_lane]
    have := Batch.lane_lt idx
    omega
  have hl (idx : GridIndex (Batch.grid size)) := StagedLayout.decoded_valid p heads base (u.withGridIndex idx) (hpid idx)
  have hex (idx : GridIndex (Batch.grid size)) := RebasedExecution.executes p r dot gate history seed final base
    gateK hasSeed storeFinal swapped a (u.withGridIndex idx)
    (StagedLayout.full_tile p heads _ hp (hl idx)) (StagedLayout.key_full p heads _ hp (hl idx))
    (StagedLayout.value_full p heads _ hp (hl idx)) hs hfsep (StagedLayout.history_noCollision p _ hp)
    (fun _ => StagedLayout.state_noCollision p _) hkey hvalue hgate hseed
  choose out he ho hf using hex
  let kernel := (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel
  let frames : Kernel.GridFrames kernel (Batch.grid size) u := fun idx =>
    ⟨out idx,RebasedExecution.footprint p (RebasedInit.decoded p base (u.withGridIndex idx)) r storeFinal,he idx,hf idx⟩
  have ht : 0 < heads*(p.K/p.BK)*(p.V/p.BV) := by omega
  have hx : 0 < p.V/p.BV := by
    by_contra hn
    simp [Nat.eq_zero_of_not_pos hn] at ht
  have hy : 0 < p.K/p.BK := by
    by_contra hn
    simp [Nat.eq_zero_of_not_pos hn] at ht
  have hdis : Kernel.GridWritesDisjoint frames := by
    intro i j hne cell hi hj
    have heq := Ownership.region_owner p _ _ r storeFinal hp hfsep cell hi hj
    exact hne (Batch.decoded_injective p base size u hx hy i j heq)
  let v := Kernel.mergeFrames (Batch.grid size) u frames
  refine ⟨v,⟨⟨frames,hdis,rfl⟩⟩,?_,Kernel.mergeFrames_writeWithin frames,rfl⟩
  intro idx
  have hm (rg : RegionName) (off : Nat)
      (hw : RebasedExecution.footprint p (RebasedInit.decoded p base (u.withGridIndex idx)) r storeFinal (rg,off)) :
      v.readMem rg off = (out idx).readMem rg off := by
    unfold BlockState.readMem
    rw [show v.mem rg off = (frames idx).final.mem rg off from Kernel.mergeFrames_mem_written hdis idx rg off hw]
  constructor
  · intro c e j hc hek
    exact (hm _ _ (Or.inl ⟨rfl,c,e,j,hc,hek,rfl⟩)).trans ((ho idx).history c e j hc hek)
  · intro hon e j hek
    exact (hm _ _ (Or.inr ⟨hon,rfl,e,j,hek,by simp [RealHStore.address,RealHFinal.base,stateAddress]⟩)).trans ((ho idx).final hon e j hek)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.DirectRealBatch
