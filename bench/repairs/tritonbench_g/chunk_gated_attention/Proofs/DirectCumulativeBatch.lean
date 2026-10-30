import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CumulativeOwnership
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.FlatBatch

/-! Every active or exhausted autotune batch has a constructed ordinary finite
flat-grid execution and a region representative for the following batch. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.DirectCumulativeBatch
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RebasedCumExecution RebasedCumAccess
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def writes (p : Layout) (heads base size : Nat) (o : RegionName) (u : BlockState) : WriteFootprint :=
  fun cell => ∃ idx : GridIndex (Batch.grid size), RebasedCumExecution.footprint p heads base o (u.withGridIndex idx) cell

def Outputs (p : Layout) (heads base size : Nat) (o : RegionName) (a : Nat → ℝ) (u v : BlockState) : Prop :=
  ∀ idx : GridIndex (Batch.grid size), enabled p heads base (u.withGridIndex idx) →
    ∀ i : TileIndex [p.BT,p.BS],
      v.readMem o (RealCumExecution.offset (q p base (u.withGridIndex idx)) localLaunch 0 i) =
        RealCumExecution.expected (q p base (u.withGridIndex idx)) localLaunch 0 a i

theorem executes (p : Layout) (heads base size : Nat) (s o : RegionName) (swapped : Bool) (dtype out_dtype : FloatDType)
    (a : Nat → ℝ) (u : BlockState)
    (hp : CumulativeHost.Canonical p) (hsize : 0<size) (hgrid : size≤Grid.limit)
    (hx : ∀ off, u.readMem s off=a off) :
    ∃ v, Nonempty (Kernel.GridLaunchedOrdinary
      (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel
      (Batch.grid size) u v) ∧
      Outputs p heads base size o a u v ∧
      BlockState.WriteWithin (writes p heads base size o u) u v ∧ v.undef=u.undef := by
  classical
  have hex (idx : GridIndex (Batch.grid size)) := RebasedCumExecution.executes p s o heads base swapped dtype out_dtype
    (u.withGridIndex idx) (fun he => CumulativeHost.noCollision p heads _ _ hp
      (CumulativeHost.decoded_valid p heads base (u.withGridIndex idx) he))
  choose out he hout hframe using hex
  let kernel := (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel
  let frames : Kernel.GridFrames kernel (Batch.grid size) u := fun idx =>
    ⟨out idx,RebasedCumExecution.footprint p heads base o (u.withGridIndex idx),he idx,hframe idx⟩
  have hdis : Kernel.GridWritesDisjoint frames := by
    intro i j hn cell hi hj
    have hl := CumulativeOwnership.owner p heads base base o (u.withGridIndex i) (u.withGridIndex j) hp cell hi hj
    simp only [linear,Batch.pid_lane] at hl
    exact hn (Batch.lane_injective (Nat.add_right_cancel hl))
  let v := Kernel.mergeFrames (Batch.grid size) u frames
  refine ⟨v,⟨⟨frames,hdis,rfl⟩⟩,?_,Kernel.mergeFrames_writeWithin frames,rfl⟩
  intro idx ha i
  have hw : (frames idx).writes (o,RealCumExecution.offset (q p base (u.withGridIndex idx)) localLaunch 0 i) :=
    ⟨ha,rfl,i,all_active p _ _ i,rfl⟩
  have hm := Kernel.mergeFrames_mem_written hdis idx o _ hw
  have hr := hout idx ha i
  have hin : (u.withGridIndex idx).readMem s=a := funext hx
  rw [hin] at hr
  exact (congrArg (fun cell : MemCell => match cell.readAs .real with | some (some x) => x | _ => 0) hm).trans hr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.DirectCumulativeBatch
