import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Batch
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.LaunchPlan

/-! Every logical head/key/value coordinate is owned by an actually launched
program and an active tile lane, including the final partial feature tiles. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.LogicalCoverage
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def index {size : Nat} (n : Fin size) : GridIndex (Batch.grid size) := fun ax =>
  ⟨n.val,by
    have hz : ax.val=0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ ax.isLt)
    simpa [Batch.grid,VeriTile.Triton.Grid.dim,hz] using n.isLt⟩

@[simp] theorem lane_index {size : Nat} (n : Fin size) : Batch.lane (index n)=n.val := rfl

theorem coverage (p : Layout) (heads h k v : Nat) (u : BlockState)
    (hp : StagedLayout.Canonical p) (hh : h<heads) (hk : k<p.K) (hv : v<p.V) :
    ∃ plan ∈ LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV)),
    ∃ idx : GridIndex (Batch.grid plan.size), ∃ e : Fin p.BK, ∃ j : Fin p.BV,
      (RebasedInit.decoded p plan.base (u.withGridIndex idx)).head=h ∧
      keyIndex p (RebasedInit.decoded p plan.base (u.withGridIndex idx)) e.val=k ∧
      valueIndex p (RebasedInit.decoded p plan.base (u.withGridIndex idx)) j.val=v := by
  have hkt : k/p.BK < p.K/p.BK := (Nat.div_lt_iff_lt_mul hp.bk).2 (by rwa [hp.key])
  have hvt : v/p.BV < p.V/p.BV := (Nat.div_lt_iff_lt_mul hp.bv).2 (by rwa [hp.value])
  let n := Grid.encode (p.V/p.BV) (p.K/p.BK) (v/p.BV) (k/p.BK) h
  have hn : n < heads*(p.K/p.BK)*(p.V/p.BV) := Grid.encode_bounded hvt hkt hh
  obtain ⟨plan,hplan,hbase,hsz⟩ := LaunchPlan.plans_cover _ n hn
  let li : Fin plan.size := ⟨n-plan.base,by omega⟩
  let idx := index li
  have hlin : (u.withGridIndex idx).pids 0+plan.base=n := by
    rw [Batch.pid_lane,lane_index]
    dsimp [li]
    omega
  have hd := Grid.decode_encode (z := h) hvt hkt
  have hvd : (RebasedInit.decoded p plan.base (u.withGridIndex idx)).value=v/p.BV := by
    simpa only [RebasedInit.decoded,hlin,Grid.decodeX] using hd.1
  have hkd : (RebasedInit.decoded p plan.base (u.withGridIndex idx)).key=k/p.BK := by
    simpa only [RebasedInit.decoded,hlin,Grid.decodeY] using hd.2.1
  have hhd : (RebasedInit.decoded p plan.base (u.withGridIndex idx)).head=h := by
    simpa only [RebasedInit.decoded,hlin,Grid.decodeZ] using hd.2.2
  refine ⟨plan,hplan,idx,⟨k%p.BK,Nat.mod_lt _ hp.bk⟩,⟨v%p.BV,Nat.mod_lt _ hp.bv⟩,hhd,?_,?_⟩
  · simp only [keyIndex,hkd]
    simpa [Nat.mul_comm] using Nat.div_add_mod k p.BK
  · simp only [valueIndex,hvd]
    simpa [Nat.mul_comm] using Nat.div_add_mod v p.BV

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.LogicalCoverage
