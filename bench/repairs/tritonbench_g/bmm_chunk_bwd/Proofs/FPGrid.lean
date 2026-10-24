import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPDefault
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostFP
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostContract

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

def DefaultPrecision (h : HostCall) : Prop :=
  h.aStorage = .fp32 ∧ h.doutStorage = .fp32 ∧
    h.residualDType = .fp32 ∧ h.outputDType = .fp32

theorem source_default (h : HostCall) (c : Tuning) (hp : DefaultPrecision h) :
    h.source c Bool.false = FPDefault.source c.BM c.BN c.BCS (HostContract.parameters h) := by
  rcases hp with ⟨ha,hd,hr,ho⟩
  rw [HostContract.source_eq,ha,hd,hr,ho]
  rfl

/-- Successful complete opaque FP grid, with the actual source program order
and fresh program registers. This does not assume a successful source run. -/
theorem run_success {α : Type} [Inhabited α] (M : Algebra α) (D : FPDefault.DotPrimitive α)
    (h : HostCall) (c : Tuning) (hp : DefaultPrecision h) (jobs : List HostFP.Pid) (s : State α) :
    ∃ t, HostFP.run (FPDefault.backend M D) h c Bool.false jobs s = some t := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons q rest ih =>
    obtain ⟨mid,hm⟩ := FPDefault.source_success M D c.BM c.BN c.BCS
      (HostContract.parameters h) (HostFP.launch h c q s)
    rw [← source_default h c hp] at hm
    obtain ⟨final,hf⟩ := ih mid
    refine ⟨final,?_⟩
    simp only [HostFP.run,hm,bind,Option.bind_some]
    exact hf

#axiomsClean run_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPGrid
