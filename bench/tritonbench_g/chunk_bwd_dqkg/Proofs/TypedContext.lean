import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedEngine
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Only typed register availability. It asserts no numerical value or result. -/
structure Ready (c : Config) (s : State α) : Prop where
  key : ∃ v : Values α .nat [],s.regs .nat [] "i_k"=some v
  time : ∃ v : Values α .nat [],s.regs .nat [] "i_t"=some v
  head : ∃ v : Values α .nat [],s.regs .nat [] "i_bh"=some v
  heads : ∃ v : Values α .nat [],s.regs .nat [] "n_bh"=some v
  rows : ∃ v : Values α .nat [64],s.regs .nat [64] "o_i"=some v
  gate : ∃ v : Values α .real [64],s.regs .real [64] "b_g"=some v
  last : ∃ v : Values α .real [],s.regs .real [] "b_g_last"=some v
  dq : ∃ v : Values α .real [64,c.BK],s.regs .real [64,c.BK] "b_dq"=some v
  dk : ∃ v : Values α .real [64,c.BK],s.regs .real [64,c.BK] "b_dk"=some v
  ds : ∃ v : Values α .real [64,64],s.regs .real [64,64] "b_ds"=some v
  dgLast : ∃ v : Values α .real [1],s.regs .real [1] "b_dg_last"=some v
  dg : ∃ v : Values α .real [64],s.regs .real [64] "b_dg"=some v

theorem entry_ready (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    Ready c (entry M dot c s).val := by
  constructor <;> simp [entry]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
