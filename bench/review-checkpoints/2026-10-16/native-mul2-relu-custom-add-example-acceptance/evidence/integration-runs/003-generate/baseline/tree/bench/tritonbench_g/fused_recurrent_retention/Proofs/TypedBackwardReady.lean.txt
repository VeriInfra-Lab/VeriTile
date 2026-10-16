import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedModel

/-! Typed register availability for the two actual backward loops. Values are
existentially produced by the source; no numerical input or carry is supplied. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def FirstReady {α : Type} (p : Layout) (s : State α) : Prop :=
  (∃ a, s.regs .nat [] "i_k" = some a) ∧
  (∃ a, s.regs .nat [] "i_v" = some a) ∧
  (∃ a, s.regs .nat [] "i_bh" = some a) ∧
  (∃ a, s.regs .real [] "b_b" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_k" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_v" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_do" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_dq" = some a) ∧
  (∃ a, s.regs .bool [p.BK] "mask_bk" = some a) ∧
  (∃ a, s.regs .bool [p.BV] "mask_bv" = some a) ∧
  (∃ a, s.regs .real [p.BK,p.BV] "h" = some a)

def SecondReady {α : Type} (p : Layout) (s : State α) : Prop :=
  (∃ a, s.regs .real [] "b_b" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_q" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_k" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_v" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_do" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_dk" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_dv" = some a) ∧
  (∃ a, s.regs .bool [p.BK] "mask_bk" = some a) ∧
  (∃ a, s.regs .bool [p.BV] "mask_bv" = some a) ∧
  (∃ a, s.regs .real [p.BK,p.BV] "d_h" = some a)

end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
