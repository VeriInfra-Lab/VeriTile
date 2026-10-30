import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedProgram

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
/-- Source registers needed by subsequent statements. Their values are produced
by execution, not supplied as precomputed numerical inputs or carries. -/
def ForwardReady {α : Type} (p : Layout) (s : State α) : Prop :=
  (∃ a, s.regs .real [] "scale" = some a) ∧
  (∃ a, s.regs .nat [] "i_k" = some a) ∧
  (∃ a, s.regs .nat [] "i_v" = some a) ∧
  (∃ a, s.regs .nat [] "i_bh" = some a) ∧
  (∃ a, s.regs .real [] "b_b" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_q" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_k" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_v" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_o" = some a) ∧
  (∃ a, s.regs .bool [p.BK] "mask_bk" = some a) ∧
  (∃ a, s.regs .bool [p.BV] "mask_bv" = some a) ∧
  (∃ a, s.regs .bool [p.BV,p.BK] "mask_kv" = some a) ∧
  (∃ a, s.regs .real [p.BV,p.BK] "h" = some a)

@[simp] theorem scatter_regs {α ι : Type} (items : List ι) (s : State α)
    (region : ι → RegionName) (offset : ι → Nat) (active : ι → Prop)
    [DecidablePred active] (value : ι → Cell α) :
    (items.foldl (fun t i => if active i then
      t.write (region i) (offset i) (value i) else t) s).regs = s.regs := by
  induction items generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

def FirstReady {α : Type} (p : Layout) (s : State α) : Prop :=
  (∃ a, s.regs .real [] "scale" = some a) ∧
  (∃ a, s.regs .nat [] "i_k" = some a) ∧
  (∃ a, s.regs .nat [] "i_v" = some a) ∧
  (∃ a, s.regs .nat [] "i_bh" = some a) ∧
  (∃ a, s.regs .nat [] "i_h" = some a) ∧
  (∃ a, s.regs .real [] "b_b" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_k" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_v" = some a) ∧
  (∃ a, s.regs .ptr [p.BV] "p_do" = some a) ∧
  (∃ a, s.regs .ptr [p.BK] "p_dq" = some a) ∧
  (∃ a, s.regs .bool [p.BK] "mask_bk" = some a) ∧
  (∃ a, s.regs .bool [p.BV] "mask_bv" = some a) ∧
  (∃ a, s.regs .real [p.BK,p.BV] "h" = some a)

def SecondReady {α : Type} (p : Layout) (s : State α) : Prop :=
  (∃ a, s.regs .real [] "scale" = some a) ∧
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

#axiomsClean scatter_regs
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
