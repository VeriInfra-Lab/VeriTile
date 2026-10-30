import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Contract

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked
open FusedRotaryEmbedding Checked
noncomputable section

/-- The new check occurs only after the original T=0 return and applies to
the actual full-width arange tile, not a device-resource estimate. -/
def tileChecks (s : Shape) : Bool := decide (s.tokens=0 ∨ s.fullBlock≤1048576)
def inputChecks (c : Call) (m : Physical.Bytes) : Bool := Checked.check c m && tileChecks c.shape

theorem check_iff (c : Call) (m : Physical.Bytes) : inputChecks c m=true ↔
    Checked.check c m=true ∧ (c.shape.tokens=0 ∨ c.shape.fullBlock≤1048576) := by
  simp [inputChecks,tileChecks]

theorem original_checks (c : Call) (m : Physical.Bytes) (h : inputChecks c m=true) :
    Checked.check c m=true := (check_iff c m).mp h |>.1

theorem empty_domain_unchanged (c : Call) (m : Physical.Bytes) (empty : c.shape.tokens=0) :
    inputChecks c m=Checked.check c m := by simp [inputChecks,tileChecks,empty]

theorem arange_bounds (c : Call) (m : Physical.Bytes) (h : inputChecks c m=true)
    (live : 0<c.shape.tokens) :
    0<c.shape.halfBlock ∧ c.shape.halfBlock≤1048576 ∧
    0<c.shape.fullBlock ∧ c.shape.fullBlock≤1048576 := by
  have hf : c.shape.fullBlock≤1048576 := by
    rcases (check_iff c m).mp h |>.2 with empty | bound
    · omega
    · exact bound
  have hh : c.shape.halfBlock≤c.shape.fullBlock := by
    apply Nat.pow_le_pow_right (by decide : 0<2)
    exact Nat.clog_mono_right 2 (Nat.div_le_self _ _)
  exact ⟨Nat.pow_pos (by decide),hh.trans hf,Nat.pow_pos (by decide),hf⟩

#axiomsClean check_iff
#axiomsClean original_checks
#axiomsClean empty_domain_unchanged
#axiomsClean arange_bounds
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked
