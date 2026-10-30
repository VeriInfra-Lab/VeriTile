import bench.tritonbench_g.attention_fwd_triton1.Proofs.Contract

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open AttentionFwdTriton1.Whole001
abbrev staticTensorLimit : Nat := 1048576

/-- The source passes BD=D after its original power-of-two dimension check. -/
def tileCheck (D : Nat) : Bool := decide (D*D≤staticTensorLimit)

/-- The new check is reached only after the original empty return. -/
def liveTileCheck (c : Shape) : Bool :=
  if c.T=0 ∨ c.heads=0 then Bool.true else tileCheck c.D

theorem tileCheck_iff (D : Nat) : tileCheck D=Bool.true ↔ D*D≤staticTensorLimit := by
  simp [tileCheck]
theorem liveTileCheck_iff (c : Shape) :
    liveTileCheck c=Bool.true ↔ (¬Host.Empty c → c.D*c.D≤staticTensorLimit) := by
  unfold liveTileCheck Host.Empty
  by_cases h : c.T=0 ∨ c.heads=0 <;> simp [h,tileCheck]
theorem empty_unchanged (c : Shape) (h : Host.Empty c) : liveTileCheck c=Bool.true := by
  exact (liveTileCheck_iff c).2 (fun hn => False.elim (hn h))
theorem boundary_accepted : tileCheck 1024=Bool.true := by decide
theorem first_power_of_two_rejected : tileCheck 2048=Bool.false := by decide

#axiomsClean tileCheck_iff
#axiomsClean liveTileCheck_iff
#axiomsClean empty_unchanged
#axiomsClean boundary_accepted
#axiomsClean first_power_of_two_rejected
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
