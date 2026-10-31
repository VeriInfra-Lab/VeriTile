import bench.tritonbench_g.swiglu_triton.Proofs.Tile

namespace VeriTile.Bench.Repairs.SwigluTriton.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

theorem row_unique (block r c r' c' : Nat) (hc : c < block) (hc' : c' < block)
    (equal : r*block+c=r'*block+c') : r=r' ∧ c=c' := by
  have forward : r≤r' := by
    by_contra h
    have hmul := Nat.mul_le_mul_right block (show r'+1≤r by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hmul
    omega
  have backward : r'≤r := by
    by_contra h
    have hmul := Nat.mul_le_mul_right block (show r+1≤r' by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hmul
    omega
  have same : r=r' := by omega
  exact ⟨same, by subst r'; omega⟩

def Outside (backward : Bool) (rows cols block : Nat) (region : RegionName) (address : Nat) : Prop :=
  ∀ ch,enabled backward ch → region≠outputRegion ch ∨
    ∀ row col,row<rows → col<cols → address≠row*block+col

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Grid
