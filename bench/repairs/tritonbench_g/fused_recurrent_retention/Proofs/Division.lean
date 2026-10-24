import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.SourceBounds
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxRecDepth 4096

/-- PTX signed division/remainder truncate toward zero. They are applied only
to nonnegative program IDs and positive H, so the Nat owner coordinates agree. -/
def signedDivide (x y : BitVec 32) : BitVec 32 := BitVec.ofInt 32 (x.toInt.tdiv y.toInt)
def signedRemainder (x y : BitVec 32) : BitVec 32 := BitVec.ofInt 32 (x.toInt.tmod y.toInt)

theorem source_division_exact (head heads : Nat) (hh : signed32 head) (hH : signed32 heads) :
    (signedDivide (BitVec.ofInt 32 head) (BitVec.ofInt 32 heads)).toInt=(head/heads : Nat) := by
  unfold signed32 at hh hH
  unfold signedDivide
  rw [BitVec.toInt_ofInt_eq_self (w := 32) (by decide) hh.1 hh.2,
    BitVec.toInt_ofInt_eq_self (w := 32) (by decide) hH.1 hH.2]
  simp only [Int.natCast_tdiv_eq_ediv,←Int.natCast_ediv]
  apply BitVec.toInt_ofInt_eq_self (w := 32) (by decide)
  · have hn : 0≤(head/heads : Nat) := Nat.zero_le _
    omega
  · have hn := Nat.div_le_self head heads
    omega

theorem source_remainder_exact (head heads : Nat) (hh : signed32 head) (hH : signed32 heads) :
    (signedRemainder (BitVec.ofInt 32 head) (BitVec.ofInt 32 heads)).toInt=(head%heads : Nat) := by
  unfold signed32 at hh hH
  unfold signedRemainder
  rw [BitVec.toInt_ofInt_eq_self (w := 32) (by decide) hh.1 hh.2,
    BitVec.toInt_ofInt_eq_self (w := 32) (by decide) hH.1 hH.2,
    Int.tmod_eq_emod_of_nonneg (by omega),←Int.natCast_emod]
  apply BitVec.toInt_ofInt_eq_self (w := 32) (by decide)
  · omega
  · have hn := Nat.mod_le head heads
    omega

#axiomsClean source_division_exact
#axiomsClean source_remainder_exact
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
