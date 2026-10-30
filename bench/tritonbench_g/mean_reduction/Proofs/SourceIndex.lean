import bench.tritonbench_g.mean_reduction.Proofs.WholeGrid

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile.Triton
abbrev i32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n

theorem i32_exact {n : Nat} (h : n < 2^31) : (i32 n).toInt = (n : Int) := by
  have hn : (i32 n).toNat = n := by
    rw [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (lt_trans h (by norm_num))
  have hd : 2*(i32 n).toNat < 2^32 := by rw [hn]; omega
  rw [BitVec.toInt_eq_toNat_of_lt hd,hn]

/-- Actual signed multiplication/addition used by the row index. -/
def rowBits (pid lane : Nat) : BitVec 32 := i32 pid*i32 8+i32 lane

def colBits (off lane : Nat) : BitVec 32 := i32 off+i32 lane

def rowProductBits (pid lane N : Nat) : BitVec 32 := rowBits pid lane*i32 N

def bytePointer (base offset : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (4*base)+BitVec.ofNat 64 (4*offset)

theorem rowBits_exact (pid lane : Nat) (bound : pid*8+lane < 2^31) :
    (rowBits pid lane).toInt = ((pid*8+lane : Nat) : Int) := by
  unfold rowBits
  rw [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact i32_exact bound

theorem colBits_exact (off lane : Nat) (bound : off+lane < 2^31) :
    (colBits off lane).toInt = ((off+lane : Nat) : Int) := by
  unfold colBits
  rw [BitVec.ofNat_add_ofNat]
  exact i32_exact bound

theorem rowProductBits_exact (pid lane N : Nat)
    (bound : (pid*8+lane)*N < 2^31) :
    (rowProductBits pid lane N).toInt = (((pid*8+lane)*N : Nat) : Int) := by
  unfold rowProductBits rowBits
  rw [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
  exact i32_exact bound

theorem bytePointer_exact (base offset : Nat) (bound : 4*(base+offset) < 2^64) :
    (bytePointer base offset).toNat = 4*(base+offset) := by
  unfold bytePointer
  rw [BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  rw [show 4*base+4*offset = 4*(base+offset) by omega,Nat.mod_eq_of_lt bound]


/-- Explicit native pointer expression after signed32 row multiplication and
column addition have been converted to the pointer-width offsets. -/
def nativeInputPointer (base pid lane N off col : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (4*base) +
  BitVec.ofNat 64 (4*(rowProductBits pid lane N).toInt.toNat) +
  BitVec.ofNat 64 (4*(colBits off col).toInt.toNat)

def nativeOutputPointer (base pid lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (4*base)+BitVec.ofNat 64 (4*(rowBits pid lane).toInt.toNat)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
