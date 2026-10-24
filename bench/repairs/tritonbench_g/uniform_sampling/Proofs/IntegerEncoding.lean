import VeriTile.Triton.Core.Ast
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.UniformSampling.Bounds
open VeriTile.Triton

theorem i64_exact {x : Nat} (hx:x<2^63) : (BitVec.ofNat 64 x).toInt=(x : Int) := by
  have hn:(BitVec.ofNat 64 x).toNat=x:=by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn];omega),hn]

def counterBits (pid B lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 pid*BitVec.ofNat 64 B+BitVec.ofNat 64 lane

def outputBits (pid B lane : Nat) : Nat→BitVec 64
  | 0 =>BitVec.ofNat 64 pid*BitVec.ofNat 64 B*BitVec.ofNat 64 4+BitVec.ofNat 64 lane
  | f+1 =>outputBits pid B lane f+BitVec.ofNat 64 B

theorem counter_bits (pid B lane : Nat) : counterBits pid B lane=BitVec.ofNat 64 (pid*B+lane) := by
  simp only [counterBits,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]

theorem output_bits (pid B lane f : Nat) : outputBits pid B lane f=BitVec.ofNat 64 ((pid*4+f)*B+lane) := by
  induction f with
  |zero=>
    simp only [outputBits,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
    congr 1
    ring
  |succ f ih=>
    rw [outputBits,ih,BitVec.ofNat_add_ofNat]
    congr 1
    ring


#axiomsClean i64_exact
#axiomsClean counter_bits
#axiomsClean output_bits
end VeriTile.Bench.Repairs.UniformSampling.Bounds
