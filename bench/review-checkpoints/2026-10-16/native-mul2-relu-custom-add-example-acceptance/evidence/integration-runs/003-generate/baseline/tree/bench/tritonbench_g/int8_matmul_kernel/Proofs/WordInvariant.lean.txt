import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordExecution
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

private theorem nat_mod_id (x : Int) (n : Nat) (h : 0 < n) :
    (x % (n : Int)).toNat % n = (x % (n : Int)).toNat := by
  have h0 := Int.emod_nonneg x (by omega : (n : Int) ≠ 0)
  have hlt := Int.emod_lt_of_pos x (by omega : (0 : Int) < n)
  apply Nat.mod_eq_of_lt
  omega
private theorem max_mod_id (x : Int) (n : Nat) (h : 0 < n) :
    max (x % (n : Int)) 0 = x % (n : Int) := by
  apply max_eq_left
  exact Int.emod_nonneg x (by omega)

theorem innerInvariant {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial current : State α) (f k : Nat) (h : InnerInvariant p initial f k current) :
    InnerInvariant p initial f (k + 1) (innerExecution M p initial current f k h).val := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · constructor <;> simp [-BitVec.toInt_ofInt,innerExecution, h.memory, h.pidm, h.pidn, h.offsk, h.offsn]
  · simp [-BitVec.toInt_ofInt,innerExecution, h.fieldReg]
  · simp [-BitVec.toInt_ofInt,innerExecution, aOffset, h.aptrs, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Nat.add_mul, Nat.add_assoc]
    funext i
    simp [-BitVec.toInt_ofInt,FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_assoc]
  · simp [-BitVec.toInt_ofInt,innerExecution, bOffset, h.bptrs, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Nat.add_mul, Nat.add_assoc]
    funext i
    simp [-BitVec.toInt_ofInt,FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_assoc]
  · have cast8 (x : Int) :
        (if 2 * ((x % 256).toNat % 256) < 256 then max (x % 256) 0 % 256
        else max (x % 256) 0 % 256 - 256) = wrap 8 x := by
      simpa [IntegerExpr.signedBits, wrap] using signedBits_int_mod 8 x
    have cast32 (x : Int) :
        (if 2 * ((x % 4294967296).toNat % 4294967296) < 4294967296 then
          max (x % 4294967296) 0 % 4294967296
        else max (x % 4294967296) 0 % 4294967296 - 4294967296) = wrap 32 x := by
      simpa [IntegerExpr.signedBits, wrap] using signedBits_int_mod 32 x
    simp [innerExecution, innerValues, wordDot, backend,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, cast8, cast32]
    funext i
    rcases i with ⟨r, c, u⟩
    cases u
    simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, cast8, cast32]
    delta aValue bValue
    simp only [cast8, cast32]
    rfl

#axiomsClean innerInvariant
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
