import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileInnerExecution
namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte)
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

theorem innerInvariant (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (f k : Nat) (hf : f < 4) (hk : k < p.steps) (defined : InputDefined p initial)
    (h : InnerInvariant cast p initial f k current) :
    InnerInvariant cast p initial f (k + 1) (innerExecution cast M p initial current f k hf hk defined h).val := by
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
    have weights : bValue p initial f k = (fun i : BIndex p =>
        (↑((if i.1.val < p.inner then
          (initial.mem p.weight (bOffset p initial k i)).read .nat else 0) &&&
          (3 <<< (2*f))) : Int) >>> (2*f)-1) := by
      funext i
      rw [bValue_source, Int.natCast_shiftRight]
    rw [weights, signedAdd32_low32]
    try dsimp only [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    erw [max_mod_id _ 4294967296 (by decide)]
    unfold wrap
    congr 1
    apply BitVec.eq_of_toNat_eq
    simp [BitVec.toNat_ofInt]
    have subtractOne (v : BIndex p → Int) :
        bop (fun (x y : Int) => x-y) Broadcast.nil.consR.leadR v
          (fun _ : TileIndex [1] => (1 : Int)) = (fun i => v i-1) := by
      funext i
      rcases i with ⟨r,c,u⟩
      cases u
      rfl
    rw [subtractOne]
    have field_mod (word field : Nat) :
        (((↑(word &&& (3 <<< (2*field))) : Int) >>> (2*field)).bmod 256) =
          ((↑(word &&& (3 <<< (2*field))) : Int) >>> (2*field)) := by
      rw [← Int.natCast_shiftRight, extracted_field]
      have bound : (word >>> (2*field) &&& 3) ≤ 3 := Nat.and_le_right
      have same : (BitVec.ofInt 8 ((word >>> (2*field) &&& 3 : Nat) : Int)).toInt =
          ((word >>> (2*field) &&& 3 : Nat) : Int) := by
        apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
      simpa only [BitVec.toInt_ofInt] using same
    simp only [field_mod]

#axiomsClean innerInvariant
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
