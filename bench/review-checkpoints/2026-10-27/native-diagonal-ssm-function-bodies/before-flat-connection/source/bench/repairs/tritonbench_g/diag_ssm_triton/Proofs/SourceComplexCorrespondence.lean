import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexExecution
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem source_mod (x d : Nat) :
    Int.toNat ((x : Int) - (d : Int) * ((x : Int) / (d : Int))) = x % d := by
  rw [← Int.emod_def, Int.toNat_emod (Int.natCast_nonneg _) (Int.natCast_nonneg _)]
  simp only [Int.toNat_natCast]

theorem source_mod_int (x d : Nat) :
    (x : Int) - (d : Int) * ((x : Int) / (d : Int)) = (x % d : Nat) := by
  rw [← Int.emod_def]
  norm_cast

private theorem nat_index_nonnegative (x : Nat) : ¬ (x : Int) < 0 := by omega

theorem prelude_initial_real (p : ForwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.regs .real [p.B] "s_real" =
      (unsignedPrelude p u).val.regs .real [p.B] "s_real" := by
  simp only [forwardPrelude, unsignedPrelude, BlockState.setReg, BlockState.readMem, ↓reduceDIte, ↓reduceIte, castTile_self]
  simp only [NumericDType.sub, ← Int.natCast_mul, ← Int.natCast_add]
  simp only [Int.ofNat_lt, nat_index_nonnegative, Nat.not_lt_zero, ↓reduceIte, source_mod, Int.toNat_natCast]
  try simp only [String.reduceEq, ↓reduceIte, source_mod_int, ← Int.natCast_mul,
    ← Int.natCast_add, Int.toNat_natCast]
  try rfl

theorem prelude_initial_imag (p : ForwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.regs .real [p.B] "s_imag" =
      (unsignedPrelude p u).val.regs .real [p.B] "s_imag" := by
  simp only [forwardPrelude, unsignedPrelude, BlockState.setReg, BlockState.readMem, ↓reduceDIte, ↓reduceIte, castTile_self]
  simp only [NumericDType.sub, ← Int.natCast_mul, ← Int.natCast_add]
  simp only [Int.ofNat_lt, nat_index_nonnegative, Nat.not_lt_zero, ↓reduceIte, source_mod, Int.toNat_natCast]
  try simp only [String.reduceEq, ↓reduceIte, source_mod_int, ← Int.natCast_mul,
    ← Int.natCast_add, Int.toNat_natCast]
  try rfl

theorem prelude_transition_real (p : ForwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.regs .real [p.B] "lambda_real" =
      (unsignedPrelude p u).val.regs .real [p.B] "lambda_real" := by
  simp only [forwardPrelude, unsignedPrelude, BlockState.setReg, BlockState.readMem, ↓reduceDIte, ↓reduceIte, castTile_self]
  simp only [NumericDType.sub, ← Int.natCast_mul, ← Int.natCast_add]
  simp only [Int.ofNat_lt, nat_index_nonnegative, Nat.not_lt_zero, ↓reduceIte, source_mod, Int.toNat_natCast]
  try simp only [String.reduceEq, ↓reduceIte, source_mod_int, ← Int.natCast_mul,
    ← Int.natCast_add, Int.toNat_natCast]
  try rfl

theorem prelude_transition_imag (p : ForwardLayout) (length : Nat) (u : BlockState) :
    (forwardPrelude p length u).val.regs .real [p.B] "lambda_imag" =
      (unsignedPrelude p u).val.regs .real [p.B] "lambda_imag" := by
  simp only [forwardPrelude, unsignedPrelude, BlockState.setReg, BlockState.readMem, ↓reduceDIte, ↓reduceIte, castTile_self]
  simp only [NumericDType.sub, ← Int.natCast_mul, ← Int.natCast_add]
  simp only [Int.ofNat_lt, nat_index_nonnegative, Nat.not_lt_zero, ↓reduceIte, source_mod, Int.toNat_natCast]
  try simp only [String.reduceEq, ↓reduceIte, source_mod_int, ← Int.natCast_mul,
    ← Int.natCast_add, Int.toNat_natCast]
  try rfl

#axiomsClean source_mod
#axiomsClean prelude_initial_real
#axiomsClean prelude_initial_imag
#axiomsClean prelude_transition_real
#axiomsClean prelude_transition_imag
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
