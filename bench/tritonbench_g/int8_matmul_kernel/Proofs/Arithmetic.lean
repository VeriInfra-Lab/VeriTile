import bench.tritonbench_g.int8_matmul_kernel.Proofs.BlockBound

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent signed integer sum of numerically truncated FP32 inputs and
unsigned two-bit fields minus one. There is no configuration or execution in it. -/
def denseSignedSum (K : Nat) (xs ys : Nat → Word) : Int :=
  ∑ f ∈ Finset.range 4, ∑ q ∈ Finset.range (K/4),
    ((truncateFloat (xs (f*(K/4)+q))).getD 0) * packedWeight (packedByte (ys q)) f

def denseWordTarget (K : Nat) (xs ys : Nat → Word) : Word :=
  BitVec.ofInt 32 (denseSignedSum K xs ys)

def DefinedRow (K : Nat) (xs : Nat → Word) : Prop :=
  ∀ k, k < K → DefinedCast (xs k)

theorem dense_cast_connection (K : Nat) (xs ys : Nat → Word)
    (defined : DefinedRow K xs) :
    denseTarget referenceCast packedByte K xs ys = denseSignedSum K xs ys := by
  unfold denseTarget denseSignedSum
  apply Finset.sum_congr rfl
  intro f hf
  apply Finset.sum_congr rfl
  intro q hq
  have hf' : f < 4 := Finset.mem_range.mp hf
  have hq' : q < K/4 := Finset.mem_range.mp hq
  have hidx : f*(K/4)+q < K := by
    have hd := Nat.div_mul_le_self K 4
    nlinarith
  obtain ⟨z, hz, hlo, hhi⟩ := defined _ hidx
  rw [referenceCast_value _ z hz hlo hhi, hz]
  rfl

/-- This is the candidate's source-order block operation: a complete existing
BK dot from zero, before the separate modular accumulator add. -/
def modularBlock (c : Fin 15) (steps f j : Nat) (xs ys : Nat → Word) : Word :=
  blockStep software c steps f j xs ys 0

def modularAccumulation (s : Shape) (c : Fin 15) (xs ys : Nat → Word) : Word :=
  foldN (fun f outer => foldN (fun j carry => ptxAdd32 carry
    (modularBlock c (s.steps c) f j xs ys)) (s.steps c) outer) 4 0

def exactArithmetic : Arithmetic Word Int :=
  mathematical referenceCast packedByte (BitVec.ofInt 32)

theorem modularBlock_exact (c : Fin 15) (steps f j : Nat) (xs ys : Nat → Word) :
    modularBlock c steps f j xs ys =
      BitVec.ofInt 32 (blockStep exactArithmetic c steps f j xs ys 0) := by
  let left : Nat → Fin 32 → Byte := fun group lane =>
    referenceCast (xs ((f*steps+j)*(configs c).bk+group*32+lane.val))
  let right : Nat → Fin 32 → Byte := fun group lane =>
    weightByte (packedByte (ys (j*(configs c).bk+group*32+lane.val))) f
  have weights : ∀ group lane, -1 ≤ (right group lane).toInt ∧ (right group lane).toInt ≤ 2 := by
    intro group lane
    dsimp only [right]
    rw [weightByte_value]
    exact weight_bounds _ _
  have count : (configs c).bk/32 ≤ 4 := by
    have bounds := config_bounds c
    omega
  have concrete := zeroSeededBlock_exact (selectedInstruction c) left right
    ((configs c).bk/32) count weights
  have math_value : blockStep exactArithmetic c steps f j xs ys 0 =
      blockPrefix left right ((configs c).bk/32) := by
    simp only [blockStep, instructionStep, exactArithmetic, mathematical, fold_sum,
      zero_add, blockPrefix, chunkSum, left, right]
  rw [math_value]
  exact concrete

theorem exactBlock_seed (c : Fin 15) (steps f j : Nat) (xs ys : Nat → Word) (seed : Int) :
    blockStep exactArithmetic c steps f j xs ys seed =
      seed + blockStep exactArithmetic c steps f j xs ys 0 := by
  simp only [exactArithmetic, blockStep_integer, zero_add]

theorem foldN_word_lift (wordStep : Nat → Word → Word) (integerStep : Nat → Int → Int)
    (step : ∀ i seed, wordStep i (BitVec.ofInt 32 seed) =
      BitVec.ofInt 32 (integerStep i seed)) (count : Nat) (seed : Int) :
    foldN wordStep count (BitVec.ofInt 32 seed) = BitVec.ofInt 32 (foldN integerStep count seed) := by
  induction count with
  | zero => rfl
  | succ count ih => rw [foldN, ih, step, foldN]

theorem modularField_exact (c : Fin 15) (steps f : Nat) (xs ys : Nat → Word) (seed : Int) :
    foldN (fun j carry => ptxAdd32 carry (modularBlock c steps f j xs ys)) steps
      (BitVec.ofInt 32 seed) = BitVec.ofInt 32 (fieldStep exactArithmetic c steps f xs ys seed) := by
  unfold fieldStep
  apply foldN_word_lift
  intro j carry
  rw [modularBlock_exact]
  unfold ptxAdd32
  rw [← BitVec.ofInt_add, exactBlock_seed c steps f j xs ys carry]

theorem modularAccumulation_dense (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs ys : Nat → Word) (defined : DefinedRow s.k xs) :
    modularAccumulation s c xs ys = denseWordTarget s.k xs ys := by
  have lifted : modularAccumulation s c xs ys =
      BitVec.ofInt 32 (accumulation exactArithmetic s c xs ys) := by
    unfold modularAccumulation accumulation
    change foldN (fun f outer => foldN (fun j carry => ptxAdd32 carry
      (modularBlock c (s.steps c) f j xs ys)) (s.steps c) outer) 4 (BitVec.ofInt 32 0) =
      BitVec.ofInt 32 (foldN (fun f z => fieldStep exactArithmetic c (s.steps c) f xs ys z) 4 0)
    apply foldN_word_lift
    intro f seed
    exact modularField_exact c (s.steps c) f xs ys seed
  rw [lifted]
  unfold exactArithmetic
  rw [accumulation_integer _ _ _ _ _ legal, dense_cast_connection _ _ _ defined]
  rfl

/-- Representability is a corollary premise, not an additional input or K guard. -/
theorem modularAccumulation_representable (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs ys : Nat → Word) (defined : DefinedRow s.k xs)
    (lo : -2147483648 ≤ denseSignedSum s.k xs ys)
    (hi : denseSignedSum s.k xs ys ≤ 2147483647) :
    (modularAccumulation s c xs ys).toInt = denseSignedSum s.k xs ys := by
  rw [modularAccumulation_dense s c legal xs ys defined]
  unfold denseWordTarget
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

#axiomsClean dense_cast_connection
#axiomsClean modularBlock_exact
#axiomsClean modularAccumulation_dense
#axiomsClean modularAccumulation_representable
#specNonCircular denseSignedSum avoiding [modularBlock, modularAccumulation, blockStep, fieldStep, accumulation]
#specNonCircular denseWordTarget avoiding [modularBlock, modularAccumulation, blockStep, fieldStep, accumulation]
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
