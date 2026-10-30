import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

/-- The source's mask-then-shift extraction equals an unsigned two-bit field. -/
theorem extracted_field (word shift : Nat) :
    (word &&& (3 <<< shift)) >>> shift = (word >>> shift) &&& 3 := by
  apply Nat.eq_of_testBit_eq
  intro bit
  simp [Nat.testBit_shiftRight, Nat.testBit_and, Nat.testBit_shiftLeft]

/-- Source int8 conversion of the extracted field, followed by its int8
subtraction of one. The finite result word keeps the original operand width. -/
def sourcePackedWeight (byte : Byte) (field : Nat) : Byte :=
  let extracted : Int := (((byte.toNat &&& (3 <<< (2*field))) >>> (2*field) : Nat) : Int)
  BitVec.ofInt 8 ((BitVec.ofInt 8 extracted).toInt - 1)

theorem sourcePackedWeight_eq (byte : Byte) (field : Nat) :
    sourcePackedWeight byte field = weightByte byte field := by
  unfold sourcePackedWeight
  rw [extracted_field]
  dsimp only
  have bounds : (byte.toNat >>> (2*field)) &&& 3 ≤ 3 := Nat.and_le_right
  have decoded : (BitVec.ofInt 8 (((byte.toNat >>> (2*field)) &&& 3 : Nat) : Int)).toInt =
      (((byte.toNat >>> (2*field)) &&& 3 : Nat) : Int) := by
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
  rw [decoded]
  rfl

theorem sourcePackedWeight_value (byte : Byte) (field : Nat) :
    (sourcePackedWeight byte field).toInt = packedWeight byte field := by
  rw [sourcePackedWeight_eq, weightByte_value]

def nativeModularBlock
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (c : Fin 15) (steps f j : Nat) (xs ys : Nat → Word) : Word :=
  blockStep (nativeArithmetic cast instruction) c steps f j xs ys 0

def nativeModularAccumulation
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (add32 : Word → Word → Word)
    (s : Shape) (c : Fin 15) (xs ys : Nat → Word) : Word :=
  foldN (fun f outer => foldN (fun j carry => add32 carry
    (nativeModularBlock cast instruction c (s.steps c) f j xs ys)) (s.steps c) outer) 4 0

theorem nativeBlock_reference
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (leaves : NumericLeaves cast instruction)
    (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs ys : Nat → Word) (defined : DefinedRow s.k xs)
    (f j : Nat) (hf : f < 4) (hj : j < s.steps c) :
    nativeModularBlock cast instruction c (s.steps c) f j xs ys =
      modularBlock c (s.steps c) f j xs ys := by
  unfold nativeModularBlock modularBlock blockStep
  apply foldN_congr_bound
  intro group hg seed
  unfold instructionStep
  simp only [nativeArithmetic, software, leaves.instruction]
  congr 1
  funext lane
  apply leaves.cast
  apply defined
  have masks := legal_all_masks s c legal f j (group*32+lane.val)
    hf hj (group_lane_bound c group hg lane)
  simpa only [Nat.add_assoc] using masks.2.2.1

theorem nativeAccumulation_reference
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (add32 : Word → Word → Word)
    (leaves : NumericLeaves cast instruction) (addLeaf : Add32Leaf add32)
    (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs ys : Nat → Word) (defined : DefinedRow s.k xs) :
    nativeModularAccumulation cast instruction add32 s c xs ys = modularAccumulation s c xs ys := by
  unfold nativeModularAccumulation modularAccumulation
  apply foldN_congr_bound
  intro f hf outer
  apply foldN_congr_bound
  intro j hj carry
  rw [nativeBlock_reference cast instruction leaves s c legal xs ys defined f j hf hj, addLeaf]

/-- Fixed-arity cast/K32/word-add leaves compose into the independent word
target. No whole-kernel lowering or dense-result equality is an assumption. -/
theorem nativeAccumulation_low32
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (add32 : Word → Word → Word)
    (leaves : NumericLeaves cast instruction) (addLeaf : Add32Leaf add32)
    (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs ys : Nat → Word) (defined : DefinedRow s.k xs) :
    nativeModularAccumulation cast instruction add32 s c xs ys = denseWordTarget s.k xs ys := by
  rw [nativeAccumulation_reference cast instruction add32 leaves addLeaf s c legal xs ys defined]
  exact modularAccumulation_dense s c legal xs ys defined

theorem nativeAccumulation_representable
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (add32 : Word → Word → Word)
    (leaves : NumericLeaves cast instruction) (addLeaf : Add32Leaf add32)
    (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs ys : Nat → Word) (defined : DefinedRow s.k xs)
    (lo : -2147483648 ≤ denseSignedSum s.k xs ys)
    (hi : denseSignedSum s.k xs ys ≤ 2147483647) :
    (nativeModularAccumulation cast instruction add32 s c xs ys).toInt = denseSignedSum s.k xs ys := by
  rw [nativeAccumulation_reference cast instruction add32 leaves addLeaf s c legal xs ys defined]
  exact modularAccumulation_representable s c legal xs ys defined lo hi

#axiomsClean sourcePackedWeight_eq
#axiomsClean sourcePackedWeight_value
#axiomsClean nativeBlock_reference
#axiomsClean nativeAccumulation_low32
#axiomsClean nativeAccumulation_representable
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
