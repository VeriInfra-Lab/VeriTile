import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Divisibility
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
abbrev Byte := BitVec 8

/-- Actual selected SM90 integer instructions; all ports are exactly K32.
BN256 is split into n128 WGMMA operations. Only the sync port is satfinite. -/
inductive Instruction where
  | syncSat | wg32 | wg64 | wg128
  deriving DecidableEq

def selectedInstruction (c : Fin 15) : Instruction :=
  if (configs c).bm<64 then .syncSat
  else if (configs c).bn=32 then .wg32
  else if (configs c).bn=64 then .wg64 else .wg128

def packedWeight (byte : Byte) (field : Nat) : Int :=
  (((byte.toNat >>> (2*field)) &&& 3 : Nat) : Int)-1

def weightByte (byte : Byte) (field : Nat) : Byte := BitVec.ofInt 8 (packedWeight byte field)

theorem weight_bounds (byte : Byte) (field : Nat) : -1≤packedWeight byte field ∧ packedWeight byte field≤2 := by
  have h : (byte.toNat >>> (2*field)) &&& 3≤3 := Nat.and_le_right
  unfold packedWeight
  omega

theorem weightByte_value (byte : Byte) (field : Nat) : (weightByte byte field).toInt=packedWeight byte field := by
  have h := weight_bounds byte field
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

/-- Input and accumulator types are separate: FP32 loads numerically become
signed8, while the accumulator is signed32 (or its mathematical Int projection).
Packed loads keep their unsigned-byte interpretation. -/
structure Arithmetic (α β : Type) where
  cast : α → Byte
  packed : α → Byte
  encode : β → α
  zero : β
  instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → β → β

def mathematical (cast packed : α → Byte) (encode : Int → α) : Arithmetic α Int where
  cast := cast
  packed := packed
  encode := encode
  zero := 0
  instruction := fun _ xs ys seed => seed+∑ i,(xs i).toInt * (ys i).toInt

/-- Source-order recurrence, including a zero-trip loop. -/
def foldN (step : Nat → β → β) : Nat → β → β
  | 0,z => z
  | n+1,z => step n (foldN step n z)

theorem foldN_congr (f g : Nat → β → β) (h : ∀ i z,f i z=g i z) (n : Nat) (z : β) :
    foldN f n z=foldN g n z := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,ih,h]

def instructionStep (a : Arithmetic α β) (kind : Instruction) (steps bk f j group : Nat)
    (xs ys : Nat → α) (seed : β) : β :=
  a.instruction kind
    (fun i => a.cast (xs ((f*steps+j)*bk+group*32+i.val)))
    (fun i => weightByte (a.packed (ys (j*bk+group*32+i.val))) f) seed

def blockStep (a : Arithmetic α β) (c : Fin 15) (steps f j : Nat)
    (xs ys : Nat → α) (seed : β) : β :=
  foldN (fun group z => instructionStep a (selectedInstruction c) steps (configs c).bk f j group xs ys z)
    ((configs c).bk/32) seed

def fieldStep (a : Arithmetic α β) (c : Fin 15) (steps f : Nat) (xs ys : Nat → α) (seed : β) : β :=
  foldN (fun j z => blockStep a c steps f j xs ys z) steps seed

def accumulation (a : Arithmetic α β) (s : Shape) (c : Fin 15) (xs ys : Nat → α) : β :=
  foldN (fun f z => fieldStep a c (s.steps c) f xs ys z) 4 a.zero

/-- Independent field-major quantized dense product. No kernel/ISA execution
appears in this specification. Native signed32 overflow is a different layer. -/
def denseTarget (cast packed : α → Byte) (K : Nat) (xs ys : Nat → α) : Int :=
  ∑ f∈Finset.range 4, ∑ k∈Finset.range (K/4),
    (cast (xs (f*(K/4)+k))).toInt*packedWeight (packed (ys k)) f

theorem instructionStep_integer (cast packed : α → Byte) (encode : Int → α)
    (kind : Instruction) (steps bk f j group : Nat) (xs ys : Nat → α) (seed : Int) :
    instructionStep (mathematical cast packed encode) kind steps bk f j group xs ys seed=
      seed+∑ k∈Finset.range 32,
        (cast (xs (f*(steps*bk)+(j*bk+group*32+k)))).toInt*
        packedWeight (packed (ys (j*bk+group*32+k))) f := by
  simp only [instructionStep,mathematical,weightByte_value]
  simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using congrArg (seed+·)
    (Fin.sum_univ_eq_sum_range (fun k =>
      (cast (xs ((f*steps+j)*bk+group*32+k))).toInt*
        packedWeight (packed (ys (j*bk+group*32+k))) f) 32)

 theorem chunk_fold (width count base : Nat) (term : Nat → Int) (seed : Int) :
    foldN (fun j z => z+∑ i∈Finset.range width,term (base+j*width+i)) count seed=
      seed+∑ k∈Finset.range (count*width),term (base+k) := by
  induction count with
  | zero => simp [foldN]
  | succ n ih =>
    rw [foldN,ih,Nat.succ_mul,Finset.sum_range_add]
    simp only [Nat.add_assoc]
    ring

theorem blockStep_integer (cast packed : α → Byte) (encode : Int → α)
    (c : Fin 15) (steps f j : Nat) (xs ys : Nat → α) (seed : Int) :
    blockStep (mathematical cast packed encode) c steps f j xs ys seed=
      seed+∑ k∈Finset.range (configs c).bk,
        (cast (xs (f*(steps*(configs c).bk)+(j*(configs c).bk+k)))).toInt*
        packedWeight (packed (ys (j*(configs c).bk+k))) f := by
  have hd : (configs c).bk/32*32=(configs c).bk :=
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero (config_bounds c).2.2.2.2.2.2.1)
  simp only [blockStep]
  simp_rw [instructionStep_integer]
  simpa only [hd] using chunk_fold 32 ((configs c).bk/32) (j*(configs c).bk)
    (fun k => (cast (xs (f*(steps*(configs c).bk)+k))).toInt*packedWeight (packed (ys k)) f) seed

theorem fieldStep_integer (cast packed : α → Byte) (encode : Int → α)
    (c : Fin 15) (steps f : Nat) (xs ys : Nat → α) (seed : Int) :
    fieldStep (mathematical cast packed encode) c steps f xs ys seed=
      seed+∑ k∈Finset.range (steps*(configs c).bk),
        (cast (xs (f*(steps*(configs c).bk)+k))).toInt*packedWeight (packed (ys k)) f := by
  simp only [fieldStep]
  simp_rw [blockStep_integer]
  simpa only [Nat.zero_add] using chunk_fold (configs c).bk steps 0
    (fun k => (cast (xs (f*(steps*(configs c).bk)+k))).toInt*packedWeight (packed (ys k)) f) seed

theorem fold_sum (term : Nat → Int) (n : Nat) (seed : Int) :
    foldN (fun i z => z+term i) n seed=seed+∑ i∈Finset.range n,term i := by
  induction n with
  | zero => simp [foldN]
  | succ n ih => simp only [foldN,ih,Finset.sum_range_succ]; ring

theorem accumulation_integer (cast packed : α → Byte) (encode : Int → α)
    (s : Shape) (c : Fin 15) (legal : s.legal c) (xs ys : Nat → α) :
    accumulation (mathematical cast packed encode) s c xs ys=denseTarget cast packed s.k xs ys := by
  have hk : s.steps c*(configs c).bk=s.k/4 := by
    simpa only [Nat.mul_comm] using (legal_decomposition s c legal).1.symm
  simp only [accumulation]
  simp_rw [fieldStep_integer]
  rw [fold_sum]
  simp only [mathematical,zero_add,hk,denseTarget]

#axiomsClean weightByte_value
#axiomsClean accumulation_integer
#specNonCircular denseTarget avoiding [accumulation,fieldStep,blockStep,instructionStep]
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
