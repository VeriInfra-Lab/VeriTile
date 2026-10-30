import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Grid
import VeriTile.Triton.Float.BitValue
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000
abbrev Word := BitVec 32

/-- Exact finite FP32 decode followed by truncation toward zero. NaN and
infinities have no conversion result. No host floating point is used. -/
def truncateFloat (x : Word) : Option Int :=
  match (VeriTile.Triton.FP.Value.mk x : VeriTile.Triton.FP.Value .fp32).decode with
  | .finite negative magnitude =>
    let integer : Int := (magnitude.num.natAbs/magnitude.den : Nat)
    some (if negative then -integer else integer)
  | _ => none

def DefinedCast (x : Word) : Prop :=
  ∃ z,truncateFloat x=some z ∧ -128≤z ∧ z≤127

def referenceCast (x : Word) : Byte := BitVec.ofInt 8 ((truncateFloat x).getD 0)
def packedByte (x : Word) : Byte := BitVec.ofNat 8 x.toNat

theorem referenceCast_value (x : Word) (z : Int) (hz : truncateFloat x=some z)
    (lo : -128≤z) (hi : z≤127) : (referenceCast x).toInt=z := by
  simp only [referenceCast,hz,Option.getD_some]
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

/-- One result lane of one actual fixed-K32 signed integer instruction.
Saturation is local to the sync instruction; WGMMA wraps to low 32 bits.
The live accumulator seed is retained, including overflow transitions. -/
def referenceInstruction (kind : Instruction) (xs ys : Fin 32 → Byte) (seed : Word) : Word :=
  let exact : Int := seed.toInt+∑ i,(xs i).toInt*(ys i).toInt
  BitVec.ofInt 32 (if kind=.syncSat then max (-2147483648) (min 2147483647 exact) else exact)

def software : Arithmetic Word Word where
  cast := referenceCast
  packed := packedByte
  encode := id
  zero := 0
  instruction := referenceInstruction

/-- The two numeric leaf families. The conversion law is restricted to its
source-defined domain. Instruction laws have fixed 32-lane inputs and one
live accumulator, and impose no no-overflow requirement. -/
structure NumericLeaves (nativeCast : Word → Byte)
    (nativeInstruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word) : Prop where
  cast : ∀ x,DefinedCast x → nativeCast x=referenceCast x
  instruction : ∀ kind xs ys seed,nativeInstruction kind xs ys seed=referenceInstruction kind xs ys seed

def nativeArithmetic (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word) : Arithmetic Word Word where
  cast := cast
  packed := packedByte
  encode := id
  zero := 0
  instruction := instruction

def InputDefined (s : Shape) (b : Buffers) (memory : Memory Word) : Prop :=
  ∀ m k,m<s.m → k<s.k → DefinedCast (memory (aAddress b m k))

theorem native_accumulation (nc : Word → Byte)
    (ni : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (leaves : NumericLeaves nc ni) (s : Shape) (b : Buffers) (c : Fin 15)
    (legal : s.legal c) (memory : Memory Word) (defined : InputDefined s b memory)
    (m n : Nat) (hm : m<s.m) :
    callerAccumulation (nativeArithmetic nc ni) s b c memory m n=
      callerAccumulation software s b c memory m n := by
  unfold callerAccumulation accumulation
  apply foldN_congr_bound
  intro f hf z
  unfold fieldStep
  apply foldN_congr_bound
  intro j hj seed
  unfold blockStep
  apply foldN_congr_bound
  intro group hg acc
  unfold instructionStep
  simp only [nativeArithmetic,software,leaves.instruction]
  congr 1
  funext lane
  apply leaves.cast
  apply defined m _ hm
  have masks := legal_all_masks s c legal f j (group*32+lane.val) hf hj (group_lane_bound c group hg lane)
  simpa only [Nat.add_assoc] using masks.2.2.1

/-- Canonical finite-width reference: four fields, then fixed 32-wide chunks.
It does not refer to kernel blocks, tile sizes, masks, grids, or source memory. -/
def wordReference (kind : Instruction) (K : Nat) (xs ys : Nat → Word) : Word :=
  foldN (fun f z => foldN (fun q seed => referenceInstruction kind
    (fun lane => referenceCast (xs (f*(K/4)+q*32+lane.val)))
    (fun lane => weightByte (packedByte (ys (q*32+lane.val))) f) seed) (K/128) z) 4 0

theorem foldN_add (f : Nat → β → β) (n m : Nat) (z : β) :
    foldN f (n+m) z=foldN (fun i v => f (n+i) v) m (foldN f n z) := by
  induction m with
  | zero => rfl
  | succ m ih =>
    change f (n+m) (foldN f (n+m) z)=f (n+m) (foldN (fun i v => f (n+i) v) m (foldN f n z))
    rw [ih]

theorem foldN_chunks (f : Nat → β → β) (width count : Nat) (z : β) :
    foldN (fun j v => foldN (fun g u => f (j*width+g) u) width v) count z=
      foldN f (count*width) z := by
  induction count with
  | zero => simp only [Nat.zero_mul,foldN]
  | succ count ih => rw [foldN,ih,Nat.succ_mul,foldN_add]

theorem software_reference (s : Shape) (c : Fin 15) (legal : s.legal c) (xs ys : Nat → Word) :
    accumulation software s c xs ys=wordReference (selectedInstruction c) s.k xs ys := by
  have hdiv : (configs c).bk/32*32=(configs c).bk :=
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero (config_bounds c).2.2.2.2.2.2.1)
  have hk := legal_decomposition s c legal
  have hcount : s.steps c*((configs c).bk/32)=s.k/128 := by
    have he : s.k=(s.steps c*((configs c).bk/32))*128 := by
      rw [hk.2]
      calc
        4*((configs c).bk*s.steps c) = ((configs c).bk/32*32)*s.steps c*4 := by rw [hdiv]; ring
        _ = (s.steps c*((configs c).bk/32))*128 := by ring
    rw [he]; omega
  unfold accumulation wordReference
  apply foldN_congr_bound
  intro f hf z
  unfold fieldStep blockStep
  have stepEq (j group : Nat) (seed : Word) :
      instructionStep software (selectedInstruction c) (s.steps c) (configs c).bk f j group xs ys seed=
      referenceInstruction (selectedInstruction c)
        (fun lane => referenceCast (xs (f*(s.k/4)+(j*((configs c).bk/32)+group)*32+lane.val)))
        (fun lane => weightByte (packedByte (ys ((j*((configs c).bk/32)+group)*32+lane.val))) f) seed := by
    simp only [instructionStep,software,Nat.add_mul,Nat.mul_assoc,hdiv,hk.1]
    congr 1 <;> funext lane <;> congr 2 <;> ring
  simp_rw [stepEq]
  simpa only [hcount] using foldN_chunks
    (fun q seed => referenceInstruction (selectedInstruction c)
      (fun lane => referenceCast (xs (f*(s.k/4)+q*32+lane.val)))
      (fun lane => weightByte (packedByte (ys (q*32+lane.val))) f) seed)
    ((configs c).bk/32) (s.steps c) z

#axiomsClean referenceCast_value
#axiomsClean native_accumulation
#axiomsClean software_reference
#specNonCircular wordReference avoiding [accumulation,sourceAccumulation,run]
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
