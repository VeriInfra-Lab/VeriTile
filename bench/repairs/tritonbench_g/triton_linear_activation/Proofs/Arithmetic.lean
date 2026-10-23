import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Config
import VeriTile.Triton.Math.Erf
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- One result lane of one named, fixed-size ISA instruction. The FP32 TF32
profiles consume the actual raw FP32 operands: this lowering inserts no CVT.
TF32 interpretation is part of that fixed instruction, never an invented RN cast. -/
inductive Instruction where
  | syncTF32
  | wg32TF32 | wg64TF32 | wg128TF32 | wg256TF32
  | syncF16
  | wg32F16 | wg64F16 | wg128F16 | wg256F16
  deriving DecidableEq

def Instruction.width : Instruction → Nat
  | .syncTF32 | .wg32TF32 | .wg64TF32 | .wg128TF32 | .wg256TF32 => 8
  | _ => 16

def selectedInstruction (c : Fin 98) (mode : Mode) : Instruction :=
  if (configs c).bm≥64 ∧ (configs c).warps≥4 then
    match mode with
    | .fp32 => if (configs c).bn=32 then .wg32TF32 else if (configs c).bn=64 then .wg64TF32
        else if (configs c).bn=128 then .wg128TF32 else .wg256TF32
    | .autocastFP16 => if (configs c).bn=32 then .wg32F16 else if (configs c).bn=64 then .wg64F16
        else if (configs c).bn=128 then .wg128F16 else .wg256F16
  else match mode with | .fp32 => .syncTF32 | .autocastFP16 => .syncF16

theorem instruction_width (c : Fin 98) (mode : Mode) :
    (selectedInstruction c mode).width=match mode with | .fp32 => 8 | .autocastFP16 => 16 := by
  unfold selectedInstruction
  split_ifs <;> cases mode <;> simp only [Instruction.width] <;> split_ifs <;> rfl

theorem instruction_divides (c : Fin 98) (mode : Mode) :
    (configs c).bk/(selectedInstruction c mode).width*(selectedInstruction c mode).width=(configs c).bk := by
  have h:= config_bounds c
  rw [instruction_width]
  cases mode
  · exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h.2.2.2.2.2.2.1)
  · exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h.2.2.2.2.2.2.2)

inductive Literal where
  | zero | one | half | cubic | sqrtTwo | sqrtTwoOverPi
  deriving DecidableEq

structure Arithmetic (α : Type) where
  literal : Literal → α
  encode : Mode → α → α
  decode : Mode → α → α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  divFull : α → α → α
  maximum : α → α → α
  tanh : α → α
  erf : α → α
  instruction : (kind : Instruction) → (Fin kind.width → α) → (Fin kind.width → α) → α → α

def realArithmetic : Arithmetic ℝ where
  literal := fun l => match l with
    | .zero => 0 | .one => 1 | .half => 1/2 | .cubic => 0.044715
    | .sqrtTwo => Real.sqrt 2 | .sqrtTwoOverPi => Real.sqrt (2/Real.pi)
  encode := fun _ => id
  decode := fun _ => id
  add := (·+·)
  mul := (·*·)
  fma := fun x y z => x*y+z
  divFull := (·/·)
  maximum := max
  tanh := Real.tanh
  erf := VeriTile.Math.realErf
  instruction := fun _ xs ys seed => seed + ∑ i, xs i*ys i

/-- The independent mathematical activation, for all strings and None. -/
def realActivation (name : Option String) (x : ℝ) : ℝ :=
  if name=some "tanh" then Real.tanh x
  else if name=some "gelu" then x/2*(1+VeriTile.Math.realErf (x/Real.sqrt 2))
  else if name=some "fast_gelu" then
    x/2*(1+Real.tanh (Real.sqrt (2/Real.pi)*(x+0.044715*x^3)))
  else if name=some "relu" then max 0 x
  else x

/-- Selected default-fusion scalar graph. In fast-GELU the last cubic product
and addition contract to FMA. Libdevice calls and div.full remain scalar ports. -/
def activation (a : Arithmetic α) (name : Option String) (x : α) : α :=
  if name=some "tanh" then a.tanh x
  else if name=some "gelu" then
    a.mul (a.mul x (a.literal .half))
      (a.add (a.erf (a.divFull x (a.literal .sqrtTwo))) (a.literal .one))
  else if name=some "fast_gelu" then
    let cubic2 := a.mul x (a.mul x (a.literal .cubic))
    let inner := a.fma x cubic2 x
    a.mul (a.mul x (a.literal .half))
      (a.add (a.tanh (a.mul inner (a.literal .sqrtTwoOverPi))) (a.literal .one))
  else if name=some "relu" then a.maximum x (a.literal .zero)
  else x

theorem activation_real (name : Option String) (x : ℝ) :
    activation realArithmetic name x=realActivation name x := by
  unfold activation realActivation
  split_ifs <;> simp only [realArithmetic,max_comm]
  · ring
  · have inner : (x*(x*(x*0.044715))+x)*Real.sqrt (2/Real.pi)=
        Real.sqrt (2/Real.pi)*(x+0.044715*x^3) := by ring
    rw [inner]
    ring

/-- Explicit chronological loop, including the zero-trip case. -/
def foldN (step : Nat → α → α) : Nat → α → α
  | 0,initial => initial
  | n+1,initial => step n (foldN step n initial)

theorem foldN_congr (f g : Nat → α → α) (h : ∀ j x,f j x=g j x)
    (n : Nat) (z : α) : foldN f n z=foldN g n z := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,ih,h]

def instructionStep (a : Arithmetic α) (kind : Instruction) (K base : Nat)
    (xs ys : Nat → α) (seed : α) : α :=
  a.instruction kind
    (fun i => if base+i.val<K then xs (base+i.val) else a.literal .zero)
    (fun i => if base+i.val<K then ys (base+i.val) else a.literal .zero) seed

def blockStep (a : Arithmetic α) (c : Fin 98) (mode : Mode) (K block : Nat)
    (xs ys : Nat → α) (seed : α) : α :=
  let kind := selectedInstruction c mode
  foldN (fun group z => instructionStep a kind K
    (block*(configs c).bk+group*kind.width) xs ys z)
    ((configs c).bk/kind.width) seed

def accumulation (a : Arithmetic α) (c : Fin 98) (mode : Mode) (K : Nat)
    (xs ys : Nat → α) (bias : Option α) : α :=
  let seed := match bias with | none => a.literal .zero | some value => a.add value (a.literal .zero)
  foldN (fun block z => blockStep a c mode K block xs ys z) (ceilDiv K (configs c).bk) seed

theorem instructionStep_real (kind : Instruction) (K base : Nat) (xs ys : Nat → ℝ) (z : ℝ) :
    instructionStep realArithmetic kind K base xs ys z=
      z+∑ i∈Finset.range kind.width, if base+i<K then xs (base+i)*ys (base+i) else 0 := by
  unfold instructionStep
  simp only [realArithmetic]
  have per (i : Fin kind.width) :
      (if base+i.val<K then xs (base+i.val) else 0) *
      (if base+i.val<K then ys (base+i.val) else 0)=
      (if base+i.val<K then xs (base+i.val)*ys (base+i.val) else 0) := by
    split_ifs <;> simp
  simp_rw [per]
  exact congrArg (z+·) (Fin.sum_univ_eq_sum_range
    (fun i => if base+i<K then xs (base+i)*ys (base+i) else 0) kind.width)

 theorem chunk_fold_real (width count base : Nat) (f : Nat → ℝ) (z : ℝ) :
    foldN (fun j v => v+∑ i∈Finset.range width,f (base+j*width+i)) count z=
      z+∑ k∈Finset.range (count*width), f (base+k) := by
  induction count with
  | zero => simp [foldN]
  | succ n ih =>
    rw [foldN,ih,Nat.succ_mul,Finset.sum_range_add]
    simp only [Nat.add_assoc]
    ring

theorem blockStep_real (c : Fin 98) (mode : Mode) (K block : Nat) (xs ys : Nat → ℝ) (z : ℝ) :
    blockStep realArithmetic c mode K block xs ys z=
      z+∑ k∈Finset.range (configs c).bk,
        if block*(configs c).bk+k<K then xs (block*(configs c).bk+k)*ys (block*(configs c).bk+k) else 0 := by
  simp only [blockStep]
  simp_rw [instructionStep_real]
  simpa only [instruction_divides] using chunk_fold_real
    (selectedInstruction c mode).width ((configs c).bk/(selectedInstruction c mode).width)
    (block*(configs c).bk) (fun k => if k<K then xs k*ys k else 0) z

 theorem accumulation_real (c : Fin 98) (mode : Mode) (K : Nat)
    (xs ys : Nat → ℝ) (bias : Option ℝ) :
    accumulation realArithmetic c mode K xs ys bias=
      bias.getD 0 + ∑ k∈Finset.range K, xs k*ys k := by
  have hb : 0<(configs c).bk := by have := config_bounds c; omega
  have cover := ceil_cover K (configs c).bk hb
  have total (seed : ℝ) :
      foldN (fun block z => blockStep realArithmetic c mode K block xs ys z)
        (ceilDiv K (configs c).bk) seed = seed+∑ k∈Finset.range K,xs k*ys k := by
    simp_rw [blockStep_real]
    have h := chunk_fold_real (configs c).bk (ceilDiv K (configs c).bk) 0
      (fun k => if k<K then xs k*ys k else 0) seed
    simp only [Nat.zero_add] at h
    rw [h]
    congr 1
    calc
      _ = ∑ i∈Finset.range K, if i<K then xs i*ys i else 0 := by
        symm
        apply Finset.sum_subset (Finset.range_mono cover)
        intro i hi hnot
        have hn : ¬i<K := by simpa using hnot
        simp [hn]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro i hi
        simp only [Finset.mem_range] at hi
        simp [hi]
  simp only [accumulation]
  rw [total]
  cases bias <;> simp [realArithmetic]

#axiomsClean instruction_width
#axiomsClean instruction_divides
#axiomsClean activation_real
#axiomsClean instructionStep_real
#axiomsClean blockStep_real
#axiomsClean accumulation_real
end
end VeriTile.Bench.Repairs.TritonLinearActivation
