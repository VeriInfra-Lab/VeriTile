import bench.tritonbench_g.flash_decode2_llama.Proofs.KernelTensor
import bench.tritonbench_g.flash_decode2_phi.Proofs.Numerical

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.KernelNative
open VeriTile.Triton FP.Structural FlashDecode2 Native
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Two observed fixed instruction patterns: ordinary loop-carried products,
or the maximal-Q single-step form with the constant multiplicand second. -/
def swapped (c : Call) : Bool := decide (c.val.blockSeq=2^31-1)

def accumulatorShape : Shape → Bool
  | .literal _ _ => Bool.true
  | .binary _ _ .add _ _ => Bool.true
  | _ => Bool.false

def schedule (c : Call) : Schedule where
  fusion := fun _ => .left
  swap := fun s => match s with
    | .binary (some .fp32) .real .mul a _ => swapped c && accumulatorShape a
    | _ => Bool.false

/-- Total scalar PTX primitives. No result or tensor callback occurs here. -/
structure Ops where
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  max : Word → Word → Word
  exp2 : Word → Word
  fmaRN : Word → Word → Word → Word

def Ops.exp32 (p : Ops) (x : Word) : Word := p.exp2 (p.mul x 0x3fb8aa3b)

def lower (p : Ops) : Instructions :=
  Numerical.lower ⟨fun _ _ => 0,p.sub,p.mul,p.div,p.max,p.exp32,p.fmaRN⟩

def fma (p : Ops) (c : Call) (carry scale fresh : Word) : Word :=
  if swapped c then p.fmaRN scale carry fresh else p.fmaRN carry scale fresh

/-- Independent raw-word summary merge. Each field is one scalar instruction
family; exp32 includes rounded log2(e) multiplication followed by exp2.approx.
Every input word is retained, including NaNs, infinities and signed zeros. -/
structure Accumulator where
  maximum : Word
  denominator : Word
  numerator : Nat → Word

def initial : Accumulator := ⟨0xff800000,0,fun _ => 0⟩
def merge (p : Ops) (c : Call) (x : Word) (v : Nat → Word) (a : Accumulator) : Accumulator :=
  let m := p.max x a.maximum
  let old := p.exp32 (p.sub a.maximum m)
  let fresh := p.exp32 (p.sub x m)
  ⟨m,fma p c a.denominator old fresh,
    fun d => fma p c (a.numerator d) old (p.mul fresh (v d))⟩
def scan (p : Ops) (c : Call) (x : Nat → Word) (v : Nat → Nat → Word) : Nat → Accumulator
  | 0 => initial
  | n+1 => merge p c (x n) (v n) (scan p c x v n)

def target (p : Ops) (c : Call) (lengths : Nat → Int) (log mid : Nat → Word)
    (row d : Nat) : Word :=
  let a := scan p c (fun j => log (row*Checked.capacity c.val+j))
    (fun j d => mid (row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+d))
    (KernelModel.iterations c lengths row)
  p.div (a.numerator d) a.denominator

def entry (c : Call) (heap : KernelTensor.Heap) (other : State Term) :=
  KernelTensor.lift Term.leaf c heap other

theorem carry_shape (s : State Term) (masked : Bool) (H S D BD n : Nat) :
    accumulatorShape (shape (values algebra s masked H S D BD n).z)=Bool.true ∧
    ∀ k,accumulatorShape (shape ((values algebra s masked H S D BD n).a k))=Bool.true := by
  cases n <;> simp [values,FlashDecode2.initial,zero,next,add,FlashDecode2.mul,algebra,shape,accumulatorShape]

theorem max_term (a b : Term) : maximum algebra a b=.binary (some .fp32) .real .max a b := rfl
theorem sub_term (a b : Term) : sub algebra a b=.binary (some .fp32) .real .sub a b := rfl
theorem mul_term (a b : Term) : FlashDecode2.mul algebra a b=.binary (some .fp32) .real .mul a b := rfl
theorem add_term (a b : Term) : add algebra a b=.binary (some .fp32) .real .add a b := rfl
theorem exp_term (a : Term) : exponential algebra a=.unary (some .fp32) .exp a := rfl

theorem eval_max (p : Ops) (c : Call) (a b : Term) :
    eval (lower p) (schedule c) (.binary (some .fp32) .real .max a b)=
      p.max (eval (lower p) (schedule c) a) (eval (lower p) (schedule c) b) := rfl
theorem eval_sub (p : Ops) (c : Call) (a b : Term) :
    eval (lower p) (schedule c) (.binary (some .fp32) .real .sub a b)=
      p.sub (eval (lower p) (schedule c) a) (eval (lower p) (schedule c) b) := rfl
theorem eval_exp (p : Ops) (c : Call) (a : Term) :
    eval (lower p) (schedule c) (.unary (some .fp32) .exp a)=
      p.exp32 (eval (lower p) (schedule c) a) := rfl
theorem eval_load (p : Ops) (c : Call) (r : RegionName) (i : Nat) (w : Word) :
    eval (lower p) (schedule c) (.load (.leaf r i w))=w := rfl
theorem eval_fresh_mul (p : Ops) (c : Call) (a b : Term) :
    eval (lower p) (schedule c) (.binary (some .fp32) .real .mul (.unary (some .fp32) .exp a) b)=
      p.mul (p.exp32 (eval (lower p) (schedule c) a)) (eval (lower p) (schedule c) b) := by
  simp only [eval,binary,schedule,shape,accumulatorShape,Bool.and_false,Bool.false_and,Bool.false_eq_true,↓reduceIte,lower,Numerical.lower]
theorem eval_acc (p : Ops) (c : Call) (a scale fresh : Term)
    (h : accumulatorShape (shape a)=Bool.true) :
    eval (lower p) (schedule c) (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a scale) fresh) =
    fma p c (eval (lower p) (schedule c) a) (eval (lower p) (schedule c) scale)
      (eval (lower p) (schedule c) fresh) := by
  simp only [eval,schedule,fused,shape,h,Bool.and_true,fma,lower,Numerical.lower]

theorem recurrence (p : Ops) (c : Call) (heap : KernelTensor.Heap) (other : State Term)
    (row n : Nat) :
    let s := Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other)
    let a := scan p c (fun j => KernelTensor.word c heap "E" (row*Checked.capacity c.val+j))
      (fun j d => KernelTensor.word c heap "M"
        (row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+d)) n
    eval (lower p) (schedule c) (values algebra s c.val.masked (Checked.heads c.val)
      (Checked.capacity c.val) (Checked.width c.val) (Checked.block c.val) n).m = a.maximum ∧
    eval (lower p) (schedule c) (values algebra s c.val.masked (Checked.heads c.val)
      (Checked.capacity c.val) (Checked.width c.val) (Checked.block c.val) n).z = a.denominator ∧
    ∀ k,eval (lower p) (schedule c) ((values algebra s c.val.masked (Checked.heads c.val)
      (Checked.capacity c.val) (Checked.width c.val) (Checked.block c.val) n).a k) = a.numerator (lane k) := by
  have hx (j : Nat) :
      logvalue algebra (Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other))
        (Checked.heads c.val) (Checked.capacity c.val) j =
      .load (.leaf "E" (row*Checked.capacity c.val+j)
        (KernelTensor.word c heap "E" (row*Checked.capacity c.val+j))) := by
    simp only [logvalue,row_log_offset]
    rfl
  have hv (j : Nat) (k : TileIndex [Checked.block c.val]) :
      FlashDecode2.value algebra
        (Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other))
        c.val.masked (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) j k =
      .load (.leaf "M" (row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+lane k)
        (KernelTensor.word c heap "M" (row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+lane k))) := by
    simp only [FlashDecode2.value,active,c.property,true_or,↓reduceIte,row_mid_offset]
    rfl
  induction n with
  | zero => simp [values,FlashDecode2.initial,zero,algebra,eval,lower,Numerical.lower,scan,initial]
  | succ n ih =>
    dsimp only at ih ⊢
    have shapes := carry_shape
      (Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other))
      c.val.masked (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) (Checked.block c.val) n
    simp only [values,next,hx,hv,max_term,exp_term,sub_term,mul_term,add_term]
    simp only [eval_max,eval_sub,eval_exp,eval_load,eval_acc,eval_fresh_mul,shapes.1,shapes.2,
      scan,merge,ih.1,ih.2.1,ih.2.2]
    simp

theorem value (p : Ops) (c : Call) (heap : KernelTensor.Heap) (other : State Term)
    (h : InputContract c (KernelTensor.lengths c heap)) (row : Nat) (hr : row<KernelModel.rows c)
    (k : TileIndex [Checked.block c.val]) :
    eval (lower p) (schedule c) (Grid.value algebra (entry c heap other) c.val.masked
      (Checked.batches c.val) (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) c.val.blockSeq (Checked.block c.val) row k) =
    target p c (KernelTensor.lengths c heap) (KernelTensor.word c heap "E")
      (KernelTensor.word c heap "M") row (lane k) := by
  have counted := KernelModel.count_bound c _ h (entry c heap other)
    (KernelTensor.lift_length _ c heap other) row hr
  have recur := recurrence p c heap other row (KernelModel.iterations c (KernelTensor.lengths c heap) row)
  simp only [Grid.value,output,counted.1,divide,algebra,eval,binary,schedule,
    lower,Numerical.lower,Bool.false_and,↓reduceIte]
  exact congrArg₂ p.div (recur.2.2 k) recur.2.1

#axiomsClean recurrence
#axiomsClean value
#specNonCircular target avoiding [KernelModel.run,Grid.source,FP.Structural.exec,values,eval]
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.KernelNative
