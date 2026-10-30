import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.RawKernel
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.RawNative
open VeriTile.Triton FP.Structural FlashDecode2 Native KernelInterface
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def swapped (g : KernelInterface.Geometry) : Bool := decide (g.Q=2^31-1)

def accumulatorShape : Shape → Bool
  | .literal _ _ => Bool.true
  | .binary _ _ .add _ _ => Bool.true
  | _ => Bool.false

def schedule (g : KernelInterface.Geometry) : Schedule where
  fusion := fun _ => .left
  swap := fun s => match s with
    | .binary (some .fp32) .real .mul a _ => swapped g && accumulatorShape a
    | _ => Bool.false

/-- Total scalar PTX primitives. No result or tensor callback occurs here. -/
abbrev Ops := KernelNative.Ops
abbrev lower := KernelNative.lower

def fma (p : Ops) (g : KernelInterface.Geometry) (carry scale fresh : Word) : Word :=
  if swapped g then p.fmaRN scale carry fresh else p.fmaRN carry scale fresh

/-- Independent raw-word summary merge. Each field is one scalar instruction
family; exp32 includes rounded log2(e) multiplication followed by exp2.approx.
Every input word is retained, including NaNs, infinities and signed zeros. -/
structure Accumulator where
  maximum : Word
  denominator : Word
  numerator : Nat → Word

def initial : Accumulator := ⟨0xff800000,0,fun _ => 0⟩
def merge (p : Ops) (g : KernelInterface.Geometry) (x : Word) (v : Nat → Word) (a : Accumulator) : Accumulator :=
  let m := p.max x a.maximum
  let old := (KernelNative.Ops.exp32 p) (p.sub a.maximum m)
  let fresh := (KernelNative.Ops.exp32 p) (p.sub x m)
  ⟨m,fma p g a.denominator old fresh,
    fun d => fma p g (a.numerator d) old (p.mul fresh (v d))⟩
def scan (p : Ops) (g : KernelInterface.Geometry) (x : Nat → Word) (v : Nat → Nat → Word) : Nat → Accumulator
  | 0 => initial
  | n+1 => merge p g (x n) (v n) (scan p g x v n)

def target (p : Ops) (g : KernelInterface.Geometry) (lengths : Nat → Nat) (log mid : Nat → Word)
    (row d : Nat) : Word :=
  let a := scan p g (fun j => log (row*g.S+j))
    (fun j d => mid (row*g.S*g.D+j*g.D+d))
    ((lengths (row/g.H)+g.Q-1)/g.Q)
  p.div (a.numerator d) a.denominator

def entry (words : RegionName → Nat → Word) (other : State Term) : State Term :=
  inputState (fun r i => Term.leaf r i (words r i)) other

theorem carry_shape (s : State Term) (masked : Bool) (H S D BD n : Nat) :
    accumulatorShape (shape (values algebra s masked H S D BD n).z)=Bool.true ∧
    ∀ k,accumulatorShape (shape ((values algebra s masked H S D BD n).a k))=Bool.true := by
  cases n <;> simp [values,FlashDecode2.initial,zero,next,add,FlashDecode2.mul,algebra,shape,accumulatorShape]

theorem max_term (a b : Term) : maximum algebra a b=.binary (some .fp32) .real .max a b := rfl
theorem sub_term (a b : Term) : sub algebra a b=.binary (some .fp32) .real .sub a b := rfl
theorem mul_term (a b : Term) : FlashDecode2.mul algebra a b=.binary (some .fp32) .real .mul a b := rfl
theorem add_term (a b : Term) : add algebra a b=.binary (some .fp32) .real .add a b := rfl
theorem exp_term (a : Term) : exponential algebra a=.unary (some .fp32) .exp a := rfl

theorem eval_max (p : Ops) (g : KernelInterface.Geometry) (a b : Term) :
    eval (lower p) (schedule g) (.binary (some .fp32) .real .max a b)=
      p.max (eval (lower p) (schedule g) a) (eval (lower p) (schedule g) b) := rfl
theorem eval_sub (p : Ops) (g : KernelInterface.Geometry) (a b : Term) :
    eval (lower p) (schedule g) (.binary (some .fp32) .real .sub a b)=
      p.sub (eval (lower p) (schedule g) a) (eval (lower p) (schedule g) b) := rfl
theorem eval_exp (p : Ops) (g : KernelInterface.Geometry) (a : Term) :
    eval (lower p) (schedule g) (.unary (some .fp32) .exp a)=
      (KernelNative.Ops.exp32 p) (eval (lower p) (schedule g) a) := rfl
theorem eval_load (p : Ops) (g : KernelInterface.Geometry) (r : RegionName) (i : Nat) (w : Word) :
    eval (lower p) (schedule g) (.load (.leaf r i w))=w := rfl
theorem eval_fresh_mul (p : Ops) (g : KernelInterface.Geometry) (a b : Term) :
    eval (lower p) (schedule g) (.binary (some .fp32) .real .mul (.unary (some .fp32) .exp a) b)=
      p.mul ((KernelNative.Ops.exp32 p) (eval (lower p) (schedule g) a)) (eval (lower p) (schedule g) b) := by
  simp only [eval,binary,schedule,shape,accumulatorShape,Bool.and_false,Bool.false_and,Bool.false_eq_true,↓reduceIte,lower,KernelNative.lower,Numerical.lower]
theorem eval_acc (p : Ops) (g : KernelInterface.Geometry) (a scale fresh : Term)
    (h : accumulatorShape (shape a)=Bool.true) :
    eval (lower p) (schedule g) (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a scale) fresh) =
    fma p g (eval (lower p) (schedule g) a) (eval (lower p) (schedule g) scale)
      (eval (lower p) (schedule g) fresh) := by
  simp only [eval,schedule,fused,shape,h,Bool.and_true,fma,lower,KernelNative.lower,Numerical.lower]

theorem recurrence (p : Ops) (g : KernelInterface.Geometry) (words : RegionName → Nat → Word) (other : State Term)
    (row n : Nat) :
    let s := Grid.setup (g.B) (g.H) row (entry words other)
    let a := scan p g (fun j => words "E" (row*g.S+j))
      (fun j d => words "M"
        (row*g.S*g.D+j*g.D+d)) n
    eval (lower p) (schedule g) (values algebra s Bool.false (g.H)
      (g.S) (g.D) (g.D) n).m = a.maximum ∧
    eval (lower p) (schedule g) (values algebra s Bool.false (g.H)
      (g.S) (g.D) (g.D) n).z = a.denominator ∧
    ∀ k,eval (lower p) (schedule g) ((values algebra s Bool.false (g.H)
      (g.S) (g.D) (g.D) n).a k) = a.numerator (lane k) := by
  have hx (j : Nat) :
      logvalue algebra (Grid.setup (g.B) (g.H) row (entry words other))
        (g.H) (g.S) j =
      .load (.leaf "E" (row*g.S+j)
        (words "E" (row*g.S+j))) := by
    simp only [logvalue,row_log_offset]
    rfl
  have hv (j : Nat) (k : TileIndex [g.D]) :
      FlashDecode2.value algebra
        (Grid.setup (g.B) (g.H) row (entry words other))
        Bool.false (g.H) (g.S) (g.D) j k =
      .load (.leaf "M" (row*g.S*g.D+j*g.D+lane k)
        (words "M" (row*g.S*g.D+j*g.D+lane k))) := by
    simp only [FlashDecode2.value,active,true_or,↓reduceIte,row_mid_offset]
    rfl
  induction n with
  | zero => simp [values,FlashDecode2.initial,zero,algebra,eval,lower,KernelNative.lower,Numerical.lower,scan,initial]
  | succ n ih =>
    dsimp only at ih ⊢
    have shapes := carry_shape
      (Grid.setup (g.B) (g.H) row (entry words other))
      Bool.false (g.H) (g.S) (g.D) (g.D) n
    simp only [values,next,hx,hv,max_term,exp_term,sub_term,mul_term,add_term]
    simp only [eval_max,eval_sub,eval_exp,eval_load,eval_acc,eval_fresh_mul,shapes.1,shapes.2,
      scan,merge,ih.1,ih.2.1,ih.2.2]
    simp

theorem value (p : Ops) (g : KernelInterface.Geometry) (words : RegionName → Nat → Word) (other : State Term)
    (bounds : LengthBounds g other) (row : Nat) (hr : row<g.B*g.H)
    (k : TileIndex [g.D]) :
    eval (lower p) (schedule g) (Grid.value algebra (entry words other) Bool.false
      (g.B) (g.H) (g.S)
      (g.D) g.Q (g.D) row k) =
    target p g (fun b => (other.mem "L" b).read .nat) (words "E")
      (words "M") row (lane k) := by
  have hb : LengthBounds g (entry words other) := by
    simpa only [LengthBounds,entry,inputState,show ¬(("L" : RegionName)="E" ∨ "L"="M") by decide,if_false] using bounds
  have counted := count_bound g (entry words other) hb row hr
  have recur := recurrence p g words other row (steps g other row)
  simp only [Grid.value,output,counted.1,divide,algebra,eval,binary,schedule,
    lower,KernelNative.lower,Numerical.lower,Bool.false_and,↓reduceIte]
  exact congrArg₂ p.div (recur.2.2 k) recur.2.1

#axiomsClean recurrence
#axiomsClean value
#specNonCircular target avoiding [Grid.source,FP.Structural.exec,values]
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.RawNative
