import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Kernels
import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Source

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

/-- Literal source graph. Dot and following addition are distinct constructors;
the native interpretation below states the selected fusion explicitly. -/
inductive Term (α : Type) where
  | input (value : α)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary)
      (left right : Term α)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (value : Term α)
  | load (value : Term α)
  | dot (precision : Option ComputeDType) (options : DotOptions) (inner : Nat)
      (left right : Fin inner → Term α)

instance [Inhabited α] : Inhabited (Term α) := ⟨.input default⟩

def dotTerms {α : Type} (precision : Option ComputeDType) (options : DotOptions) :
    {batch : TileShape} → {m k n : Nat} →
    Values (Term α) .real (batch++[m,k]) → Values (Term α) .real (batch++[k,n]) →
    Option (Values (Term α) .real (batch++[m,n]))
  | [],_,k,_,left,right => some (fun i =>
      .dot precision options k (fun d => left (i.1,d,PUnit.unit))
        (fun d => right (d,i.2.1,PUnit.unit)))
  | _::_,_,_,_,_,_ => none

def algebra {α : Type} [Inhabited α] : Algebra (Term α) where
  literal := .literal
  negInf := .literal none .real 0
  binary := .binary
  unary := fun _ _ value => value
  cast := .cast
  fromNat := fun _ _ => .input default
  fromInt := fun _ _ => .input default
  fp32Bits := fun _ => .input default
  fp32Load := .load
  reduceMax := fun _ {_} _ _ _ _ => .input default
  reduceSum := fun _ {_} _ _ _ _ => .input default
  dot := fun p {_} {_} {_} {_} => dotTerms p {}
  dotWithOptions := fun p options {_} {_} {_} {_} => dotTerms p options

/-- Ordered scalar FMA expansion with the literal seed. -/
def finChain (ops : Ops α) (n : Nat) (left right : Fin n → α) (seed : α) : α :=
  chain ops n
    (fun d => if hd : d<n then left ⟨d,hd⟩ else ops.zero)
    (fun d => if hd : d<n then right ⟨d,hd⟩ else ops.zero) seed

/-- Explicit selected native lowering: a source add with a right-hand dot
uses that dot's accumulator seed. Other binary additions are unreachable in
this kernel and have no invented arithmetic law. Cast and load nodes persist
in the source graph even when the selected FP32 interpretation is identity. -/
def Term.evaluate (ops : Ops α) (literal : ℝ → α) : Term α → α
  | .input value => value
  | .literal _ _ x => literal x
  | .binary _ _ .mul a b => ops.mul (a.evaluate ops literal) (b.evaluate ops literal)
  | .binary _ _ .add a (.dot _ _ n left right) =>
      finChain ops n (fun d => (left d).evaluate ops literal)
        (fun d => (right d).evaluate ops literal) (a.evaluate ops literal)
  | .binary _ _ _ _ _ => ops.zero
  | .cast _ _ _ value => ops.cast (value.evaluate ops literal)
  | .load value => value.evaluate ops literal
  | .dot _ _ n left right =>
      finChain ops n (fun d => (left d).evaluate ops literal)
        (fun d => (right d).evaluate ops literal) ops.zero

theorem finChain_eq (ops : Ops α) (n : Nat) (left right : Nat → α) (seed : α) :
    finChain ops n (fun d => left d.val) (fun d => right d.val) seed=
      chain ops n left right seed := by
  apply chain_congr
  · intro i hi; simp only [dif_pos hi]
  · intro i hi; simp only [dif_pos hi]

#axiomsClean finChain_eq
#specNonCircular Term.evaluate avoiding [FP.Structural.exec,Source.run]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
