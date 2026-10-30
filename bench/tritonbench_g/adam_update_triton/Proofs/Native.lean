import bench.tritonbench_g.adam_update_triton.Proofs.CommutedTile
import bench.tritonbench_g.adam_update_triton.Proofs.Math
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_Observed

/-! The v003 repair normalizes every accepted numeric payload to Python float
before launch. This is the resulting runtime-FP32 scalar graph; integer to
binary64 and binary64 to FP32 remain distinct local conversions in Scalar.
Rejected/overflowing conversions and empty calls retain their host branches. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
noncomputable section
set_option maxRecDepth 24000

abbrev Word := BitVec 32

/-- Typed graph constructors for this source. No arithmetic oracle is used. -/
def graphAlgebra : Algebra (Term α) where
  literal := Term.literal
  negInf := .unused
  binary := Term.binary
  unary := Term.unary
  cast := Term.cast
  fromNat := fun _ _ => .unused
  fromInt := fun _ _ => .unused
  fp32Bits := fun _ => .unused
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => .unused
  reduceSum := fun _ {_} _ _ _ => fun _ => .unused

/-- Local instruction ports of the pinned CUDA profile. neg.f32 and
sub.f32(+0,lr) are deliberately different operations. Comparisons retain
unordered != and ordered >, including every raw NaN payload. -/
structure Instructions (α : Type) where
  literal : ℝ → α
  neg : α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  neu : α → α → Bool
  gt : α → α → Bool
  unused : α

/-- Site-specific lowering of the original and existing commuted source graphs. Rules preserve observed
operand order and contraction; they assert no FP commutativity or split/FMA
identity. Comparison ports evaluate this same lowered intermediate graph. -/
def interpret (native : Instructions α) : Term α → α
  | .input x => x
  | .literal (some .fp32) .real x => native.literal x
  | .binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul value
        (.binary (some .fp32) .real .sub (.literal (some .fp32) .real one)
          (.binary (some .fp32) .real .mul lr wd))) step =>
      native.fma (native.fma (native.neg (interpret native lr)) (interpret native wd) (native.literal one))
        (interpret native value) (interpret native step)
  | .binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul difference beta) gradient =>
      native.fma (interpret native beta) (interpret native difference) (interpret native gradient)
  | .binary (some .fp32) .real .add gradient
      (.binary (some .fp32) .real .mul difference beta) =>
      native.fma (interpret native beta) (interpret native difference) (interpret native gradient)
  | .binary (some .fp32) .real .sub (.literal (some .fp32) .real one)
      (.binary (some .fp32) .real .mul lr wd) =>
      native.fma (native.neg (interpret native lr)) (interpret native wd) (native.literal one)
  | .binary (some .fp32) .real .sub a b => native.sub (interpret native a) (interpret native b)
  | .binary (some .fp32) .real .mul a b => native.mul (interpret native a) (interpret native b)
  | .unary (some .fp32) (.neg .real) value => native.sub (native.literal 0) (interpret native value)
  | _ => native.unused

def semantics (native : Instructions α) : Tile.Semantics (Term α) where
  algebra := graphAlgebra
  less := fun a b => native.gt (interpret native b) (interpret native a)
  equal := fun a b => !(native.neu (interpret native a) (interpret native b))

def arguments (params : Scalars α) : Scalars (Term α) :=
  ⟨.input params.lr,.input params.wd,.input params.beta1,.input params.beta2⟩

/-- Independent observed native recurrence, with all instruction order fixed.
The scalar parameters are actual ABI FP32 words, not Real-encoded literals. -/
def reference (native : Instructions α) (params : Scalars α) (output : Output) (p g m : α) : α :=
  let decay := native.fma (native.neg params.lr) params.wd (native.literal 1)
  let difference := native.sub m g
  let update := native.fma params.beta1 difference g
  match output with
  | .parameter =>
    let chosen := if native.gt update (native.literal 0) then native.sub (native.literal 0) params.lr else params.lr
    let gate := if native.neu update (native.literal 0) then native.literal 1 else native.literal 0
    native.fma decay p (native.mul chosen gate)
  | .momentum => native.fma params.beta2 difference g

def sourceScalar (native : Instructions α) (params : Scalars α) (output : Output) (p g m : α) : α :=
  interpret native (Tile.scalar (semantics native) (arguments params) output (.input p) (.input g) (.input m))

theorem lowering (native : Instructions α) (params : Scalars α) (output : Output) (p g m : α) :
    sourceScalar native params output p g m=reference native params output p g m := by
  cases output
  · simp only [sourceScalar,Tile.scalar,semantics,arguments,graphAlgebra,interpret,reference]
    split <;> split <;> simp_all [interpret]
  · rfl

/-- Lowering of the complete existing candidate's scalar output graph. -/
theorem commuted_lowering (native : Instructions α) (params : Scalars α)
    (output : Output) (p g m : α) :
    interpret native (CommutedTile.scalar (semantics native) (arguments params)
      output (.input p) (.input g) (.input m)) =
      reference native params output p g m := by
  cases output
  · simp only [CommutedTile.scalar,semantics,arguments,graphAlgebra,interpret,reference]
    split <;> split <;> simp_all [interpret]
  · rfl

def realInstructions : Instructions ℝ where
  literal := id
  neg := fun x => -x
  sub := fun a b => a-b
  mul := fun a b => a*b
  fma := fun a b c => a*b+c
  neu := fun a b => decide (a≠b)
  gt := fun a b => decide (a>b)
  unused := 0

theorem real_projection (params : Scalars ℝ) (output : Output) (p g m : ℝ) :
    reference realInstructions params output p g m=Math.target params output p g m := by
  rw [←Math.source_scalar_correct params output p g m]
  cases output <;> simp only [reference,realInstructions,Math.sourceScalar,id_eq,decide_eq_true_eq]
  · have direction : params.beta1*(m-g)+g=(m-g)*params.beta1+g := by ring
    rw [direction]
    split_ifs <;> ring
  · ring

#specNonCircular reference avoiding [interpret,sourceScalar,Tile.scalar,Tile.sourceRun]
#axiomsClean lowering
#axiomsClean commuted_lowering
#axiomsClean real_projection
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Native
