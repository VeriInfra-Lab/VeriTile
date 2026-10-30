import bench.tritonbench_g.rope_transform.Proofs.KernelPhysical

namespace VeriTile.Bench.Repairs.RopeTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Tagged := VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics.Tagged
abbrev leaf := VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics.leaf
/-- Construct against the current Algebra field layout. The cached historical
Operations.backend predates the core FMA field and is not reused here. -/
def backend {α : Type} (zero : α) (cast : α→α) (add sub mul : α→α→α) : Algebra α where
  literal:=fun _ _ _=>zero
  negInf:=zero
  binary:=fun _ _ op=>match op with
    | .add=>add | .sub=>sub | .mul=>mul | _=>fun a _=>a
  unary:=fun _ _=>id
  cast:=fun _ _ _=>cast
  fromNat:=fun _ _=>zero
  fromInt:=fun _ _=>zero
  fp32Bits:=fun _=>zero
  fp32Load:=id
  reduceMax:=fun _ {_} _ _ _ _=>zero
  reduceSum:=fun _ {_} _ _ _ _=>zero

abbrev realAlgebra := backend (0 : ℝ) id (·+·) (·-·) (·*·)

/-- Independent forward polynomial with the supplied coefficients. -/
def realPair (a b c s : ℝ) : ℝ×ℝ := (a*c-b*s,b*c+a*s)

/-- Selected source-specific lowering. Both products put the coefficient
first; the low output contracts the left product and the high output contracts
its right product. Primitive FP32 conformance is the explicit cfg boundary. -/
def nativeAlgebra (cfg : FP.Config) : Algebra Tagged :=
  backend (leaf (FP.Value.zero .fp32)) (fun x=>leaf x.word)
    (fun x y=>leaf (match y.product with
      | some (a,b)=>FP.fma cfg a b x.word
      | none=>FP.add cfg x.word y.word))
    (fun x y=>leaf (match x.product with
      | some (a,b)=>FP.fma cfg a b y.word.neg
      | none=>FP.sub cfg x.word y.word))
    (fun x y=>⟨FP.mul cfg y.word x.word,some (y.word,x.word)⟩)

def nativePair (cfg : FP.Config) (a b c s : Word) (second : Bool) : Word :=
  if second then FP.fma cfg s a (FP.mul cfg c b)
  else FP.fma cfg c a (FP.mul cfg s b).neg

theorem scalar_real (a b c s : ℝ) (second : Bool) :
    scalar realAlgebra a b c s second=if second then (realPair a b c s).2 else (realPair a b c s).1 := by
  cases second <;> rfl

theorem scalar_native (cfg : FP.Config) (a b c s : Word) (second : Bool) :
    (scalar (nativeAlgebra cfg) (leaf a) (leaf b) (leaf c) (leaf s) second).word=
      nativePair cfg a b c s second := by
  cases second <;> rfl

def realValue (c : Call) (before : Nat→ℝ) (key : Bool) (row head col : Nat) : ℝ :=
  let j:=if col<c.geometry.half then col else col-c.geometry.half
  let pair:=realPair
    (logicalRead before (c.q key) (c.geometry.offset key row head j))
    (logicalRead before (c.q key) (c.geometry.offset key row head (j+c.geometry.half)))
    (logicalRead before (c.tables Bool.false) (row%c.geometry.sequence*c.geometry.half+j))
    (logicalRead before (c.tables Bool.true) (row%c.geometry.sequence*c.geometry.half+j))
  if col<c.geometry.half then pair.1 else pair.2

def nativeValue (cfg : FP.Config) (c : Call) (before : Nat→Word)
    (key : Bool) (row head col : Nat) : Word :=
  let j:=if col<c.geometry.half then col else col-c.geometry.half
  nativePair cfg
    (logicalRead before (c.q key) (c.geometry.offset key row head j))
    (logicalRead before (c.q key) (c.geometry.offset key row head (j+c.geometry.half)))
    (logicalRead before (c.tables Bool.false) (row%c.geometry.sequence*c.geometry.half+j))
    (logicalRead before (c.tables Bool.true) (row%c.geometry.sequence*c.geometry.half+j))
    (decide (c.geometry.half≤col))

theorem real_binding (c : Call) (before : Nat→ℝ) (key : Bool) (row head col : Nat) :
    tensorScalar realAlgebra id c before key row head col=realValue c before key row head col := by
  unfold tensorScalar realValue
  rw [scalar_real]
  by_cases low : col<c.geometry.half <;> simp [low,show (c.geometry.half≤col)↔¬col<c.geometry.half from by omega]

theorem native_binding (cfg : FP.Config) (c : Call) (before : Nat→Word) (key : Bool) (row head col : Nat) :
    (tensorScalar (nativeAlgebra cfg) leaf c before key row head col).word=
      nativeValue cfg c before key row head col := scalar_native cfg _ _ _ _ _

#axiomsClean scalar_real
#axiomsClean scalar_native
#axiomsClean real_binding
#axiomsClean native_binding
#specNonCircular realValue avoiding [Call.source,sourceRun,Grid.run,physical]
#specNonCircular nativeValue avoiding [Call.source,sourceRun,Grid.run,physical]
end
end VeriTile.Bench.Repairs.RopeTransform.KernelContract
