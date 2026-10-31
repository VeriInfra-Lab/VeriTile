import bench.tritonbench_g.matmul_triton1.Sources.Numerics
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.DotProfile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
noncomputable section
inductive Value (α : Type) where
  | input : α → Value α
  | zero : Value α
  | dot {k : Nat} (precision : Option ComputeDType) (options : DotOptions)
      (left right : Fin k → Value α) (carry : Value α) : Value α
instance [Inhabited α] : Inhabited (Value α) := ⟨.input default⟩

def observe (P : Core.Primitives α) : Value α → α
  | .input value => value
  | .zero => P.zero
  | .dot _ _ left right carry => P.dot (fun i => observe P (left i))
      (fun i => observe P (right i)) (observe P carry)

/-- This is a selected compiler execution profile, not an arithmetic identity
between split dot/add rounding and a fused dot. It recognizes only the FP32
carry plus the dot's explicit zero seed. No matrix, memory or whole-kernel
result is supplied by a primitive. Actual operands remain in the source tree. -/
def add (rest : Algebra α) (P : Core.Primitives α) (selected : DotOptions) (p : Option ComputeDType)
    (d : FloatDType) (op : Binary) (left right : Value α) : Value α :=
  match op,right with
  | .add,.dot q options xs ys .zero =>
    if (p=none ∨ p=some .fp32) ∧ (d=.real ∨ d=.fp32) ∧ p=q ∧ options=selected then
      .dot q options xs ys left
    else .input (rest.binary p d op (observe P left) (observe P right))
  | _,_ => .input (rest.binary p d op (observe P left) (observe P right))

def tileDot {batch : TileShape} {m k n : Nat}
    (precision : Option ComputeDType) (options : DotOptions)
    (left : TileIndex (batch++[m,k]) → Value α)
    (right : TileIndex (batch++[k,n]) → Value α)
    (carry : TileIndex (batch++[m,n]) → Value α) :
    TileIndex (batch++[m,n]) → Value α :=
  match batch with
  | [] => fun i => .dot precision options (fun t => left (i.1,t,PUnit.unit))
      (fun t => right (t,i.2.1,PUnit.unit)) (carry i)
  | _::tail => fun i => tileDot (batch:=tail) precision options
      (fun j => left (i.1,j)) (fun j => right (i.1,j)) (fun j => carry (i.1,j)) i.2
termination_by batch.length

/-- Each unrecognized scalar operation retains the supplied primitive model. Only the selected source dot options are supported.
Unsupported source effects remain unavailable, rather than supplying a row. -/
def algebra (rest : Algebra α) (P : Core.Primitives α)
    (selected : DotOptions := {allowTF32 := some Bool.false}) : Algebra (Value α) where
  literal := fun p d value => if (d=.real ∨ d=.fp32) ∧ value=0 then .zero
    else .input (rest.literal p d value)
  negInf := .input rest.negInf
  binary := add rest P selected
  unary := fun p op value => .input (rest.unary p op (observe P value))
  cast := fun p src dst value => if ((src=.real ∨ src=.fp32) ∧ dst=.fp32) ∨
      (p=some .fp32 ∧ src=.real ∧ dst=.real) then value
    else .input (rest.cast p src dst (observe P value))
  fromNat := fun p n => .input (rest.fromNat p n)
  fromInt := fun p n => .input (rest.fromInt p n)
  fp32Bits := fun bits => .input (rest.fp32Bits bits)
  fp32Load := id
  reduceMax := fun p {shape} ax keep values => fun i => .input (rest.reduceMax p ax keep (fun j => observe P (values j)) i)
  reduceSum := fun p {shape} ax keep values => fun i => .input (rest.reduceSum p ax keep (fun j => observe P (values j)) i)
  dot := fun p {batch} {m} {k} {n} xs ys => if selected = {} then some (tileDot p {} xs ys (fun _ => .zero)) else none
  dotWithOptions := fun p options {batch} {m} {k} {n} xs ys => if options = selected then some (tileDot p options xs ys (fun _ => .zero)) else none
  dotAcc := fun p l r out options {batch} {m} {k} {n} xs ys carry =>
    match l,r,out with
    | .real,.real,.real => if options=selected then some (tileDot p options xs ys carry) else none
    | _,_,_ => none

theorem add_dot (rest : Algebra α) (P : Core.Primitives α) {k : Nat}
    (xs ys : Fin k → Value α) (carry : Value α) (options : DotOptions) :
    (algebra rest P options).binary (some .fp32) .real .add carry
      (.dot (some .fp32) options xs ys .zero) =
      .dot (some .fp32) options xs ys carry := by simp [algebra,add]

theorem observed_dot (P : Core.Primitives α) {k : Nat}
    (xs ys : Fin k → α) (carry : α) (options : DotOptions) :
    observe P (.dot (some .fp32) options (fun i => .input (xs i))
      (fun i => .input (ys i)) (.input carry)) = P.dot xs ys carry := rfl
theorem fp32_carrier_cast (rest : Algebra α) (P : Core.Primitives α) (value : Value α) :
    (algebra rest P).cast (some .fp32) .real .real value = value := by simp [algebra]
#axiomsClean fp32_carrier_cast
#axiomsClean add_dot
#axiomsClean observed_dot
end
end VeriTile.Bench.Repairs.MatmulTriton1.DotProfile
