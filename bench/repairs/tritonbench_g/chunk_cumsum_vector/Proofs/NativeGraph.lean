import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPReference

/-! Native FP32/IEEE graph staging. The compiler's accumulating dot receives
the incoming carry. No equality with separately rounded dot-plus-add is used. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.NativeGraph
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 4800000
set_option maxRecDepth 24000

inductive Expr where
  | leaf (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (x : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (x y : Expr)
  | reduceSum (config : Config) (precision : Option ComputeDType) (shape : TileShape)
      (ax : Fin shape.length) (keepDims : Bool) (x : TileIndex shape → Expr)
      (i : TileIndex (TileShape.reduceShape shape ax keepDims))
  | pendingDot (config : Config) (precision : Option ComputeDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat) (a : TileIndex (batch++[m,k]) → Expr)
      (b : TileIndex (batch++[k,n]) → Expr) (i : TileIndex (batch++[m,n]))
  | dotAcc (config : Config) (precision : Option ComputeDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat) (a : TileIndex (batch++[m,k]) → Expr)
      (b : TileIndex (batch++[k,n]) → Expr) (acc : Expr) (i : TileIndex (batch++[m,n]))
instance : Inhabited Expr := ⟨.leaf 0⟩

def consume (precision : Option ComputeDType) (dtype : FloatDType) : Binary → Expr → Expr → Expr
  | .add,acc,.pendingDot c p opts batch m k n a b i => .dotAcc c p opts batch m k n a b acc i
  | op,a,b => .binary precision dtype op a b

def algebra (rest : Algebra Expr) (c : Config) : Algebra Expr :=
  {rest with
    literal := Expr.literal
    cast := Expr.cast
    binary := consume
    reduceSum := fun p {shape} ax keepDims x i => .reduceSum c p shape ax keepDims x i}

def backend (c : Config) : DotBackend Expr :=
  fun p opts {batch} {m k n} a b i => .pendingDot c p opts batch m k n a b i

/-- This property excludes staged payloads throughout a stored expression. -/
def Materialized : Expr → Prop
  | .leaf _ => True
  | .literal _ _ _ => True
  | .cast _ _ _ x => Materialized x
  | .binary _ _ _ x y => Materialized x ∧ Materialized y
  | .reduceSum _ _ _ _ _ x _ => ∀ i, Materialized (x i)
  | .pendingDot _ _ _ _ _ _ _ _ _ _ => False
  | .dotAcc _ _ _ _ _ _ _ a b acc _ =>
      (∀ i,Materialized (a i)) ∧ (∀ i,Materialized (b i)) ∧ Materialized acc

theorem output_dotAcc (rest : Algebra Expr) (c : Config)
    (input : Values Expr .real [c.BT,BS]) (acc : Values Expr .real [BS]) (i : TileIndex [c.BT,BS]) :
    FPValues.output (algebra rest c) (backend c) c input acc i=
      Expr.dotAcc c (some .fp32) {allowTF32:=some Bool.false} [] c.BT c.BT BS
        (FPValues.triangular (algebra rest c) c) input (acc i.2) i := rfl

theorem loaded_materialized (rest : Algebra Expr) (h : Call) (c : Config)
    (prog : Geometry.Program h) (logical : Nat → Nat → Nat → Expr)
    (ready : ∀ r t s, Materialized (logical r t s)) (padding : FPStorage.Padding Expr)
    (q : Nat) (i : TileIndex [c.BT,BS]) :
    Materialized (FPReference.loaded (algebra rest c) h c prog logical padding q i) := by
  unfold FPReference.loaded
  split_ifs
  · exact ready _ _ _
  · trivial

theorem triangular_materialized (rest : Algebra Expr) (c : Config) (i : TileIndex [c.BT,c.BT]) :
    Materialized (FPValues.triangular (algebra rest c) c i) := by
  unfold FPValues.triangular
  split_ifs <;> trivial

theorem carry_materialized (rest : Algebra Expr) (h : Call) (c : Config)
    (prog : Geometry.Program h) (logical : Nat → Nat → Nat → Expr)
    (ready : ∀ r t s, Materialized (logical r t s)) (padding : FPStorage.Padding Expr)
    (q : Nat) (j : TileIndex [BS]) :
    Materialized (FPReference.carry (algebra rest c) h c prog logical padding q j) := by
  induction q with
  | zero => trivial
  | succ q ih =>
    change Materialized (FPReference.carry (algebra rest c) h c prog logical padding q j) ∧
      ∀ i, Materialized (FPReference.loaded (algebra rest c) h c prog logical padding q i)
    exact ⟨ih,loaded_materialized rest h c prog logical ready padding q⟩

theorem expected_materialized (rest : Algebra Expr) (h : Call) (outType : h.outputDType=.fp32)
    (c : Config) (positive : 0<c.BT) (prog : Geometry.Program h)
    (logical : Nat → Nat → Nat → Expr) (ready : ∀ r t s, Materialized (logical r t s))
    (padding : FPStorage.Padding Expr) (t : Nat) (j : Fin BS) :
    Materialized ((FPReference.expected (algebra rest c) (backend c) h c positive prog logical padding t j).read .fp32) := by
  unfold FPReference.expected FPMemory.stored
  rw [outType]
  change Materialized (FPValues.output (algebra rest c) (backend c) c
    (FPReference.loaded (algebra rest c) h c prog logical padding (t/c.BT))
    (FPReference.carry (algebra rest c) h c prog logical padding (t/c.BT))
    (⟨t%c.BT,Nat.mod_lt _ positive⟩,j,PUnit.unit))
  rw [output_dotAcc]
  exact ⟨triangular_materialized rest c,loaded_materialized rest h c prog logical ready padding _,
    carry_materialized rest h c prog logical ready padding _ _⟩

/-- Local numerical ports. DotAcc is IEEE FP32 with its actual carry; the
configured reduction retains its source ax, shape and output lane. -/
structure Primitives (α : Type) where
  leaf : BitVec 32 → α
  literal : Option ComputeDType → FloatDType → ℝ → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  reduceSum : Config → Option ComputeDType → (shape : TileShape) → (ax : Fin shape.length) → (keepDims : Bool) →
    (TileIndex shape → α) → TileIndex (TileShape.reduceShape shape ax keepDims) → α
  dotAcc : Config → Option ComputeDType → DotOptions → (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) → α → TileIndex (batch++[m,n]) → α

/-- Pending nodes have no floating interpretation. The proved Materialized
property excludes this default case from every stored output graph. -/
def evaluate {α : Type} [Inhabited α] (p : Primitives α) : Expr → α
  | .leaf w => p.leaf w
  | .literal q d v => p.literal q d v
  | .cast q a b x => p.cast q a b (evaluate p x)
  | .binary q d op x y => p.binary q d op (evaluate p x) (evaluate p y)
  | .reduceSum c q shape ax keep x i => p.reduceSum c q shape ax keep (fun j => evaluate p (x j)) i
  | .pendingDot _ _ _ _ _ _ _ _ _ _ => default
  | .dotAcc c q opts batch m k n a b acc i => p.dotAcc c q opts batch m k n
      (fun j => evaluate p (a j)) (fun j => evaluate p (b j)) (evaluate p acc) i

#print axioms output_dotAcc
#print axioms expected_materialized
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.NativeGraph
