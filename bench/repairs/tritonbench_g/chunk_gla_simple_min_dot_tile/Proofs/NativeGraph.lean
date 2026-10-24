import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeGraph
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
noncomputable section

/-- Native staging payloads, not already-rounded dot results. The input words
remain leaves and every gate/cast operation remains an explicit node. -/
inductive Expr where
  | leaf (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | load (value : Expr)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Expr)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (left right : Expr)
  | pendingDot (precision : Option ComputeDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (i : TileIndex (batch++[m,n]))
  | dotAcc (precision : Option ComputeDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (carry : Expr) (i : TileIndex (batch++[m,n]))
instance : Inhabited Expr := ⟨.leaf 0⟩

def options : DotOptions := {allowTF32 := some Bool.false}

/-- This is graph construction before numerical interpretation. It is not an
identity between a separately rounded dot/add and a fused native operation. -/
def consume (precision : Option ComputeDType) (dtype : FloatDType) : Binary → Expr → Expr → Expr
  | .add, carry, .pendingDot p opts batch m k n a b i => .dotAcc p opts batch m k n a b carry i
  | op, a, b => .binary precision dtype op a b

def algebra (rest : Algebra Expr) : Algebra Expr :=
  {rest with
    literal := Expr.literal
    fp32Load := Expr.load
    cast := Expr.cast
    unary := Expr.unary
    binary := consume
    dotWithOptions := fun p opts {batch} {m k n} a b => some (fun i => .pendingDot p opts batch m k n a b i)}

def dots (rest : Algebra Expr) (c : Shape) : Typed.Dots (algebra rest) c where
  qh := fun a b i => .pendingDot (some .fp32) options [] c.BT c.BK c.BV a b i
  qk := fun a b i => .pendingDot (some .fp32) options [] c.BT c.BK c.BT a b i
  sv := fun a b i => .pendingDot (some .fp32) options [] c.BT c.BT c.BV a b i
  qh_eq := fun _ _ => rfl
  qk_eq := fun _ _ => rfl
  sv_eq := fun _ _ => rfl

/-- Independent initial-memory recurrence forms the actual QH and QK dotAcc
nodes with their live carries at every K block. -/
theorem update_dotAcc (rest : Algebra Expr) (c : Shape) (s : State Expr)
    (block : Nat) (acc : FPValue.Acc Expr c) :
    FPValue.update (algebra rest) c (dots rest c) s block acc =
      (fun i => Expr.dotAcc (some .fp32) options [] c.BT c.BK c.BV
        (FPValue.qValue (algebra rest) c s block) (FPValue.hValue (algebra rest) c s block) (acc.1 i) i,
       fun i => Expr.dotAcc (some .fp32) options [] c.BT c.BK c.BT
        (FPValue.qValue (algebra rest) c s block) (FPValue.kValue (algebra rest) c s block) (acc.2 i) i) := rfl

theorem output_dotAcc (rest : Algebra Expr) (c : Shape) (s : State Expr) (scale : ℝ)
    (i : FPValue.OIndex c) :
    FPValue.outputValue (algebra rest) c (dots rest c) s scale i =
      Expr.cast (some .fp32) .real .fp32
        (.binary (some .fp32) .real .mul
          (.dotAcc (some .fp32) options [] c.BT c.BT c.BV
            (fun j => .cast (some .fp32) .real .real
              (FPValue.gatedS (algebra rest) c s (FPValue.accumulated (algebra rest) c (dots rest c) s) j))
            (FPValue.vValue (algebra rest) c s)
            (FPValue.gatedO (algebra rest) c s (FPValue.accumulated (algebra rest) c (dots rest c) s) i) i)
          (.literal (some .fp32) .real scale)) := rfl

/-- The actual reached loop body constructs the two native accumulator nodes.
The equality is at graph construction, before either dot is interpreted. -/
theorem reached_body (rest : Algebra Expr) (c : Shape) (scale : ℝ)
    (initial current : State Expr) (block : Nat) (acc : FPValue.Acc Expr c)
    (ctx : FPValue.Context c initial current) (ha : FPValue.HasAcc c acc current) :
    let t := (FPValue.bodyExecution (algebra rest) c scale (dots rest c) initial current block acc ctx ha).val
    FPValue.HasAcc c
      (fun i => Expr.dotAcc (some .fp32) options [] c.BT c.BK c.BV
        (FPValue.qValue (algebra rest) c initial block) (FPValue.hValue (algebra rest) c initial block) (acc.1 i) i,
       fun i => Expr.dotAcc (some .fp32) options [] c.BT c.BK c.BT
        (FPValue.qValue (algebra rest) c initial block) (FPValue.kValue (algebra rest) c initial block) (acc.2 i) i) t :=
  (FPValue.bodyContext (algebra rest) c scale (dots rest c) initial current block acc ctx ha).2

/-- Exact local numerical ports for the staged FP32/IEEE dot graph. No
whole-output, execution, address, or split-rounded-equals-dotAcc law appears. -/
structure Primitives (α : Type) where
  leaf : BitVec 32 → α
  literal : Option ComputeDType → FloatDType → ℝ → α
  load : α → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  unary : Option ComputeDType → Unary → α → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  dot : Option ComputeDType → DotOptions → (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) → TileIndex (batch++[m,n]) → α
  dotAcc : Option ComputeDType → DotOptions → (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) → α → TileIndex (batch++[m,n]) → α

def evaluate {α : Type} (p : Primitives α) : Expr → α
  | .leaf w => p.leaf w
  | .literal q d v => p.literal q d v
  | .load x => p.load (evaluate p x)
  | .cast q a b x => p.cast q a b (evaluate p x)
  | .unary q op x => p.unary q op (evaluate p x)
  | .binary q d op x y => p.binary q d op (evaluate p x) (evaluate p y)
  | .pendingDot q opts batch m k n a b i => p.dot q opts batch m k n
      (fun j => evaluate p (a j)) (fun j => evaluate p (b j)) i
  | .dotAcc q opts batch m k n a b acc i => p.dotAcc q opts batch m k n
      (fun j => evaluate p (a j)) (fun j => evaluate p (b j)) (evaluate p acc) i

/-- Native graph execution reaches one merged grid state, and numerical
readback is the independent initial-input recurrence evaluated by local ports. -/
theorem whole_grid {α : Type} (p : Primitives α) (rest : Algebra Expr) (c : Shape)
    (scale : ℝ) (s : State Expr) :
    ∃ t, FPGrid.run (algebra rest) c scale (FPGrid.programs c) s=some t ∧
      FPGrid.Result (algebra rest) c scale (dots rest c) s t ∧
      (∀ q : FPGrid.Pid c, ∀ i, FPValue.active c (FPGrid.launch c s q) i →
        evaluate p ((t.mem "o" (FPValue.oOffset c (FPGrid.launch c s q) i)).read .fp32)=
          evaluate p (FPValue.outputValue (algebra rest) c (dots rest c) (FPGrid.launch c s q) scale i)) := by
  obtain ⟨t,he,hr⟩ := FPGrid.whole_grid (algebra rest) c scale (dots rest c) s
  refine ⟨t,he,hr,?_⟩
  intro q i hi
  rw [hr.1 q i hi]
  rfl

#axiomsClean update_dotAcc
#axiomsClean reached_body
#axiomsClean whole_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.NativeGraph
