import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeBody

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

/-- Local native primitives. `dotAcc` consumes its carry as an operand, with
no equation to a separately rounded dot and addition. Configuration, precision,
dtypes, options, complete block operands and lane are all explicit. -/
structure Primitives (α : Type) where
  leaf : FloatDType → BitVec 32 → α
  literal : Option ComputeDType → FloatDType → ℝ → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  dot : Host.Config → Option ComputeDType → FloatDType → FloatDType → FloatDType → DotOptions →
    (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) →
    TileIndex (batch++[m,n]) → α
  dotAcc : Host.Config → Option ComputeDType → FloatDType → FloatDType → FloatDType → DotOptions →
    (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) → α →
    TileIndex (batch++[m,n]) → α

/-- Interpretation of the staged graph. A pending dot only denotes a standalone
dot if explicitly observed; the source body consumes it before storing acc. -/
def evaluate {α : Type} (p : Primitives α) : Expr → α
  | .leaf d w => p.leaf d w
  | .literal q d v => p.literal q d v
  | .cast q l r x => p.cast q l r (evaluate p x)
  | .binary q d op x y => p.binary q d op (evaluate p x) (evaluate p y)
  | .pendingDot c q l r out opts batch m k n a b i =>
      p.dot c q l r out opts batch m k n (fun j => evaluate p (a j)) (fun j => evaluate p (b j)) i
  | .dotAcc c q l r out opts batch m k n a b acc i =>
      p.dotAcc c q l r out opts batch m k n (fun j => evaluate p (a j))
        (fun j => evaluate p (b j)) (evaluate p acc) i

theorem evaluate_dotAcc {α : Type} (p : Primitives α) (c : Host.Config)
    (q : Option ComputeDType) (l r out : FloatDType) (opts : DotOptions)
    (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch++[m,n])) :
    evaluate p (.dotAcc c q l r out opts batch m k n a b acc i)=
      p.dotAcc c q l r out opts batch m k n (fun j => evaluate p (a j))
        (fun j => evaluate p (b j)) (evaluate p acc) i := rfl

/-- Native output observation follows the exact final typed memory graph. -/
def returnedValue {α : Type} (p : Primitives α) (h : Host.Call) (s : State Expr)
    (b ch g r n : Nat) : α := evaluate p (FPGrid.readback h s b ch g r n)

#axiomsClean evaluate_dotAcc
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph

#guard_msgs (drop info) in
#auditModuleAxioms
