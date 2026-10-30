import bench.tritonbench_g.llama_ff_triton.Proofs.Source
import bench.tritonbench_g.llama_ff_triton.Proofs.Selected

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- Scalar numerical instructions and one fixed K16 MMA result lane.
There is no output, execution, dot, reduction, or matrix-valued field. -/
structure Arithmetic (α : Type) where
  literal : Option ComputeDType → FloatDType → ℝ → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  unary : Option ComputeDType → Unary → α → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  fromInt : Option ComputeDType → Int → α
  powi : Option ComputeDType → α → Int → α
  decode : FloatDType → Float8E5Bits → α
  mma16 : (Fin 16 → α) → (Fin 16 → α) → α → α

def fragment {α : Type} [Inhabited α] {k : Nat} (x : Fin k → α) (j : Nat) (e : Fin 16) : α :=
  if h : j*16+e.val<k then x ⟨j*16+e.val,h⟩ else default
def chain {α : Type} [Inhabited α] {k : Nat} (a : Arithmetic α)
    (x y : Fin k → α) (initial : α) : α :=
  (List.range (k/16)).foldl (fun old j => a.mma16 (fragment x j) (fragment y j) old) initial
def tensor {α : Type} [Inhabited α] (a : Arithmetic α) {batch : TileShape} {m k n : Nat}
    (x : TileIndex (batch++[m,k]) → α) (y : TileIndex (batch++[k,n]) → α)
    (acc : TileIndex (batch++[m,n]) → α) : TileIndex (batch++[m,n]) → α :=
  match batch with
  | [] => fun i => chain a (fun t => x (i.1,t,PUnit.unit))
      (fun t => y (t,i.2.1,PUnit.unit)) (acc i)
  | _::rest => fun i => tensor a (batch := rest)
      (fun j => x (i.1,j)) (fun j => y (i.1,j)) (fun j => acc (i.1,j)) i.2
termination_by batch.length

def reduceLine {α : Type} [Inhabited α] {n : Nat} (a : Arithmetic α)
    (precision : Option ComputeDType) (x : Fin n → α) : α :=
  if h : n=64 then LlamaFF.butterfly64.tree.eval (a.binary precision .real .add)
    (fun k => x ⟨k.val,by omega⟩)
  else default
def reduction {α : Type} [Inhabited α] (a : Arithmetic α)
    (precision : Option ComputeDType) {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → (TileIndex shape → α) →
      TileIndex (TileShape.reduceShape shape ax keepDims) → α
  | Bool.false, x => fun out => reduceLine a precision
      (fun k => x (TileShape.insertAxisIndex shape ax out k))
  | Bool.true, x => fun out => reduceLine a precision
      (fun k => x (TileShape.replaceAxisIndex shape ax out k))

def Arithmetic.model {α : Type} [Inhabited α] (a : Arithmetic α) : Algebra α where
  literal := a.literal
  negInf := default
  binary := a.binary
  unary := a.unary
  cast := a.cast
  fromNat := fun p n => a.fromInt p n
  fromInt := a.fromInt
  fp32Bits := fun _ => default
  float8E5Bits := some a.decode
  libdevicePowi := fun p => some (a.powi p)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => default
  reduceSum := reduction a
  dotTyped := fun p left right out _ {_} {_} {_} {_} x y =>
    some (fun i => ofFloat out (tensor a (fun j => toFloat left (x j))
      (fun j => toFloat right (y j)) (fun _ => a.literal p .real 0) i))
  dotAcc := fun _ left right out _ {_} {_} {_} {_} x y acc =>
    some (fun i => ofFloat out (tensor a (fun j => toFloat left (x j))
      (fun j => toFloat right (y j)) (fun j => toFloat out (acc j)) i))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
