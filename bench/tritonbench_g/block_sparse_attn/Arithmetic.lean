import bench.tritonbench_g.block_sparse_attn.Primitives

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- Four physical copies of a row scalar. Reductions and MMA select the
actual owning copy; this is finite layout data, not a reduction/result port. -/
abbrev Replicas (α : Type) := Fin 4 → α

/-- Only scalar primitives and a fixed eight-pair instruction. -/
structure Arithmetic (α : Type) extends Ops α where
  literal : ℝ → α
  argument : ℝ → α

def broadcast (x : α) : Replicas α := fun _ => x

def bindScale {α : Type} (scale : α) (s : State α) : State α :=
  s.setReg "softmax_scale" .real [] (fun _ => scale)
def binary {α : Type} (a : Arithmetic α) : Binary → Replicas α → Replicas α → Replicas α
  | .add,x,y => fun o => a.add (x o) (y o)
  | .sub,x,y => fun o => a.sub (x o) (y o)
  | .mul,x,y => fun o => a.mul (x o) (y o)
  | .div,x,y => fun o => a.div (x o) (y o)
  | .max,x,y => fun o => a.max (x o) (y o)
  | .pow,_,_ => broadcast a.zero

def fragment {α : Type} (a : Arithmetic α) {k : Nat}
    (x : Fin k → Replicas α) (chunk : Nat) (lane : Fin 8) : α :=
  if h : chunk*8+lane.val<k then
    x ⟨chunk*8+lane.val,h⟩ (owner (chunk*8+lane.val)) else a.zero

def chain {α : Type} (a : Arithmetic α) {k : Nat}
    (x y : Fin k → Replicas α) (seed : α) : α :=
  (List.range (k/8)).foldl (fun old chunk =>
    a.mma8 .mma16n8k8 (fragment a x chunk) (fragment a y chunk) old) seed

def tensor {α : Type} (a : Arithmetic α) {batch : TileShape} {m k n : Nat}
    (x : TileIndex (batch++[m,k]) → Replicas α)
    (y : TileIndex (batch++[k,n]) → Replicas α)
    (seed : TileIndex (batch++[m,n]) → Replicas α) :
    TileIndex (batch++[m,n]) → Replicas α :=
  match batch with
  | [] => fun i => broadcast (chain a (fun t => x (i.1,t,PUnit.unit))
      (fun t => y (t,i.2.1,PUnit.unit)) (seed i (owner i.2.1.val)))
  | _::rest => fun i => tensor a (batch := rest)
      (fun j => x (i.1,j)) (fun j => y (i.1,j)) (fun j => seed (i.1,j)) i.2
termination_by batch.length

def reduceLine {α : Type} (a : Arithmetic α) (op : α → α → α)
    {n : Nat} (x : Fin n → Replicas α) : Replicas α :=
  if h : n=16 then fun o => (SM90B16.tree o).eval op
    (fun k => x ⟨k.val,by omega⟩ (owner k.val)) else broadcast a.zero

def reduction {α : Type} (a : Arithmetic α) (op : α → α → α)
    {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → (TileIndex shape → Replicas α) →
      TileIndex (TileShape.reduceShape shape ax keepDims) → Replicas α
  | Bool.false,x => fun out => reduceLine a op
      (fun k => x (TileShape.insertAxisIndex shape ax out k))
  | Bool.true,x => fun out => reduceLine a op
      (fun k => x (TileShape.replaceAxisIndex shape ax out k))

def Arithmetic.model {α : Type} (a : Arithmetic α) : Algebra (Replicas α) where
  literal := fun _ _ x => broadcast (a.literal x)
  negInf := broadcast a.negInf
  binary := fun _ _ => binary a
  fma := fun _ => some (fun x y z o => a.fma (x o) (y o) (z o))
  unary := fun _ op x => match op with
    | .exp => fun o => a.exp (x o)
    | _ => broadcast a.zero
  cast := fun _ _ _ => id
  fromNat := fun _ n => broadcast (a.literal n)
  fromInt := fun _ n => broadcast (a.literal n)
  fp32Bits := fun _ => broadcast a.zero
  fp32Load := id
  reduceMax := fun _ => reduction a a.max
  reduceSum := fun _ => reduction a a.add
  dot := fun _ {_} {_} {_} {_} x y =>
    some (tensor a x y (fun _ => broadcast a.zero))
  dotTyped := fun _ left right out _ {_} {_} {_} {_} x y =>
    some (fun i => ofFloat out (tensor a (fun j => toFloat left (x j))
      (fun j => toFloat right (y j)) (fun _ => broadcast a.zero) i))
  dotAcc := fun _ left right out _ {_} {_} {_} {_} x y seed =>
    some (fun i => ofFloat out (tensor a (fun j => toFloat left (x j))
      (fun j => toFloat right (y j)) (fun j => toFloat out (seed j)) i))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
