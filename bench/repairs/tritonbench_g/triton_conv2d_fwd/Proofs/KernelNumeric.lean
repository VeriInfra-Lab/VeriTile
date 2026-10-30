import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelLaunch
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelOperands
namespace VeriTile.Bench.Repairs.Conv2d.KernelNumeric
open VeriTile.Triton FP.Structural KernelBinding KernelGrid
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16384
set_option linter.unusedSimpArgs false

/-- Internal proof interface. Public entries instantiate this with either the
finite Real sum or the literal native instruction expansion. It is never a
public configurable-dot or result premise. -/
structure Operations (α : Type) where
  zero : α
  half : α → α
  lower32 : (Fin 32 → α) → (Fin 32 → α) → α → α

/-- A source dot remains a node until its immediately following accumulator
addition. This exposes compiler carry absorption as a lowering interpretation;
it does not assert an identity between rounded split and fused arithmetic. -/
inductive Value (α : Type) where
  | scalar (value : α)
  | dot32 (left right : Fin 32 → α)

instance {α} [Inhabited α] : Inhabited (Value α) := ⟨.scalar default⟩

def scalar (o : Operations α) : Value α → α
  | .scalar x => x
  | .dot32 a b => o.lower32 a b o.zero

def add (o : Operations α) (a b : Value α) : Value α :=
  match b with
  | .dot32 left right => .scalar (o.lower32 left right (scalar o a))
  | .scalar _ => .scalar o.zero

/-- Only the source's zero literal, operand conversion, representation casts,
and dot/add pair are reachable. Other operations have inert fixed meanings. -/
def algebra (o : Operations α) : Algebra (Value α) where
  literal := fun _ _ _ => .scalar o.zero
  negInf := .scalar o.zero
  binary := fun _ _ op a b => if op=.add then add o a b else .scalar o.zero
  unary := fun _ _ x => x
  cast := fun _ src dst x => if src=.fp32 ∧ dst=.fp16 then .scalar (o.half (scalar o x)) else x
  fromNat := fun _ _ => .scalar o.zero
  fromInt := fun _ _ => .scalar o.zero
  fp32Bits := fun _ => .scalar o.zero
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => .scalar o.zero
  reduceSum := fun _ _ _ _ _ _ => .scalar o.zero

def dot (o : Operations α) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Value α) (b : TileIndex (batch++[k,n]) → Value α) :
    TileIndex (batch++[m,n]) → Value α :=
  match batch with
  | [] => fun i => if hk : k=32 then .dot32
      (fun e => scalar o (a (i.1,⟨e.val,by omega⟩,PUnit.unit)))
      (fun e => scalar o (b (⟨e.val,by omega⟩,i.2.1,PUnit.unit)))
      else .scalar o.zero
  | _::tail => fun i => dot o (batch:=tail) (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2
termination_by batch.length

def halfDot (o : Operations α) : KernelHalf.Dot (Value α) := ⟨fun _ _ {batch} {m} {k} {n} a b=>dot o (batch:=batch) (m:=m) (k:=k) (n:=n) a b⟩
def defaultDot (o : Operations α) : KernelDefault.Dot (Value α) := ⟨fun _ _ {batch} {m} {k} {n} a b=>dot o (batch:=batch) (m:=m) (k:=k) (n:=n) a b⟩

variable {α : Type} [Inhabited α]

def input (o : Operations α) (half : Bool) (p : KernelHalf.Parameters)
    (t : State (Value α)) (h w c : Nat) (i : KernelHalf.AIndex p) : α :=
  let x:=scalar o (KernelOperands.input (.scalar o.zero) p t h w c i)
  if half then o.half x else x

def weight (o : Operations α) (half : Bool) (p : KernelHalf.Parameters)
    (t : State (Value α)) (h w c : Nat) (i : KernelHalf.BIndex p) : α :=
  let x:=scalar o (KernelOperands.weight (.scalar o.zero) p t h w c i)
  if half then o.half x else x

/-- Scalar instruction recurrence; no source statements, registers, stores or
execution result occur. Every padded channel operation is retained. -/
def laneReference (o : Operations α) (s : Shape) (f : Flags) (pid : Program)
    (t : State (Value α)) (i : Index) : α :=
  foldRange (fun h acc=>foldRange (fun w acc=>foldRange
    (fun c acc=>o.lower32
      (fun e=>input o f.fp16 (parameters s f) (atProgram pid t) h w c (i.1,e,PUnit.unit))
      (fun e=>weight o f.fp16 (parameters s f) (atProgram pid t) h w c (e,i.2.1,PUnit.unit)) acc)
    0 s.cg 32 acc) 0 s.kw 1 acc) 0 s.kh 1 o.zero

end
end VeriTile.Bench.Repairs.Conv2d.KernelNumeric
