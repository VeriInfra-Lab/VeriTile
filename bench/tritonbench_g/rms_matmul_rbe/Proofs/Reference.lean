import bench.tritonbench_g.rms_matmul_rbe.Proofs.Core
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Memory

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open RmsMatmulRbe (Parameters RowInput DotPlan Reduction steps masked dotBlock)
noncomputable section
set_option maxHeartbeats 4000000

/-- Source primitives. Subtraction and integer-exponent libdevice power are
independent operations; neither is silently replaced by add/neg or x*x.
The inherited MMA port is exactly one fixed eight-pair instruction lane. -/
structure Ops (α : Type) extends RmsMatmulRbe.Ops α where
  sub : α → α → α
  powi : α → Int → α
  /-- Compile-time integer literal materialization (HEAD_DIM). Runtime K,
  pair coordinates and signed positions use the distinct `ofInt` port. -/
  literalNat : Nat → α

/-- The two-argument source dot starts from zero on every K64 iteration.
The old carry is an operand of a distinct outer addition. -/
def accumulation (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → DotPlan) (n : Nat) : Nat → α
  | 0 => a.zero
  | t+1 => a.add (accumulation a p x plans n t)
      (dotBlock a.toOps p x (plans t) n t a.zero)

def moment (a : Ops α) (p : Parameters α) (x : RowInput α) (l : Fin 64) : Nat → α
  | 0 => a.zero
  | t+1 => a.add (moment a p x l t) (a.powi (masked a.toOps p x.x t l) 2)

def norm (a : Ops α) (p : Parameters α) (x : RowInput α) (r : Reduction) : α :=
  a.rsqrt (a.add (a.divide
    (r.tree.eval a.add (fun l => moment a p x l (steps p.K)))
    (a.ofInt p.K)) p.epsilon)

def projection (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (n : Nat) : α :=
  a.mul (accumulation a p x (plans n) n (steps p.K)) (norm a p x r)

def angle (a : Ops α) (p : Parameters α) (row pair : Nat) : α :=
  a.divide (a.ofInt (p.start+(row:Int)))
    (a.pow p.theta (a.divide (a.ofInt ((pair%p.headDim:Nat):Int)) (a.literalNat p.headDim)))

def value (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (rotary : Bool) (row n : Nat) : α :=
  if rotary then
    let pair := n/2*2
    let phase := angle a p row pair
    let real := projection a p x plans r pair
    let imag := projection a p x plans r (pair+1)
    if n%2=0 then a.sub (a.mul real (a.cos phase)) (a.mul imag (a.sin phase))
    else a.add (a.mul real (a.sin phase)) (a.mul imag (a.cos phase))
  else projection a p x plans r n

/-- Independent native reference. Each block owns a fresh MMA stream with
zero seed, then a separate outer fold adds completed block results. This
definition does not call the source recurrence or an execution interpreter. -/
def referenceAccumulation (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → DotPlan) (n count : Nat) : α :=
  ((List.range count).map (fun t =>
    (plans t).instructions.foldl (fun seed lanes =>
      a.mma8 (fun j => RmsMatmulRbe.scaled a.toOps p x t (lanes j))
        (fun j => masked a.toOps p (x.weight n) t (lanes j)) seed) a.zero)).foldl a.add a.zero

def referenceMoment (a : Ops α) (p : Parameters α) (x : RowInput α) (l : Fin 64) : α :=
  (List.range (steps p.K)).foldl
    (fun old t => a.add old (a.powi (masked a.toOps p x.x t l) 2)) a.zero

def referenceNorm (a : Ops α) (p : Parameters α) (x : RowInput α) (r : Reduction) : α :=
  a.rsqrt (a.add (a.divide (r.tree.eval a.add (referenceMoment a p x))
    (a.ofInt p.K)) p.epsilon)

def referenceProjection (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (n : Nat) : α :=
  a.mul (referenceAccumulation a p x (plans n) n (steps p.K)) (referenceNorm a p x r)

def referenceValue (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (rotary : Bool) (row n : Nat) : α :=
  let project := referenceProjection a p x plans r
  if rotary then
    let even := n-n%2
    let phase := a.divide (a.ofInt (p.start+(row:Int)))
      (a.pow p.theta (a.divide (a.ofInt ((even%p.headDim:Nat):Int)) (a.literalNat p.headDim)))
    if n%2=0 then a.sub (a.mul (project even) (a.cos phase))
      (a.mul (project (even+1)) (a.sin phase))
    else a.add (a.mul (project even) (a.sin phase))
      (a.mul (project (even+1)) (a.cos phase))
  else project n

/-- Storage conversions are scalar ports. E5 RMS uses E5→FP16→FP32;
E5 weight uses E5→FP32→FP16→FP32. There is one final output conversion. -/
structure StorageOps (α H : Type) extends toSource : Ops α where
  hzero : H
  widen : H → α
  narrow : α → H
  e5Half : BitVec 8 → H
  e5Single : BitVec 8 → α
  e5Encode : α → BitVec 8

def StorageOps.storage (a : StorageOps α H) : RmsMatmulRbe.StorageOps α H where
  toOps := a.toSource.toOps
  hzero := a.hzero
  widen := a.widen
  narrow := a.narrow
  e5Half := a.e5Half
  e5Single := a.e5Single
  e5Encode := a.e5Encode

def wordTarget (a : StorageOps α H) (p : Parameters α) (b : RmsMatmulRbe.Buffers)
    (plans : Fin 3 → Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Fin 3 → Nat → Nat → Reduction)
    (initial : RmsMatmulRbe.Memory α H) (x : RmsMatmulRbe.Cell s) : RmsMatmulRbe.Stored α H :=
  RmsMatmulRbe.encode a.storage b x.1 (referenceValue a.toSource p
    (RmsMatmulRbe.rowInput a.storage b initial x.1 x.2.1 x.2.2.1)
    (plans x.1 x.2.1 x.2.2.1) (reductions x.1 x.2.1 x.2.2.1)
    (RmsMatmulRbe.rotary x.1) x.2.2.1 x.2.2.2)

theorem accumulation_reference (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → DotPlan) (n t : Nat) :
    accumulation a p x plans n t=referenceAccumulation a p x plans n t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [accumulation,ih,referenceAccumulation,List.range_succ,List.map_append,
      List.map_cons,List.map_nil,List.foldl_append,List.foldl_cons,List.foldl_nil,
      dotBlock,RmsMatmulRbe.instruction]

theorem moment_reference (a : Ops α) (p : Parameters α) (x : RowInput α) (l : Fin 64) :
    moment a p x l (steps p.K)=referenceMoment a p x l := by
  suffices h : ∀ t,moment a p x l t=(List.range t).foldl
      (fun old t => a.add old (a.powi (masked a.toOps p x.x t l) 2)) a.zero from h _
  intro t
  induction t with
  | zero => rfl
  | succ t ih => simp [moment,ih,List.range_succ,List.foldl_append]

theorem projection_reference (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (n : Nat) :
    projection a p x plans r n=referenceProjection a p x plans r n := by
  simp only [projection,referenceProjection,accumulation_reference,norm,referenceNorm,moment_reference]

theorem value_reference (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (rotary : Bool) (row n : Nat) :
    value a p x plans r rotary row n=referenceValue a p x plans r rotary row n := by
  have pair : n/2*2=n-n%2 := by omega
  simp only [value,referenceValue,projection_reference,pair,angle]

#axiomsClean accumulation_reference
#axiomsClean moment_reference
#axiomsClean projection_reference
#axiomsClean value_reference
#specNonCircular referenceValue avoiding [value,accumulation,moment,projection]
#specNonCircular wordTarget avoiding [value,accumulation,moment,projection,RmsMatmulRbe.accumulation]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
