import bench.tritonbench_g.int8_dequant_matmul.Proofs.Math

namespace VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Shape where
  B : Nat
  M : Nat
  N : Nat
  K : Nat
  heads : Nat
  headDim : Nat

def Shape.Valid (s : Shape) : Prop :=
  0<s.K ∧ 0<s.heads ∧ 0<s.headDim ∧ s.headDim%2=0 ∧ s.N=s.heads*s.headDim
def steps (K : Nat) := (K+63)/64

/-- Only scalar ports and one output lane of a fixed m16n8k8 TF32 MMA.
The live accumulator is an explicit argument. No whole dot/reduction port. -/
structure Ops (F : Type) where
  zero : F
  one : F
  add : F → F → F
  mul : F → F → F
  mulFTZ : F → F → F
  divide : F → F → F
  rsqrt : F → F
  neg : F → F
  fma : F → F → F → F
  ofInt : Int → F
  pow : F → F → F
  cos : F → F
  sin : F → F
  mma8 : (Fin 8 → F) → (Fin 8 → F) → F → F

structure Parameters (F : Type) where
  K : Nat
  headDim : Nat
  start : Int
  epsilon : F
  theta : F

structure RowInput (F : Type) where
  x : Nat → F
  rms : Nat → F
  weight : Nat → Nat → F

/-- Structural compiler certificate: every K64 operand occurs once in the
selected fixed K8 instructions, in their actual instruction order. -/
structure DotPlan where
  instructions : List (Fin 8 → Fin 64)
  complete : (instructions.flatMap List.ofFn).Perm (List.finRange 64)

inductive Tree where
  | leaf (k : Fin 64)
  | node (a b : Tree)
def Tree.leaves : Tree → List (Fin 64)
  | .leaf k => [k]
  | .node a b => a.leaves++b.leaves
def Tree.eval (add : F → F → F) (v : Fin 64 → F) : Tree → F
  | .leaf k => v k
  | .node a b => add (a.eval add v) (b.eval add v)
structure Reduction where
  tree : Tree
  complete : tree.leaves.Perm (List.finRange 64)

def masked (a : Ops F) (p : Parameters F) (v : Nat → F) (t : Nat) (l : Fin 64) : F :=
  if t*64+l.val<p.K then v (t*64+l.val) else a.zero
def scaled (a : Ops F) (p : Parameters F) (x : RowInput F) (t : Nat) (l : Fin 64) : F :=
  a.mul (masked a p x.x t l) (masked a p x.rms t l)
def instruction (a : Ops F) (p : Parameters F) (x : RowInput F) (n t : Nat)
    (lanes : Fin 8 → Fin 64) (old : F) : F :=
  a.mma8 (fun i => scaled a p x t (lanes i))
    (fun i => masked a p (x.weight n) t (lanes i)) old
def dotBlock (a : Ops F) (p : Parameters F) (x : RowInput F) (plan : DotPlan)
    (n t : Nat) (old : F) := plan.instructions.foldl (fun v lanes => instruction a p x n t lanes v) old
def accumulation (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → DotPlan) (n : Nat) : Nat → F
  | 0 => a.zero
  | t+1 => dotBlock a p x (plans t) n t (accumulation a p x plans n t)

/-- Native libdevice integer power lowering is x*x followed by 1*square,
both FTZ multiplies. This deliberately does not identify it with x*x. -/
def square (a : Ops F) (x : F) := a.mulFTZ a.one (a.mulFTZ x x)
def moment (a : Ops F) (p : Parameters F) (x : RowInput F) (l : Fin 64) : Nat → F
  | 0 => a.zero
  | t+1 => a.add (moment a p x l t) (square a (masked a p x.x t l))
def norm (a : Ops F) (p : Parameters F) (x : RowInput F) (reduction : Reduction) : F :=
  a.rsqrt (a.add (a.divide
    (reduction.tree.eval a.add (fun l => moment a p x l (steps p.K)))
    (a.ofInt p.K)) p.epsilon)
def projection (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) : F :=
  a.mul (accumulation a p x (plans n) n (steps p.K)) (norm a p x reduction)
def angle (a : Ops F) (p : Parameters F) (row pair : Nat) : F :=
  a.divide (a.ofInt (p.start+(row:Int)))
    (a.pow p.theta (a.divide (a.ofInt ((pair%p.headDim : Nat):Int)) (a.ofInt p.headDim)))
def rotate (a : Ops F) (u v angle : F) : F × F :=
  (a.fma u (a.cos angle) (a.neg (a.mul v (a.sin angle))),
   a.fma v (a.cos angle) (a.mul u (a.sin angle)))
def value (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (rotary : Bool) (row n : Nat) : F :=
  if rotary then
    let pair := n/2*2
    let uv := rotate a (projection a p x plans reduction pair)
      (projection a p x plans reduction (pair+1)) (angle a p row pair)
    if n%2=0 then uv.1 else uv.2
  else projection a p x plans reduction n

/-- Independent word reference: collect all fixed instructions with their
logical K-block labels, then fold once over that stream. -/
def instructionStream (plans : Nat → DotPlan) (count : Nat) :=
  (List.range count).flatMap (fun t => (plans t).instructions.map (fun lanes => (t,lanes)))
def referenceAccumulation (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → DotPlan) (n count : Nat) :=
  (instructionStream plans count).foldl
    (fun old e => instruction a p x n e.1 e.2 old) a.zero
def referenceMoment (a : Ops F) (p : Parameters F) (x : RowInput F) (l : Fin 64) :=
  (List.range (steps p.K)).foldl
    (fun old t => a.add old (square a (masked a p x.x t l))) a.zero
def referenceNorm (a : Ops F) (p : Parameters F) (x : RowInput F) (reduction : Reduction) :=
  a.rsqrt (a.add (a.divide (reduction.tree.eval a.add (referenceMoment a p x))
    (a.ofInt p.K)) p.epsilon)
def referenceProjection (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :=
  a.mul (referenceAccumulation a p x (plans n) n (steps p.K)) (referenceNorm a p x reduction)
def referenceValue (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (rotary : Bool) (row n : Nat) : F :=
  let project := referenceProjection a p x plans reduction
  if rotary then
    let even := n-n%2
    let phi := a.divide (a.ofInt (p.start+(row:Int)))
      (a.pow p.theta (a.divide (a.ofInt ((even%p.headDim : Nat):Int)) (a.ofInt p.headDim)))
    if n%2=0 then a.fma (project even) (a.cos phi) (a.neg (a.mul (project (even+1)) (a.sin phi)))
    else a.fma (project (even+1)) (a.cos phi) (a.mul (project even) (a.sin phi))
  else project n

theorem accumulation_reference (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → DotPlan) (n t : Nat) :
    accumulation a p x plans n t=referenceAccumulation a p x plans n t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [accumulation,ih,referenceAccumulation,instructionStream,List.range_succ,
      List.flatMap_append,List.flatMap_cons,List.flatMap_nil,List.append_nil,
      List.foldl_append,List.foldl_map,dotBlock]
theorem moment_reference (a : Ops F) (p : Parameters F) (x : RowInput F) (l : Fin 64) :
    moment a p x l (steps p.K)=referenceMoment a p x l := by
  suffices h : ∀ t,moment a p x l t=(List.range t).foldl
      (fun old t => a.add old (square a (masked a p x.x t l))) a.zero from h _
  intro t
  induction t with
  | zero => rfl
  | succ t ih => simp [moment,ih,List.range_succ,List.foldl_append]
theorem projection_reference (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    projection a p x plans reduction n=referenceProjection a p x plans reduction n := by
  simp only [projection,referenceProjection,accumulation_reference,norm,referenceNorm,moment_reference]
theorem value_reference (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (rotary : Bool) (row n : Nat) :
    value a p x plans reduction rotary row n=referenceValue a p x plans reduction rotary row n := by
  have pair : n/2*2=n-n%2 := by omega
  simp only [value,referenceValue,projection_reference,pair,rotate,angle]

#axiomsClean accumulation_reference
#axiomsClean moment_reference
#axiomsClean projection_reference
#axiomsClean value_reference
end
end VeriTile.Bench.Repairs.RmsMatmulRbe
