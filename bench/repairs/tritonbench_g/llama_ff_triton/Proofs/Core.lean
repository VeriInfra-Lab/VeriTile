import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Math

namespace VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Shape where
  B : Nat
  M : Nat
  N : Nat
  K : Nat

def Shape.Valid (s : Shape) : Prop :=
  s.B=1 ∧ 0<s.K ∧ 0<s.N
def steps (K : Nat) := (K+63)/64

/-- Only scalar ports and one output lane of a fixed m16n8k16 half MMA.
`mulInput` is one half multiplication, returned in the common word carrier;
it preserves the source half rounding before either projection.
The live accumulator is an explicit argument. No whole dot/reduction port. -/
structure Ops (F : Type) where
  zero : F
  one : F
  add : F → F → F
  mul : F → F → F
  mulInput : F → F → F
  sigmoidNeg : F → F
  neg : F → F
  fma : F → F → F → F
  mulFTZ : F → F → F
  divide : F → F → F
  rsqrt : F → F
  ofInt : Int → F
  mma16 : (Fin 16 → F) → (Fin 16 → F) → F → F

structure Parameters (F : Type) where
  K : Nat
  epsilon : F

structure RowInput (F : Type) where
  x : Nat → F
  rms : Nat → F
  weight : Nat → Nat → F
  weight3 : Nat → Nat → F

/-- Structural compiler certificate: every K64 operand occurs once in the
selected fixed K16 instructions, in their actual instruction order. -/
structure DotPlan where
  instructions : List (Fin 16 → Fin 64)
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
  a.mulInput (masked a p x.x t l) (masked a p x.rms t l)
def instruction (a : Ops F) (p : Parameters F) (x : RowInput F) (n t : Nat)
    (lanes : Fin 16 → Fin 64) (old : F) : F :=
  a.mma16 (fun i => scaled a p x t (lanes i))
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


/-- The second projection has independent W3, sharing X, RMS and normalization. -/
def third (x : RowInput F) : RowInput F := {x with weight := x.weight3}
def value (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) : F :=
  let u := projection a p x plans reduction n
  let v := projection a p (third x) plans reduction n
  a.mul v (a.mul u (a.sigmoidNeg (a.fma (a.neg (accumulation a p x (plans n) n (steps p.K)))
    (norm a p x reduction) a.zero)))
/-- Independently flattened instruction streams for W1 and W3; terminal gate
uses the selected scalar sigmoidNeg, with no whole activation/matrix callback. -/
def referenceValue (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) : F :=
  let u := referenceProjection a p x plans reduction n
  let v := referenceProjection a p {x with weight := x.weight3} plans reduction n
  a.mul v (a.mul u (a.sigmoidNeg (a.fma (a.neg (referenceAccumulation a p x (plans n) n (steps p.K)))
    (referenceNorm a p x reduction) a.zero)))
theorem value_reference (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    value a p x plans reduction n=referenceValue a p x plans reduction n := by
  simp only [value,referenceValue,projection_reference,third,accumulation_reference,norm,referenceNorm,moment_reference]
#axiomsClean value_reference

#axiomsClean accumulation_reference
#axiomsClean moment_reference
#axiomsClean projection_reference

end
end VeriTile.Bench.Repairs.LlamaFF
