import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelValues

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Scalar
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Source scalar nodes and the compiler-observed ternary contraction. Every
input atom is an already materialized FP32 word; no input carries a graph. -/
inductive Term (α : Type) where
  | input (value : α)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Term α)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Term α)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (left right : Term α)
  | minimum (left right : Term α)
  | fma (left right addend : Term α)
  | unused

instance : Inhabited (Term α) := ⟨.unused⟩

/-- Ten local FP32 ports. Bounds are already materialized independent
runtime words. Binary min is preserved as an operation even for NaN or signed
zero operands; it is never replaced by a Boolean choice of input words. -/
structure NativeModel (α : Type) where
  biasReverse : Bool
  literal : ℝ → α
  cast : FloatDType → FloatDType → α → α
  add : α → α → α
  mul : α → α → α
  maximum : α → α → α
  exp : α → α
  log : α → α
  minimum : α → α → α
  lessEqual : α → α → Bool
  fma : α → α → α → α
  unused : α

/-- Exact local operation interpretation. The pinned preparation graph uses
exp(x)+1; the source surface writes 1+exp(x). No commutation law is invoked.
The FMA is a separate node and a separate ternary numerical primitive. -/
def interpret (M : NativeModel α) : Term α → α
  | .input x => x
  | .literal (some .fp32) .real x => M.literal x
  | .cast (some .fp32) a b x => M.cast a b (interpret M x)
  | .unary (some .fp32) .exp x => M.exp (interpret M x)
  | .unary (some .fp32) .log x => M.log (interpret M x)
  | .binary (some .fp32) .real .add (.cast (some .fp32) .fp32 .real x)
      (.cast (some .fp32) .real .real bias) =>
      let x := M.cast .fp32 .real (interpret M x)
      let bias := M.cast .real .real (interpret M bias)
      if M.biasReverse then M.add bias x else M.add x bias
  | .binary (some .fp32) .real .add (.literal (some .fp32) .real one)
      (.unary (some .fp32) .exp x) => M.add (M.exp (interpret M x)) (M.literal one)
  | .binary (some .fp32) .real .add x y => M.add (interpret M x) (interpret M y)
  | .binary (some .fp32) .real .mul x y => M.mul (interpret M x) (interpret M y)
  | .binary (some .fp32) .real .max x y => M.maximum (interpret M x) (interpret M y)
  | .minimum x y => M.minimum (interpret M x) (interpret M y)
  | .fma x y z => M.fma (interpret M x) (interpret M y) (interpret M z)
  | _ => M.unused

/-- A contraction is permitted only at an explicitly marked scan node. Its
input must still be the original multiplication graph, not a rounded word. -/
def contract : Term α → Term α → Term α
  | .binary (some .fp32) .real .mul x y,z => .fma x y z
  | _,_ => .unused

def graphScan (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (precision : Option ComputeDType) {shape : TileShape} (op : ScanOp)
    (ax : Fin shape.length) (direction : ScanDirection)
    (x : FP.Structural.Values (Term α) .real shape) : Option (FP.Structural.Values (Term α) .real shape) := by
  classical
  exact if hs : shape = [c.BH,c.BC] then
    if precision = some .fp32 ∧ op = .sum ∧ ax.val = 1 ∧ direction = .forward then
      let xx : FP.Structural.Values (Term α) .real [c.BH,c.BC] := hs ▸ x
      let yy : FP.Structural.Values (Term α) .real [c.BH,c.BC] := fun i =>
        (plan.tree i.1 i.2.1).eval (Term.binary (some .fp32) .real .add) contract
          (fun j => xx (i.1,j,PUnit.unit))
      some (hs.symm ▸ yy)
    else none
  else none

def graphAlgebra (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC) : Algebra (Term α) where
  literal := Term.literal
  negInf := .unused
  binary := Term.binary
  minimum := fun _ _ => some Term.minimum
  unary := Term.unary
  cast := Term.cast
  fromNat := fun _ _ => .unused
  fromInt := fun _ _ => .unused
  fp32Bits := fun _ => .unused
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => .unused
  reduceSum := fun _ {_} _ _ _ _ => .unused
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareLe := fun _ _ => some (fun x y => M.lessEqual (interpret M x) (interpret M y))
  scan := graphScan c plan

def primitives (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC) : Primitives (Term α) c where
  model := graphAlgebra M c plan
  minimum := Term.minimum
  lessEqual := fun x y => M.lessEqual (interpret M x) (interpret M y)
  le_support := rfl
  minimum_support := rfl
  fused := contract
  plan := plan
  scan_support := by intro x; simp [graphAlgebra,graphScan]

/-- Independent scalar recurrence in materialized native words. -/
def referencePrepared (M : NativeModel α) (hasBias useSoftplus : Bool) (lower upper : α)
    (active : Bool) (x bias : α) : α :=
  let x := M.cast .fp32 .real x
  let bias := M.cast .real .real bias
  let x := if hasBias then (if M.biasReverse then M.add bias x else M.add x bias) else x
  let x := if useSoftplus then
    if M.lessEqual x (M.literal 20.0) then M.log (M.add (M.exp x) (M.literal 1.0)) else x
    else x
  let x := M.maximum x lower
  let x := M.minimum x upper
  if active then x else M.literal 0.0

/-- Independent mixed multiply/add/FMA recurrence on raw prepared values.
An FMA reads its own dt and A operands directly; a prior rounded product is
never substituted for them. The ordered graph is selected by native profile. -/
def referenceTree (M : NativeModel α) (prepared : Fin n → α) (rate : α) : SumTree n → α
  | .leaf i => M.mul (prepared i) rate
  | .add l r => M.add (referenceTree M prepared rate l) (referenceTree M prepared rate r)
  | .fused i r => M.fma (prepared i) rate (referenceTree M prepared rate r)

theorem source_preparation (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias useSoftplus : Bool) (lower upper : α) (active : Bool) (x bias : α) :
    interpret M (prepare c (primitives M c plan) hasBias useSoftplus (.input lower) (.input upper) active (.input x) (.input bias)) =
      referencePrepared M hasBias useSoftplus lower upper active x bias := by
  cases active <;> cases hasBias <;> cases useSoftplus
  all_goals simp [prepare,primitives,graphAlgebra,interpret,referencePrepared]
  all_goals split_ifs <;> simp_all [interpret]

/-- The source-derived scan graph lowers to the independent native recurrence.
Every exact FMA marker is structural; no floating identity occurs here. -/
theorem source_scan (M : NativeModel α) (prepared : Fin n → Term α) (rate : Term α) (tree : SumTree n) :
    interpret M (tree.eval (Term.binary (some .fp32) .real .add) contract
      (fun i => .binary (some .fp32) .real .mul (prepared i) rate)) =
      referenceTree M (fun i => interpret M (prepared i)) (interpret M rate) tree := by
  induction tree with
  | leaf i => rfl
  | fused i r ih => simpa only [SumTree.eval,contract,interpret,referenceTree] using congrArg (M.fma _ _) ih
  | add l r il ir =>
    simp only [SumTree.eval,referenceTree]
    cases l <;> cases r <;> simp only [SumTree.eval,contract,interpret] at il ir ⊢ <;> rw [il,ir]


/-- Logical input values, indexed independently of launch tiling. -/
structure LogicalInputs (α : Type) where
  dt : Nat → Nat → Nat → α
  rate : Nat → α
  bias : Nat → α

def preparedValue (M : NativeModel α) (c : Geometry) (a : LogicalInputs α)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k j : Nat) : α :=
  if h<c.heads ∧ j<c.chunk ∧ k*c.chunk+j<c.sequence then
    referencePrepared M hasBias useSoftplus lower upper Bool.true
      (a.dt b (k*c.chunk+j) h) (a.bias h)
  else M.literal 0

def cumulativeValue (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (a : LogicalInputs α) (hasBias useSoftplus : Bool) (lower upper : α)
    (b h k : Nat) (j : Fin c.BC) : α :=
  referenceTree M (fun t => preparedValue M c a hasBias useSoftplus lower upper b h k t.val)
    (M.cast .fp32 .real (a.rate h)) (plan.tree ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j)

#guard_msgs (drop info) in
#specNonCircular referencePrepared avoiding [interpret,Kernel.prepare,Kernel.sourceRun]
#guard_msgs (drop info) in
#specNonCircular referenceTree avoiding [interpret,Kernel.prepare,Kernel.sourceRun]
#guard_msgs (drop info) in
#specNonCircular preparedValue avoiding [interpret,Kernel.prepare,Kernel.sourceRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Scalar
