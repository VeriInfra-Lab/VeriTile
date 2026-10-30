import bench.tritonbench_g.rope_backward_transform.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.LogicalIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

@[simp] theorem query_input (r : Request) (raw : PublicIO.Inputs ℝ r)
    (key : Bool) (row head col : Nat) :
    query (inputEquiv r raw) key row head col = PublicIO.q raw key row head col := by
  cases key <;> rfl

@[simp] theorem coefficient_input (r : Request) (raw : PublicIO.Inputs ℝ r)
    (sine : Bool) (row lane : Nat) :
    coefficient (inputEquiv r raw) sine row lane = PublicIO.coefficient raw sine row lane := by
  cases sine <;> rfl

/-- Full previous result-read judgment, including all odd-tail observations. -/
theorem judgment_iff (r : Request) (f : Inputs r → Outputs r) :
    (view ideal id r ⊨ f) ↔
    (PublicIO.sourceIO ideal id r).Implements
      (fun raw => (outputEquiv r).symm (f (inputEquiv r raw))) := Iff.rfl

/-- The established independent adjoint formula, on the unchanged raw interface. -/
def rawFormula (r : Request) (xs : PublicIO.Inputs ℝ r)
    (key : Bool) (row head col : Nat) : ℝ :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    let a := PublicIO.q xs key row head j
    let b := PublicIO.q xs key row head (j+r.half)
    let c := PublicIO.coefficient xs Bool.false (row%r.sequence) j
    let s := PublicIO.coefficient xs Bool.true (row%r.sequence) j
    if col<r.half then a*c+b*s else b*c-a*s
  else PublicIO.q xs key row head col

theorem raw_correct (r : Request) :
    (PublicIO.sourceIO ideal id r).Implements (fun xs => PublicIO.result (rawFormula r xs)) := by
  apply PublicIO.implements_formula
  intro xs key row head col
  simp only [PublicIO.logicalResult,rawFormula,id_eq]
  split
  · rw [ideal_pair]
    by_cases low : col<r.half <;>
      simp [low,show (r.half≤col)↔¬col<r.half from by omega,Spec.pair]
  · rfl

/-- The same adjoint polynomial evaluated on named logical tensor coordinates. -/
def formula (r : Request) (xs : Inputs r)
    (key : Bool) (row head col : Nat) : ℝ :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    let a := query xs key row head j
    let b := query xs key row head (j+r.half)
    let c := coefficient xs Bool.false (row%r.sequence) j
    let s := coefficient xs Bool.true (row%r.sequence) j
    if col<r.half then a*c+b*s else b*c-a*s
  else query xs key row head col

def mathematical (r : Request) (xs : Inputs r) : Outputs r where
  query := fun row head col => formula r xs Bool.false row.val head.val col.val
  key := fun row head col => formula r xs Bool.true row.val head.val col.val

/-- Equality of complete old/new targets, including every odd-tail result read. -/
theorem target_eq (r : Request) :
    (fun raw => (outputEquiv r).symm (mathematical r (inputEquiv r raw))) =
      (fun raw => PublicIO.result (rawFormula r raw)) := by
  funext raw port
  fin_cases port
  · funext j
    change formula r (inputEquiv r raw) Bool.false j.1.val j.2.1.val j.2.2.1.val =
      rawFormula r raw Bool.false j.1.val j.2.1.val j.2.2.1.val
    simp only [formula,rawFormula,query_input,coefficient_input]
  · funext j
    change formula r (inputEquiv r raw) Bool.true j.1.val j.2.1.val j.2.2.1.val =
      rawFormula r raw Bool.true j.1.val j.2.1.val j.2.2.1.val
    simp only [formula,rawFormula,query_input,coefficient_input]

theorem logical_correct (r : Request) : view ideal id r ⊨ mathematical r := by
  apply (judgment_iff r _).mpr
  exact (congrArg (fun f => (PublicIO.sourceIO ideal id r).Implements f) (target_eq r)).mpr
    (raw_correct r)

end
end VeriTile.Bench.Repairs.RopeBackwardTransform.LogicalIO
