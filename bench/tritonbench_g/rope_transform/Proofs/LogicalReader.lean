import bench.tritonbench_g.rope_transform.Proofs.IOGrid
import bench.tritonbench_g.rope_transform.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.RopeTransform.LogicalIO
open VeriTile.Triton FP.Structural KernelContract
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

@[simp] theorem query_input (r : Request) (raw : PublicIO.Inputs ℝ r)
    (key : Bool) (row head col : Nat) :
    query (inputEquiv r raw) key row head col = PublicIO.q raw key row head col := by
  cases key <;> rfl

@[simp] theorem coefficient_input (r : Request) (raw : PublicIO.Inputs ℝ r)
    (sine : Bool) (row lane : Nat) :
    coefficient (inputEquiv r raw) sine row lane = PublicIO.coefficient raw sine row lane := by
  cases sine <;> rfl

/-- Complete previous typed source judgment, with every original input/output. -/
theorem judgment_iff (r : Request) (f : Inputs r → Outputs r) :
    (view realAlgebra r ⊨ f) ↔
    (PublicIO.sourceIO realAlgebra id id r).Implements
      (fun raw => (outputEquiv r).symm (f (inputEquiv r raw))) := Iff.rfl

def pair (a b c s : ℝ) (second : Bool) : ℝ := if second then b*c+a*s else a*c-b*s

theorem raw_correct (r : Request) :
    (PublicIO.sourceIO realAlgebra id id r).Implements (PublicIO.target pair r) := by
  have proved := PublicIO.implements realAlgebra (id : ℝ → ℝ) id r
  have same : PublicIO.scalarTarget realAlgebra (id : ℝ → ℝ) id r =
      PublicIO.target pair r := by rfl
  rw [same] at proved
  exact proved

def rotation (r : Request) (xs : Inputs r)
    (key : Bool) (row head col : Nat) : ℝ :=
  let lane := if col<r.half then col else col-r.half
  let a := query xs key row head lane
  let b := query xs key row head (lane+r.half)
  let c := coefficient xs Bool.false (row%r.sequence) lane
  let s := coefficient xs Bool.true (row%r.sequence) lane
  if decide (r.half≤col) then b*c+a*s else a*c-b*s

def mathematical (r : Request) (xs : Inputs r) : Outputs r where
  query := fun row head col => rotation r xs Bool.false row.val head.val col.val
  key := fun row head col => rotation r xs Bool.true row.val head.val col.val

/-- Equality of both complete tensor targets, not only the written half lanes. -/
theorem target_eq (r : Request) :
    (fun raw => (outputEquiv r).symm (mathematical r (inputEquiv r raw))) =
      PublicIO.target pair r := by
  funext raw port
  fin_cases port
  · funext j
    change rotation r (inputEquiv r raw) Bool.false j.1.val j.2.1.val j.2.2.1.val =
      pair _ _ _ _ _
    simp only [rotation,query_input,coefficient_input,pair]
  · funext j
    change rotation r (inputEquiv r raw) Bool.true j.1.val j.2.1.val j.2.2.1.val =
      pair _ _ _ _ _
    simp only [rotation,query_input,coefficient_input,pair]

theorem logical_correct (r : Request) : view realAlgebra r ⊨ mathematical r := by
  apply (judgment_iff r _).mpr
  exact (congrArg (fun f => (PublicIO.sourceIO realAlgebra id id r).Implements f) (target_eq r)).mpr
    (raw_correct r)

end
end VeriTile.Bench.Repairs.RopeTransform.LogicalIO
