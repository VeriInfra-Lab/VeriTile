import bench.tritonbench_g.swiglu_triton.IO
import bench.tritonbench_g.swiglu_triton.Proofs.PublicIO
namespace VeriTile.Bench.Repairs.SwigluTriton.Logical
open VeriTile.Triton FP.Structural
noncomputable section
theorem judgment_iff (backward : Bool) (block cols row : Nat)
    (f : Channel → ℝ → ℝ → ℝ → ℝ) :
    (io backward block cols row).Implements (fun xs ch i => f ch (xs.a i) (xs.b i) (xs.dc i)) ↔
    (PublicIO.io Math.algebra id id backward block cols row).Implements (PublicIO.formula f block) := by
  have same : (io backward block cols row).lift (fun xs ch i => f ch (xs.a i) (xs.b i) (xs.dc i)) =
      PublicIO.formula f block := by
    funext xs port
    fin_cases port <;> rfl
  change (PublicIO.io Math.algebra id id backward block cols row).Implements _ ↔ _
  rw [same]
private def readerScalar (ch : Channel) (a b dc : ℝ) : ℝ :=
  let sigmoid := (1 + Real.exp (-a))⁻¹
  match ch with
  | .c => a*b*sigmoid
  | .da => dc*b*sigmoid*(1+a*(1-sigmoid))
  | .db => dc*a*sigmoid

theorem reader_implements (backward : Bool) (block cols row : Nat) :
    (io backward block cols row).Implements (fun xs ch i =>
      readerScalar ch (xs.a i) (xs.b i) (xs.dc i)) := by
  apply (judgment_iff backward block cols row readerScalar).2
  have scalar_eq : readerScalar = Math.formula := by
    funext ch a b dc
    cases ch <;> rfl
  simpa only [scalar_eq] using PublicIO.implements Math.algebra id id Math.formula
    (fun ch a b dc => Math.scalar_real ch a b dc) backward block cols row

end
end VeriTile.Bench.Repairs.SwigluTriton.Logical
