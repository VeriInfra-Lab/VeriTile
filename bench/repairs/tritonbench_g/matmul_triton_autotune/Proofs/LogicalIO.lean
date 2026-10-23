import bench.repairs.tritonbench_g.matmul_triton_autotune.IO
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.RealIO

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.LogicalIOProof
open VeriTile.Triton FP.Structural Whole
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def target (g : Geometry) (ab : (Fin g.rows → Fin g.inner → ℝ) ×
    (Fin g.inner → Fin g.columns → ℝ)) (row : Fin g.rows) (col : Fin g.columns) : ℝ :=
  let value := ∑ k : Fin g.inner, ab.1 row k * ab.2 k col
  if g.activation == "leaky_relu" then
    if 0 ≤ value then value else (1 / 100 : ℝ) * value else value

theorem headline_iff (g : Geometry) (ids : List Nat) :
    (Interface.logicalIO (idealAlgebra id) idealDot g ids ⊨ target g) ↔
    (Interface.programIO (idealAlgebra id) idealDot g ids).Implements (fun values =>
      Interface.result g (Whole.mathematical g.request (Interface.left values) (Interface.right values))) := by
  have formula : (Interface.logicalIO (idealAlgebra id) idealDot g ids).lift (target g) =
      (fun values => Interface.result g (Whole.mathematical g.request (Interface.left values) (Interface.right values))) := by
    funext values port i
    fin_cases port
    rfl
  change (Interface.programIO _ _ _ _).Implements _ ↔ _
  rw [formula] <;> rfl

theorem implements (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C ≠ g.A) (sepB : g.C ≠ g.B) :
    (Interface.logicalIO (idealAlgebra id) idealDot g ids ⊨ target g) :=
  (headline_iff g ids).2 (Interface.real_implements g ids order sepA sepB)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.LogicalIOProof
