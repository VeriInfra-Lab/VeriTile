import bench.tritonbench_g.matmul_triton_autotune.Proofs.Interface
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Mathematical_Real

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface
open VeriTile.Triton FP.Structural Whole
noncomputable section

theorem real_implements (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C ≠ g.A) (sepB : g.C ≠ g.B) :
    (programIO (idealAlgebra id) idealDot g ids).Implements (fun values =>
      result g (Whole.mathematical g.request (left values) (right values))) := by
  intro values domain bounds hb s hin
  obtain ⟨safe, final, ran, observed, frame⟩ := implements (idealAlgebra id) idealDot
    (idealNumerics id) (ideal_primitives id) g ids order sepA sepB values domain bounds hb s hin
  obtain ⟨a, b⟩ := input_values (idealAlgebra id) idealDot g ids values s hin
  refine ⟨safe, final, ran, ?_, frame⟩
  intro o j active
  have output := observed o j active
  fin_cases o
  have native := reference_logical (idealNumerics id) g.request s (left values) (right values) a b j.1 j.2.1
  have formula := ideal_reference id g.request s (left values) (right values) a b j.1 j.2.1
  rw [native] at formula
  change mapCell id (final.mem g.C (outputOffset g.request j.1.val j.2.1.val)) =
    .mk .fp16 (logicalReference (idealNumerics id) g.request (left values) (right values) j.1 j.2.1) at output
  rw [formula] at output
  exact output

#axiomsClean real_implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface
