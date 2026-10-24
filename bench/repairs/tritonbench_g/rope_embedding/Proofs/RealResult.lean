import bench.repairs.tritonbench_g.rope_embedding.Proofs.Kernel
import bench.repairs.tritonbench_g.rope_embedding.Proofs.SourceBinding
namespace VeriTile.Bench.Repairs.RopeEmbedding.KernelRealResult
open VeriTile.Triton FP.Structural
noncomputable section

/-- Every head and both halves in an admitted program execute successfully.
The target is the independent polynomial in Contract; coefficients need not
lie on the unit circle. Odd tails and every unwritten cell are preserved. -/
theorem rope_embedding_kernel_correct (r : Request) (back : Bool) (s : State ℝ)
    (row group : Nat) (launch : r.Launched row group) :
    ∃ final, FP.Structural.exec ideal.backend (Source.source (parameters r) back)
        (programState s row group)=some final ∧
      Kernel.Post r row group s final (fun h c => Spec.result r back
        (fun o => (s.mem "Q" o).read .real) (fun o => (s.mem "C" o).read .real)
        (fun o => (s.mem "S" o).read .real) row h c) := by
  simpa only [ideal_result] using Kernel.executes ideal r back s row group launch.2.1 launch.2.2

/-- The specified launch geometry covers every rotated coordinate. -/
theorem rope_embedding_kernel_coverage (r : Request) (row h c : Nat)
    (hr : row < r.rows) (hh : h < r.heads) (hc : c < 2*r.half) :
    ∃ group,r.Launched row group ∧ Kernel.Written r row group "Q" (r.offset row h c) := by
  obtain ⟨g,hg,hw⟩ := Kernel.complete_coverage r row h c hr hh hc
  exact ⟨g,⟨by omega,hr,hg⟩,hw⟩

#axiomsClean rope_embedding_kernel_correct
#axiomsClean rope_embedding_kernel_coverage
end
end VeriTile.Bench.Repairs.RopeEmbedding.KernelRealResult
