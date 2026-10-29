import bench.tritonbench_g.layer_norm_fwd.Proofs.RealSurface
import bench.tritonbench_g.layer_norm_fwd.Proofs.RealIO
import VeriTile.Meta.Specification

/-! Complete original and candidate real execution with an independent
initial-memory specification for all seven outputs and the full tagged frame.
All source flags, row strides and block sizes remain symbolic. The standard
region realization and finite-memory execution/frame are included explicitly.
Finite IO additionally requires the valid row, active allocation extents and
zero undef used by the flatten bridge. There is no intermediate-execution or
desired-output premise. Other storage profiles remain outside this fp32 scope;
the explicit typed consistency witness is exposed separately in FPEquiv. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Correctness
open VeriTile VeriTile.Triton
noncomputable section

def original (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : ComputeKernel :=
  (LayerNormFwd.original f r stride M N eps p block hBlock).eraseDType

def indexCommuted (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : ComputeKernel :=
  (LayerNormFwd.candidate f r stride M N eps p block hBlock).eraseDType

specification layer_norm_fwd_original_real (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    Spec.Real ((∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
      ∃ final, exec (original f r stride M N eps p block hBlock).toAlgKernel initial = some final ∧
        Contract.Post f r stride M N eps p initial final) ∧
      (∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (original f r stride M N eps p block hBlock) initial
          (RealSurface.writes f r stride M N initial.pid)
          (RealSurface.expected f r stride M N eps p initial)) ∧
      RealIO.Implements (original f r stride M N eps p block hBlock).toAlgKernel
        f r stride M N eps p block) := by
  exact ⟨RealExecution.source_correct f r stride M N eps p block hBlock,
    RealSurface.original_realizes f r stride M N eps p block hBlock,
    RealIO.original_correct f r stride M N eps p block hBlock⟩

specification layer_norm_fwd_candidate_real (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    Spec.Real ((∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
      ∃ final, exec (indexCommuted f r stride M N eps p block hBlock).toAlgKernel initial = some final ∧
        Contract.Post f r stride M N eps p initial final) ∧
      (∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (indexCommuted f r stride M N eps p block hBlock) initial
          (RealSurface.writes f r stride M N initial.pid)
          (RealSurface.expected f r stride M N eps p initial)) ∧
      RealIO.Implements (indexCommuted f r stride M N eps p block hBlock).toAlgKernel
        f r stride M N eps p block) := by
  exact ⟨CandidateExecution.source_correct f r stride M N eps p block hBlock,
    RealSurface.candidate_realizes f r stride M N eps p block hBlock,
    RealIO.candidate_correct f r stride M N eps p block hBlock⟩

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Correctness
