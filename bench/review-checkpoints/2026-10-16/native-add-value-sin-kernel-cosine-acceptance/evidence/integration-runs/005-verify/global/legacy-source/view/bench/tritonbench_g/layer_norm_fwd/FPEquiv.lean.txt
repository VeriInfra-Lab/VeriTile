import bench.tritonbench_g.layer_norm_fwd.Proofs.FP
import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedExecution
import VeriTile.Triton.Float.ProfiledRewrite

/-! Complete typed fp32 implementation equivalence, with an empty assumption
list. Original and candidate real correctness and finite IO are proved
separately. The complete typed nonempty witness below uses an explicit
WithBot-real consistency algebra; it asserts no IEEE or native GPU execution.
Other storage profiles remain outside the proved fp32 scope. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : Program where
  kernel := LayerNormFwd.original f r stride M N eps p block hBlock
  domain := []

def candidate (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : Program where
  kernel := LayerNormFwd.candidate f r stride M N eps p block hBlock
  domain := []

specification layer_norm_fwd_fp32_equiv (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    original f r stride M N eps p block hBlock ≡[assumptions]
      candidate f r stride M N eps p block hBlock := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl (by rfl)
  intro α _ A D _ u _
  exact FPProof.whole_state f r stride M N eps p block hBlock (engine A) u

/-- Both complete typed kernels successfully execute in the stated consistency
algebra, starting with empty registers and negative int64 seeds at PID 1.
All optional features are enabled, N = 2 and BLOCK_N = 4, and both normalization
modes are covered. This is a nonempty model witness, not an IEEE/native run. -/
theorem layer_norm_fwd_fp32_nonempty (rms : Bool) (p : ℝ) :
    ∃ final,
      FP.Structural.exec (engine TypedExecution.model)
        (original (TypedExecution.flags rms) TypedExecution.regions TypedExecution.strides
          2 2 1 p 4 TypedExecution.legal_block).kernel TypedExecution.initial = some final ∧
      FP.Structural.exec (engine TypedExecution.model)
        (candidate (TypedExecution.flags rms) TypedExecution.regions TypedExecution.strides
          2 2 1 p 4 TypedExecution.legal_block).kernel TypedExecution.initial = some final :=
  TypedExecution.nonempty_pair rms p

end VeriTile.Bench.TritonBenchG.LayerNormFwd.Equivalence
