import bench.repairs.tritonbench_g.parallel_attention.Kernels
import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeExecutionReal
import bench.repairs.tritonbench_g.parallel_attention.Proofs.RealContract
import bench.repairs.tritonbench_g.parallel_attention.Proofs.WrapperReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Every active result cell and the entire untouched memory complement. -/
theorem sourceCorrect (p : Plans) (s : Shape) (domain : ShapeContract s) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (kinds : List OutputKind)
    (ks hs : List Event) (layout : RunLayout s b trees scratch output kinds ks hs) (m : Memory ℝ) :
    (∀ kind,kind∈kinds → ∀ bh t e,bh < s.B*s.H → t < s.T → e < kind.width s →
      kernelRun realOps p s b trees scratch output ks hs m (output kind+dense s.T (kind.width s) bh t e)=
        mathematicalResult s kind (currentInputs realOps s b m) bh t e) ∧
    (∀ address,(∀ ev∈ks,tensorAddress scratch ev ≠ address) →
      (∀ ev∈hs,tensorAddress output ev ≠ address) →
      kernelRun realOps p s b trees scratch output ks hs m address=m address) := by
  constructor
  · intro kind hk bh t e hb ht he
    dsimp only [kernelRun,mathematicalResult]
    rw [combined_output realOps p s b trees scratch output kinds ks hs
      layout.kernelCoverage layout.hostCoverage layout.kernelFresh layout.kernelUnique
      layout.hostFresh layout.hostUnique m kind hk _ (dense_bound _ _ _ _ _ _ hb ht he)]
    exact wordReference_spec p s kind (trees kind) (layout.hostTrees kind hk) _ bh t e hb ht he
      domain.key_positive (domain.key_fits s) domain.value_positive (domain.value_tile_positive s)
  · exact combined_frame realOps p s b trees scratch output ks hs m

/-- All public flags, including return-both priority, read actual completed O/Z. -/
theorem public_forward (p : Plans) (s : Shape) (domain : ShapeContract s) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s b trees scratch output forwardKinds ks hs) (m : Memory ℝ)
    (bh : Nat) (hb : bh < s.B*s.H) (eps : ℝ) (returnBoth normalize : Bool) :
    PublicResult.Agrees s.T s.V
      (publicRead realOps eps s (output .forward) (output .normalizer)
        (kernelRun realOps p s b trees scratch output ks hs m) bh returnBoth normalize)
      (publicSpec s.T s.K (selectedScale s.K s.useScale) eps
        (headInputs (currentInputs realOps s b m) bh) returnBoth normalize) := by
  apply public_read_real
  · intro t ht e he
    exact (sourceCorrect p s domain b trees scratch output forwardKinds ks hs layout m).1
      .forward (by simp [forwardKinds]) bh t e hb ht he
  · intro t ht
    exact (sourceCorrect p s domain b trees scratch output forwardKinds ks hs layout m).1
      .normalizer (by simp [forwardKinds]) bh t 0 hb ht (by norm_num [OutputKind.width])

def returnedPairing (s : Shape) (bh : Nat) (dx : Inputs ℝ) (output : OutputKind → Nat) (m : Memory ℝ) : ℝ :=
  (∑ t∈Finset.range s.T,∑ e∈Finset.range s.K,dx.q t e*m (output .query+dense s.T s.K bh t e))+
  (∑ t∈Finset.range s.T,∑ e∈Finset.range s.K,dx.k t e*m (output .key+dense s.T s.K bh t e))+
  (∑ t∈Finset.range s.T,∑ e∈Finset.range s.V,dx.v t e*m (output .value+dense s.T s.V bh t e))

/-- The returned gradient memory, rather than a separately defined gradient
oracle, is the VJP of the independent O/Z mathematical function. -/
theorem returned_vjp (p : Plans) (s : Shape) (domain : ShapeContract s) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s b trees scratch output backwardKinds ks hs) (m : Memory ℝ)
    (bh : Nat) (hb : bh < s.B*s.H) (dx : Inputs ℝ) :
    HasDerivAt (fun a : ℝ => lossSpec s.T s.K s.V (selectedScale s.K s.useScale)
      (affineInputs (headInputs (currentInputs realOps s b m) bh) dx a)
      (headCotangent (currentInputs realOps s b m) bh))
      (returnedPairing s bh dx output (kernelRun realOps p s b trees scratch output ks hs m)) 0 := by
  have correct := (sourceCorrect p s domain b trees scratch output backwardKinds ks hs layout m).1
  have pairing : returnedPairing s bh dx output (kernelRun realOps p s b trees scratch output ks hs m)=
      (∑ t∈Finset.range s.T,∑ e∈Finset.range s.K,dx.q t e*logicalSpec s .query (currentInputs realOps s b m) bh t e)+
      (∑ t∈Finset.range s.T,∑ e∈Finset.range s.K,dx.k t e*logicalSpec s .key (currentInputs realOps s b m) bh t e)+
      (∑ t∈Finset.range s.T,∑ e∈Finset.range s.V,dx.v t e*logicalSpec s .value (currentInputs realOps s b m) bh t e) := by
    unfold returnedPairing
    congr 1
    · congr 1
      · apply Finset.sum_congr rfl
        intro t ht
        apply Finset.sum_congr rfl
        intro e he
        congr 1
        exact correct .query (by simp [backwardKinds]) bh t e hb (Finset.mem_range.mp ht) (Finset.mem_range.mp he)
      · apply Finset.sum_congr rfl
        intro t ht
        apply Finset.sum_congr rfl
        intro e he
        congr 1
        exact correct .key (by simp [backwardKinds]) bh t e hb (Finset.mem_range.mp ht) (Finset.mem_range.mp he)
    · apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro e he
      congr 1
      exact correct .value (by simp [backwardKinds]) bh t e hb (Finset.mem_range.mp ht) (Finset.mem_range.mp he)
  rw [pairing]
  exact independent_vjp s.T s.K s.V (selectedScale s.K s.useScale) _ dx _

/-- The normalization chain's cotangent tensors are the ordinary backward
inputs. With those values materialized, the actual returned memory is the VJP
of the normalized public function. This premise binds input values only. -/
theorem returned_normalized_vjp (p : Plans) (s : Shape) (domain : ShapeContract s) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s b trees scratch output backwardKinds ks hs) (m : Memory ℝ)
    (bh : Nat) (hb : bh < s.B*s.H) (dx : Inputs ℝ) (dy : Nat → Nat → ℝ) (eps : ℝ)
    (cotangent : headCotangent (currentInputs realOps s b m) bh=
      normalizedCotangent s.T s.K s.V (selectedScale s.K s.useScale) eps
        (headInputs (currentInputs realOps s b m) bh) dy)
    (nonzero : ∀ t,t < s.T → normalizerSpec s.T s.K (selectedScale s.K s.useScale)
      (headInputs (currentInputs realOps s b m) bh) t+eps ≠ 0) :
    HasDerivAt (fun a : ℝ => normalizedLoss s.T s.K s.V (selectedScale s.K s.useScale) eps
      (affineInputs (headInputs (currentInputs realOps s b m) bh) dx a) dy)
      (returnedPairing s bh dx output (kernelRun realOps p s b trees scratch output ks hs m)) 0 := by
  have h := returned_vjp p s domain b trees scratch output ks hs layout m bh hb dx
  rw [cotangent] at h
  have equal := h.unique (independent_vjp s.T s.K s.V (selectedScale s.K s.useScale)
    (headInputs (currentInputs realOps s b m) bh) dx
    (normalizedCotangent s.T s.K s.V (selectedScale s.K s.useScale) eps
      (headInputs (currentInputs realOps s b m) bh) dy))
  rw [equal]
  exact normalized_vjp s.T s.K s.V (selectedScale s.K s.useScale) eps _ dx dy nonzero

/-- Compiler-resolved live-accumulator execution has the same independent
mathematical outputs and complement frame under Real scalar interpretation. -/
theorem compiled_source_correct (c : CompilerLayout) (s : Shape) (domain : ShapeContract s) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (kinds : List OutputKind)
    (ks hs : List Event) (layout : RunLayout s b trees scratch output kinds ks hs) (m : Memory ℝ) :
    (∀ kind,kind∈kinds → ∀ bh t e,bh < s.B*s.H → t < s.T → e < kind.width s →
      compiledRun realOps c s b trees scratch output ks hs m (output kind+dense s.T (kind.width s) bh t e)=
        mathematicalResult s kind (currentInputs realOps s b m) bh t e) ∧
    (∀ address,(∀ ev∈ks,tensorAddress scratch ev ≠ address) →
      (∀ ev∈hs,tensorAddress output ev ≠ address) →
      compiledRun realOps c s b trees scratch output ks hs m address=m address) := by
  rw [compiledRun_real]
  exact sourceCorrect c.plans s domain b trees scratch output kinds ks hs layout m

#axiomsClean compiled_source_correct
#axiomsClean returned_normalized_vjp
#axiomsClean sourceCorrect
#axiomsClean public_forward
#axiomsClean returned_vjp
end
end VeriTile.Bench.Repairs.ParallelAttention
