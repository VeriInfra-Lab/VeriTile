import bench.repairs.tritonbench_g.parallel_attention.Kernels
import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeExecution
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

abbrev FP32Word := VeriTile.Triton.FP.Value .fp32

/-- Only scalar constants/arithmetic/casts/scale conversion remain numerical
ports. Live dot accumulators, dZ seeds, compiler FMA contractions and reduction trees are explicit finite syntax with proved coverage;
loops, masks, stores, shard reductions and final memory are never ports. -/
theorem fp32_public_words (native reference : Ops FP32Word)
    (primitives : ScalarAgreement native reference) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (kinds : List OutputKind)
    (ks hs : List Event) (layout : RunLayout s b trees scratch output kinds ks hs) (m : Memory FP32Word) :
    (∀ kind,kind∈kinds → ∀ i,i < kind.cells s →
      compiledRun native c s b trees scratch output ks hs m (output kind+i)=
        nativeWordReference reference c s kind (trees kind) (currentInputs reference s b m) i) ∧
    (∀ address,(∀ ev∈ks,tensorAddress scratch ev ≠ address) →
      (∀ ev∈hs,tensorAddress output ev ≠ address) →
      compiledRun native c s b trees scratch output ks hs m address=m address) := by
  rw [primitives.eq]
  constructor
  · intro kind hk i hi
    exact compiled_output reference c s b trees scratch output kinds ks hs
      layout.kernelCoverage layout.hostCoverage layout.kernelFresh layout.kernelUnique
      layout.hostFresh layout.hostUnique m kind hk i hi
  · exact compiled_frame reference c s b trees scratch output ks hs m

/-- Bitwise scalar correspondence lifts through the whole source graph and
public normalization memory. This is not an FP=Real theorem. -/
theorem fp32_backend_equiv (native reference : Ops FP32Word)
    (primitives : ScalarAgreement native reference) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory FP32Word) (eps : FP32Word) (o z dst bh : Nat)
    (returnBoth normalize : Bool) :
    compiledRun native c s b trees scratch output ks hs m=
      compiledRun reference c s b trees scratch output ks hs m ∧
    normalizeRun native eps (s.B*s.H) s.T s.V o z dst m=
      normalizeRun reference eps (s.B*s.H) s.T s.V o z dst m ∧
    publicRead native eps s o z m bh returnBoth normalize=
      publicRead reference eps s o z m bh returnBoth normalize := by
  rw [primitives.eq]
  exact ⟨rfl,rfl,rfl⟩

#axiomsClean fp32_public_words
#axiomsClean fp32_backend_equiv
end
end VeriTile.Bench.Repairs.ParallelAttention
