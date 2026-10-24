import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeExecution
import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

theorem nativeScratchValue_real (c : CompilerLayout) (s : Shape) (kind : OutputKind)
    (x : TensorInputs ℝ) (i : Nat) :
    nativeScratchValue realOps c s kind x i=scratchValue realOps c.plans s kind x i := by
  unfold nativeScratchValue scratchValue
  cases kind <;> dsimp only
  · rw [(native_graph_real c s x _ _ 0 _).1]
  · rw [(native_graph_real c s x _ _ 0 0).1]
  · exact (native_graph_real c s x _ _ _ 0).2.1
  · rw [(native_graph_real c s x _ _ _ 0).2.2]
  · rw [(native_graph_real c s x _ _ 0 _).2.2]

theorem nativeWordReference_real (c : CompilerLayout) (s : Shape) (kind : OutputKind)
    (tree : SumPlan) (x : TensorInputs ℝ) (i : Nat) :
    nativeWordReference realOps c s kind tree x i=wordReference realOps c.plans s kind tree x i := by
  simp only [nativeWordReference,wordReference,nativeScratchValue_real]

theorem compiledRun_real (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory ℝ) :
    compiledRun realOps c s b trees scratch output ks hs m=
      combinedRun realOps c.plans s b trees scratch output ks hs m := by
  simp only [compiledRun,compiledScratch,combinedRun,combinedScratch,scratchCurrent,nativeScratchValue_real]


end
end VeriTile.Bench.Repairs.ParallelAttention
