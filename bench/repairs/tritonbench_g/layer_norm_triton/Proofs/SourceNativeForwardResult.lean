import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeStorage

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeForwardResult
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceNative.Native SourceNativeStorage SourceNativeStorage.Forward SourceStrideRefinement
noncomputable section

theorem executes (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (hb : N≤B) :
    ∃ result,exec p B eps api neg c memory view=some result ∧
      result=final p B eps api neg c memory ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.buffers.output k).tensor i.val=values p (graphRow p B eps api neg c memory) c k i) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (∀ key,Outside c.buffers key → observe api result key=observe api memory key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe api result key=observe api memory key) := by
  refine ⟨final p B eps api neg c memory,exec_eq p B eps api neg c memory view hb,rfl,?_,?_,?_,?_⟩
  · intro hl k i
    simp only [final,hl,if_true]
    exact KernelMemory.run_read c.buffers api neg _ memory k i
  · intro hl; simp [final,hl]
  · intro key hk
    by_cases hl : KernelContract.launched M N=Bool.true
    · simp only [final,hl,if_true]
      exact KernelMemory.run_outside c.buffers api _ memory key hk
    · simp [final,hl]
  · intro key hk
    by_cases hl : KernelContract.launched M N=Bool.true
    · simp only [final,hl,if_true]
      exact KernelMemory.run_frame c.buffers api _ memory key hk
    · simp [final,hl]

/-- Source and the unrolled candidate generate the same complete symbolic
state. Consequently every native observation agrees, including FMA policies. -/
theorem candidate (p : Policy) (stride N B : Nat) (eps : ℝ) (s : VeriTile.Triton.FP.Structural.State Word)
    (hn : 0<N) (hb : N≤B) :
    VeriTile.Triton.FP.Structural.exec (SourceModel.engine (SourceNative.model p.reductions))
      (SourceCandidate.forward stride N B eps) (SourceNative.seed s)=
    VeriTile.Triton.FP.Structural.exec (SourceModel.engine (SourceNative.model p.reductions))
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) (SourceNative.seed s) :=
  SourceCandidate.forward_equiv _ stride N B eps hn hb _

#axiomsClean executes
#axiomsClean candidate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeForwardResult
