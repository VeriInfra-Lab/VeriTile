import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeForwardResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeResult

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeComposition
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceNative.Native SourceNativeStorage SourceStrideRefinement
noncomputable section

theorem saved (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (r mu : Input)
    (rBinding : r.dense=c.buffers.output 0) (muBinding : mu.dense=c.buffers.output 1)
    (launch : KernelContract.launched M N=Bool.true) (row : Nat) (hr : row<M) :
    KernelBackward.initialStats api neg r mu (Forward.final p B eps api neg c memory) row=
      ⟨p.read (Forward.graphRow p B eps api neg c memory row) "Mean" row,
       p.read (Forward.graphRow p B eps api neg c memory row) "Rstd" row⟩ := by
  have ri : row<r.dense.count := by rw [rBinding,c.inverseCount]; exact hr
  have mi : row<mu.dense.count := by rw [muBinding,c.meanCount]; exact hr
  have ir : row<(c.buffers.output 0).count := by rw [c.inverseCount]; exact hr
  have im : row<(c.buffers.output 1).count := by rw [c.meanCount]; exact hr
  have rv := KernelMemory.run_read c.buffers api neg
    (Forward.values p (Forward.graphRow p B eps api neg c memory) c) memory 0 ⟨row,ir⟩
  have mv := KernelMemory.run_read c.buffers api neg
    (Forward.values p (Forward.graphRow p B eps api neg c memory) c) memory 1 ⟨row,im⟩
  unfold KernelBackward.initialStats
  rw [inputValue_logical api neg mu _ row mi,inputValue_logical api neg r _ row ri]
  simp only [rBinding,muBinding,Forward.final,launch,if_true]
  rw [rv,mv]
  rfl

/-- Sequential native source execution, with actual physical saved statistics
and scratch words reloaded as fresh leaves at each kernel boundary. -/
theorem executes (p q z : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (f : KernelContract.Forward M N) (dy r mu : Input) (b : KernelContract.Backward M N f.x f.w dy r mu)
    (rBinding : r.dense=f.buffers.output 0) (muBinding : mu.dense=f.buffers.output 1)
    (memory : World Store) (view : MatrixView f.x M N) (hb : N≤B) :
    ∃ forward backward,
      Forward.exec p B eps api neg f memory view=some forward ∧
      Backward.exec q z B api neg b forward view=some backward ∧
      forward=Forward.final p B eps api neg f memory ∧
      backward=Backward.final q z B api neg b forward ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg forward (f.buffers.output k).tensor i.val=Forward.values p (Forward.graphRow p B eps api neg f memory) f k i) ∧
      (∀ k i,logical api neg backward (KernelBackward.outputs b k).tensor i.val=
        if KernelContract.launched M N then SourceNativeResult.values q z B api neg b forward k i
        else logical api neg forward (KernelBackward.outputs b k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.true → ∀ row,row<M →
        KernelBackward.initialStats api neg r mu forward row=
          ⟨p.read (Forward.graphRow p B eps api neg f memory row) "Mean" row,
           p.read (Forward.graphRow p B eps api neg f memory row) "Rstd" row⟩) ∧
      (KernelContract.launched M N=Bool.false → forward=memory ∧ backward=memory) ∧
      (∀ key,Outside f.buffers key → KernelBackward.Outside b key → observe api backward key=observe api memory key) := by
  obtain ⟨forward,hfe,hfm,hfv,hfz,hfo,hff⟩ := SourceNativeForwardResult.executes p B eps api neg f memory view hb
  obtain ⟨backward,hbe,hbm,hbv,hbz,hbs,hbo,hbf⟩ := SourceNativeResult.executes q z B api neg b forward view
  refine ⟨forward,backward,hfe,hbe,hfm,hbm,hfv,hbv,?_,?_,?_⟩
  · intro hl row hr
    rw [hfm]
    exact saved p B eps api neg f memory r mu rBinding muBinding hl row hr
  · intro hl
    exact ⟨hfz hl,(hbz hl).trans (hfz hl)⟩
  · intro key hf hb
    exact (hbo key hb).trans (hfo key hf)

#axiomsClean saved
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeComposition
