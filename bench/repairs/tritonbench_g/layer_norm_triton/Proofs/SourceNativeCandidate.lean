import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeForwardResult

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeCandidate
open VeriTile.Triton FP.Structural SourceModel SourceNative SourceNative.Native
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceStrideRefinement SourceNativeStorage
noncomputable section
open scoped Classical

def runRow (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (row : Nat) : Option (State Term) :=
  FP.Structural.exec (engine (model p.reductions)) (SourceCandidate.forward view.stride N B eps)
    (seed (Forward.initial api neg c memory row))

def resultRow (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (row : Nat) : State Term :=
  (runRow p B eps api neg c memory view row).getD (seed (Forward.initial api neg c memory row))

def exec (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) : Option (World Store) :=
  if KernelContract.launched M N then
    if ∀ row,row<M → (runRow p B eps api neg c memory view row).isSome then
      some (KernelMemory.run c.buffers api (Forward.values p (resultRow p B eps api neg c memory view) c) memory)
    else none
  else some memory

theorem row_eq (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N)
    (hn : 0<N) (hb : N≤B) (row : Nat) :
    runRow p B eps api neg c memory view row=Forward.runRow p B eps api neg c memory view row :=
  SourceNativeForwardResult.candidate p view.stride N B eps (Forward.initial api neg c memory row) hn hb

theorem exec_eq_source (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (hb : N≤B) :
    exec p B eps api neg c memory view=Forward.exec p B eps api neg c memory view := by
  unfold exec Forward.exec
  split
  · rename_i h
    have hn := ((KernelContract.launched_iff M N).mp h).2
    have states : resultRow p B eps api neg c memory view=Forward.sourceRow p B eps api neg c memory view := by
      funext row
      simp only [resultRow,Forward.sourceRow,row_eq p B eps api neg c memory view hn hb]
    simp only [row_eq p B eps api neg c memory view hn hb,states]
  · rfl

/-- Entire candidate and source launches terminate with identical physical
memory. This includes saved statistics, output words, and every exterior byte. -/
theorem executes (p : Policy) (B : Nat) (eps : ℝ) (api : Memory.API Word Store Byte) (neg : Word → Word)
    (c : KernelContract.Forward M N) (memory : World Store) (view : MatrixView c.x M N) (hb : N≤B) :
    ∃ result,Forward.exec p B eps api neg c memory view=some result ∧
      exec p B eps api neg c memory view=some result ∧ result=Forward.final p B eps api neg c memory := by
  refine ⟨Forward.final p B eps api neg c memory,Forward.exec_eq p B eps api neg c memory view hb,?_,rfl⟩
  rw [exec_eq_source p B eps api neg c memory view hb]
  exact Forward.exec_eq p B eps api neg c memory view hb

#axiomsClean row_eq
#axiomsClean exec_eq_source
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeCandidate
