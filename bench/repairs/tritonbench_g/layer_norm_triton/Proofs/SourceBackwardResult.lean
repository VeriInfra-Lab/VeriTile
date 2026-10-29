import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardStorage

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardResult
open VeriTile.Triton SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory SourceBackwardStorage
noncomputable section
variable {x w dy r mu : Input}

def values [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) :
    ∀ k : Fin 3,Fin (KernelBackward.outputs c k).count → α
  | ⟨0,_⟩,i => groupValues (graphGroup model B api neg c memory) c 0 i
  | ⟨1,_⟩,i => finalValues (graphFinal model api c (afterGroup model B api neg c memory)) c 0 i
  | ⟨2,_⟩,i => finalValues (graphFinal model api c (afterGroup model B api neg c memory)) c 1 i

theorem run_first_read [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store)
    (k : Fin 3) (i : Fin (c.first.buffers.output k).count) :
    logical api neg (SourceBackwardStorage.run model B api neg c memory) (c.first.buffers.output k).tensor i.val=
      groupValues (graphGroup model B api neg c memory) c k i := by
  let gv := groupValues (graphGroup model B api neg c memory) c
  let after := afterGroup model B api neg c memory
  have owned : (c.first.buffers.output k).owner=c.last.buffers.owner :=
    (c.first.buffers.owned k).trans ((c.first.buffers.owned 0).symm.trans c.sameOwner)
  rw [dense_logical api neg _ _ _ i.isLt]
  apply (Execution.run_input c.last.buffers api (finalValues (graphFinal model api c after) c)
    (schedule c.last.buffers) after (c.first.buffers.output k) owned (fun j => c.preserveScratch j k) i).trans
  rw [c.first.buffers.owned k]
  exact Execution.run_readback c.first.buffers api gv (schedule c.first.buffers)
    (schedule_covered c.first.buffers) memory k i

theorem run_read [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store)
    (k : Fin 3) (i : Fin (KernelBackward.outputs c k).count) :
    logical api neg (SourceBackwardStorage.run model B api neg c memory) (KernelBackward.outputs c k).tensor i.val=
      values model B api neg c memory k i := by
  fin_cases k
  · exact run_first_read model B api neg c memory 0 i
  · exact KernelMemory.run_read c.last.buffers api neg _ (afterGroup model B api neg c memory) 0 i
  · exact KernelMemory.run_read c.last.buffers api neg _ (afterGroup model B api neg c memory) 1 i

theorem run_outside [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store)
    (key : Key) (h : KernelBackward.Outside c key) :
    observe api (SourceBackwardStorage.run model B api neg c memory) key=observe api memory key :=
  (KernelMemory.run_outside c.last.buffers api _ _ key h.2).trans
    (KernelMemory.run_outside c.first.buffers api _ memory key h.1)

theorem run_frame [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store)
    (key : Key) (h : caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key) :
    observe api (SourceBackwardStorage.run model B api neg c memory) key=observe api memory key :=
  (KernelMemory.run_frame c.last.buffers api _ _ key h).trans
    (KernelMemory.run_frame c.first.buffers api _ memory key h)

theorem executes [Inhabited α] (model : Model α) (B : Nat) (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (memory : World Store) :
    ∃ result,SourceBackwardStorage.exec model B api neg c memory=some result ∧
      result=SourceBackwardStorage.final model B api neg c memory ∧
      (∀ k i,logical api neg result (KernelBackward.outputs c k).tensor i.val=
        if KernelContract.launched M N then values model B api neg c memory k i
        else logical api neg memory (KernelBackward.outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.first.buffers.output k).tensor i.val=groupValues (graphGroup model B api neg c memory) c k i) ∧
      (∀ key,KernelBackward.Outside c key → observe api result key=observe api memory key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe api result key=observe api memory key) := by
  refine ⟨SourceBackwardStorage.final model B api neg c memory,exec_eq model B api neg c memory,rfl,?_,?_,?_,?_,?_⟩
  · intro k i
    by_cases h : KernelContract.launched M N=Bool.true
    · simp only [SourceBackwardStorage.final,h,if_true]
      exact run_read model B api neg c memory k i
    · simp [SourceBackwardStorage.final,h]
  · intro h; simp [SourceBackwardStorage.final,h]
  · intro h k i
    simp only [SourceBackwardStorage.final,h,if_true]
    exact run_first_read model B api neg c memory k i
  · intro key hk
    by_cases h : KernelContract.launched M N=Bool.true
    · simp only [SourceBackwardStorage.final,h,if_true]
      exact run_outside model B api neg c memory key hk
    · simp [SourceBackwardStorage.final,h]
  · intro key hk
    by_cases h : KernelContract.launched M N=Bool.true
    · simp only [SourceBackwardStorage.final,h,if_true]
      exact run_frame model B api neg c memory key hk
    · simp [SourceBackwardStorage.final,h]

#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceBackwardResult
