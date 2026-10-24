import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.LayerNormTriton.KernelForward
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section
open scoped Classical

def row (api : Memory.API α Store Byte) (neg : α → α)
    (x : Input) (m : World Store) (N r : Nat) : Nat → α :=
  fun j => inputValue api neg x m (r*N+j)

def values (ops : Ops α) (p : KernelGraph.ForwardPlan N B) (api : Memory.API α Store Byte)
    (neg : α → α) (c : KernelContract.Forward M N) (eps : α) (m : World Store) :
    ∀ k : Fin 3,Fin (c.buffers.output k).count → α
  | ⟨0,_⟩,i => (KernelGraph.stats ops p (row api neg c.x m N i.val) eps 0).inverse
  | ⟨1,_⟩,i => (KernelGraph.stats ops p (row api neg c.x m N i.val) eps 0).mean
  | ⟨2,_⟩,i => KernelGraph.output ops p (row api neg c.x m N (i.val/N))
      (inputValue api neg c.w m) (inputValue api neg c.bias m) eps (i.val%N)

def final (ops : Ops α) (p : KernelGraph.ForwardPlan N B) (api : Memory.API α Store Byte)
    (neg : α → α) (c : KernelContract.Forward M N) (eps : α) (m : World Store) : World Store :=
  if KernelContract.launched M N then run c.buffers api (values ops p api neg c eps m) m else m

def exec (ops : Ops α) (p : KernelGraph.ForwardPlan N B) (api : Memory.API α Store Byte)
    (neg : α → α) (c : KernelContract.Forward M N) (eps : α) (m : World Store) : Except Unit (World Store) :=
  if KernelContract.launched M N then KernelMemory.exec c.buffers api (values ops p api neg c eps m) m else .ok m

def realValues (c : KernelContract.Forward M N) (eps : ℝ) (m : World (Nat → ℝ)) :
    ∀ k : Fin 3,Fin (c.buffers.output k).count → ℝ
  | ⟨0,_⟩,i => RealProof.Spec.inverse N (row Memory.realAPI Neg.neg c.x m N i.val) eps
  | ⟨1,_⟩,i => RealProof.Spec.mean N (row Memory.realAPI Neg.neg c.x m N i.val)
  | ⟨2,_⟩,i => RealProof.Spec.output N (row Memory.realAPI Neg.neg c.x m N (i.val/N))
      (inputValue Memory.realAPI Neg.neg c.w m) (inputValue Memory.realAPI Neg.neg c.bias m) eps (i.val%N)

theorem values_real (p : KernelGraph.ForwardPlan N B) (c : KernelContract.Forward M N)
    (eps : ℝ) (m : World (Nat → ℝ)) :
    values RealProof.ops p Memory.realAPI Neg.neg c eps m=realValues c eps m := by
  funext k i
  fin_cases k
  · exact congrArg (fun s : Reuse.Statistics ℝ => s.inverse) (KernelGraph.stats_real p _ eps 0)
  · exact congrArg (fun s : Reuse.Statistics ℝ => s.mean) (KernelGraph.stats_real p _ eps 0)
  · exact KernelGraph.output_real p _ _ _ eps _

/-- Complete constructive graph result, including exact empty dispatch and all
statistics/output stores. `exec` is the independent graph interpreter, not an
assumed execution of the Python/typed source. -/
theorem executes (ops : Ops α) (p : KernelGraph.ForwardPlan N B) (api : Memory.API α Store Byte)
    (neg : α → α) (c : KernelContract.Forward M N) (eps : α) (m : World Store) :
    ∃ result,exec ops p api neg c eps m=.ok result ∧
      result=final ops p api neg c eps m ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.buffers.output k).tensor i.val=values ops p api neg c eps m k i) ∧
      (KernelContract.launched M N=Bool.false → result=m) ∧
      (∀ key,Outside c.buffers key → observe api result key=observe api m key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe api result key=observe api m key) := by
  by_cases h : KernelContract.launched M N=Bool.true
  · obtain ⟨result,he,hm,hr,ho,hf⟩ := KernelMemory.executes c.buffers api neg (values ops p api neg c eps m) m
    refine ⟨result,by simpa only [exec,h,if_true] using he,?_,fun _ => hr,?_,ho,hf⟩
    · simpa only [final,h,if_true,run,Execution.Buffers.run,writes] using hm
    · intro hn; simp [h] at hn
  · have hf : KernelContract.launched M N=Bool.false := Bool.eq_false_iff.mpr h
    exact ⟨m,by simp [exec,hf],by simp [final,hf],fun ht => False.elim (h ht),
      fun _ => rfl,fun _ _ => rfl,fun _ _ => rfl⟩

/-- Read operands and ABI pointer are composed on the very same input Tensor;
this permits nonzero offsets, singleton strides and input aliases. -/
theorem row_input_binding (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (m : World Store) (r j : Nat) (hr : r<M) (hj : j<N) :
    row api neg c.x m N r j=logical api neg m c.x.dense.tensor (r*N+j) ∧
    row api neg c.x m N r j=api.read (m c.x.dense.owner) (c.x.metadata.dataPtr.toNat+4*(r*N+j)) := by
  have hi : r*N+j<c.x.dense.count := by
    rw [c.xCount]
    have hm := Nat.mul_le_mul_right N (show r+1≤M by omega)
    nlinarith
  exact ⟨inputValue_logical api neg c.x m _ hi,
    (inputValue_load api neg c.x m _).trans (congrArg (api.read (m c.x.dense.owner)) (input_pointer c.x _ hi))⟩

#axiomsClean values_real
#axiomsClean executes
#axiomsClean row_input_binding
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.KernelForward
