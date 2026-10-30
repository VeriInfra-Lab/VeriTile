import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Physical

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

namespace TensorEffect
variable {A : Type} [Inhabited A]

theorem output (ops : Ops A) (literal : ℝ → A) (zero : literal 0=ops.zero)
    (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (scale : ℝ) (storeFlag ifcond : Bool)
    (admit : InputContract c (t.arguments storeFlag ifcond) b.pointers)
    (before : Cells A) (rest : State (Term A)) (head row : Nat) (j : Fin c.D)
    (hh : head<c.heads) (hr : row<c.T) :
    logical (Physical.result ops literal c abi scale storeFlag ifcond before rest) t.output
      (c.mainOffset head row j.val)=
    ops.cast (Reference.output ops c (tensorInputs c abi before)
      (literal scale) head (row/32) ifcond (row%32) j.val) := by
  have live : ¬Host.Empty c := by unfold Host.Empty; omega
  have ptr := pointer_bases c abi t b storeFlag ifcond admit live
  rw [←physical_logical _ t.output abi.o.base _ ptr.o b.oContiguous
    (by rw [main_numel c t.output b.oShape]; exact c.main_bound _ _ _ hh hr j.isLt)]
  rw [Physical.output ops literal zero c abi scale storeFlag ifcond before rest head row j hh hr]
  rfl

theorem history (ops : Ops A) (literal : ℝ → A) (zero : literal 0=ops.zero)
    (c : Shape) (abi : ABI c) (t : Tensors) (b : TensorBinding c abi t)
    (scale : ℝ) (ifcond : Bool)
    (admit : InputContract c (t.arguments Bool.true ifcond) b.pointers)
    (before : Cells A) (rest : State (Term A)) (head chunk : Nat) (i j : Fin c.D)
    (hh : head<c.heads) (hc : chunk<c.chunks) :
    logical (Physical.result ops literal c abi scale Bool.true ifcond before rest) t.history
      (c.historyOffset head chunk i.val j.val)=
    ops.cast (Reference.carry ops c (tensorInputs c abi before)
      (literal scale) head ifcond chunk i.val j.val) := by
  have ht : c.T≠0 := by intro h; have := c.chunks_zero.mpr h; omega
  have live : ¬Host.Empty c := by unfold Host.Empty; omega
  have ptr := pointer_bases c abi t b Bool.true ifcond admit live
  rw [←physical_logical _ t.history abi.h.base _ ptr.h b.hContiguous
    (by rw [history_numel c t.history b.hShape]; exact c.history_bound _ _ _ _ hh hc i.isLt j.isLt)]
  rw [Physical.history ops literal zero c abi scale ifcond before rest head chunk i j hh hc]
  rfl

theorem input_frame (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) (a : Allocation) (ha : a∈[abi.q,abi.k,abi.v])
    (address : Nat) (inside : a.Contains (address/4)) :
    Physical.result ops literal c abi scale storeFlag ifcond before rest address=before address := by
  apply Physical.frame
  · intro ho
    apply abi.outputSeparate a (by simp only [List.mem_cons,List.not_mem_nil,or_false] at *; tauto) _ ho.2 inside
  · right
    intro hh
    apply abi.historySeparate a (by simp only [List.mem_cons,List.not_mem_nil,or_false] at *; tauto) _ hh.2 inside

theorem history_disabled (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) (address : Nat)
    (inside : abi.h.Contains (address/4)) :
    Physical.result ops literal c abi scale Bool.false ifcond before rest address=before address := by
  apply Physical.frame
  · intro ho
    exact abi.outputSeparate abi.h (by simp) _ ho.2 inside
  · exact Or.inl rfl

/-- The public kernel endpoint retains the actual output Tensor object. -/
def publicReturn (t : Tensors) : VeriTile.Torch.FP32.Tensor := t.output.tensor
@[simp] theorem output_identity (t : Tensors) : (publicReturn t).object=t.output.tensor.object := rfl

#axiomsClean output
#axiomsClean history
#axiomsClean input_frame
#axiomsClean history_disabled
end TensorEffect
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
