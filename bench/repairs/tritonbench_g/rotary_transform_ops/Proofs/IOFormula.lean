import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.IOFacts

namespace VeriTile.Bench.Repairs.RotaryTransformOps.IOFacts
open VeriTile.Triton FP.Structural Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem pair_values (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (observe : α → α) (vals : Inputs α c.shape) (u : State α)
    (hin : (io p c cu64 off64 observe).InputBinding vals u)
    (w : Work) (hw : c.Active w) :
    pairValues c p w (KernelAlgebra.memory u)=pairValues c p w (Interface.inputMemory vals) := by
  have low := input_value p c cu64 off64 observe vals u hin _ (cell_bound c valid w hw Bool.false)
  have high := input_value p c cu64 off64 observe vals u hin _ (cell_bound c valid w hw Bool.true)
  simp only [pairValues,low,high]
  by_cases cached : c.hasCoeff w
  · simp only [if_pos cached,
      cosine_value p c cu64 off64 observe vals u hin _ (coeff_bound c w hw cached),
      sine_value p c cu64 off64 observe vals u hin _ (coeff_bound c w hw cached)]
  · simp only [if_neg cached]

theorem real_formula (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (vals : Inputs ℝ c.shape) (u : State ℝ)
    (hin : (io realOps c cu64 off64 id).InputBinding vals u)
    (batch row head col : Nat) (hb : batch<c.shape.batches) (hr : row<c.length batch)
    (hh : head<c.shape.heads) (hd : col<c.shape.width) :
    realFormula c (KernelAlgebra.memory u) batch row head col=
      realFormula c (Interface.inputMemory vals) batch row head col := by
  by_cases rotary : col<2*c.shape.half
  · obtain ⟨hp,back⟩ := column_inverse c.shape col rotary
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have hw : c.Active w := ⟨hb,hr,hh,hp⟩
    have equal := congrArg (fun pair : ℝ×ℝ => component pair (c.shape.second col))
      (pair_values realOps c valid cu64 off64 id vals u hin w hw)
    dsimp only at equal
    rw [real_pair_formula c _ w hw,real_pair_formula c _ w hw] at equal
    simpa only [w,back] using equal
  · simp only [realFormula,if_neg rotary]
    exact input_value realOps c cu64 off64 id vals u hin _
      (Host.address_bound c.shape _ head col (row_bound c valid batch row hb hr) hh hd)

theorem native_formula (c : Config) (valid : c.Valid) (p : Native.Instructions)
    (cu64 off64 : Bool) (vals : Inputs Native.Node c.shape) (u : State Native.Node)
    (hin : (io (Native.ops p) c cu64 off64 observeNative).InputBinding vals u)
    (batch row head col : Nat) (hb : batch<c.shape.batches) (hr : row<c.length batch)
    (hh : head<c.shape.heads) (hd : col<c.shape.width) :
    nativeFormula c p (Native.project (KernelAlgebra.memory u)) batch row head col=
      nativeFormula c p (Native.project (Interface.inputMemory vals)) batch row head col := by
  by_cases rotary : col<2*c.shape.half
  · obtain ⟨hp,back⟩ := column_inverse c.shape col rotary
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have hw : c.Active w := ⟨hb,hr,hh,hp⟩
    have equal := congrArg (fun pair : Native.Node×Native.Node => component pair (c.shape.second col))
      (pair_values (Native.ops p) c valid cu64 off64 observeNative vals u hin w hw)
    dsimp only at equal
    rw [Native.pairWord_source,Native.pairWord_source,
      native_pair_formula c p _ w hw,native_pair_formula c p _ w hw] at equal
    simpa only [Native.atom,w,back,Native.Node.mk.injEq,and_true] using equal
  · simp only [nativeFormula,if_neg rotary,Native.project]
    exact congrArg Native.Node.value
      (input_value (Native.ops p) c cu64 off64 observeNative vals u hin _
        (Host.address_bound c.shape _ head col (row_bound c valid batch row hb hr) hh hd))

end
end VeriTile.Bench.Repairs.RotaryTransformOps.IOFacts
