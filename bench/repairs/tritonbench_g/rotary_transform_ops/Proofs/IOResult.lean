import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.IOFormula

namespace VeriTile.Bench.Repairs.RotaryTransformOps.IOResult
open VeriTile.Triton FP.Structural Interface
noncomputable section

theorem real_result (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (vals : Inputs ℝ c.shape) (ready : Ready c vals) (u : State ℝ)
    (hin : (io realOps c cu64 off64 id).InputBinding vals u) :
    ∃ final, runSchedule (Interface.schedule realOps c cu64 off64) u=some final ∧
      (io realOps c cu64 off64 id).OutputReadback vals (realReference c vals) final ∧
      (io realOps c cu64 off64 id).Frame vals u final := by
  obtain ⟨final,ran,values,frame⟩ := KernelEndpoints.real c valid u
    (IOFacts.metadata realOps c cu64 off64 id vals ready u hin)
    (IOFacts.suffix realOps c cu64 off64 id vals ready u hin) cu64 off64
  refine ⟨final,?_,?_,?_⟩
  · rw [IOFacts.schedule_eq]
    exact ran
  · intro port j mask
    fin_cases port
    exact (values j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
      j.1.isLt mask j.2.2.1.isLt j.2.2.2.1.isLt).trans
      (IOFacts.real_formula c valid cu64 off64 vals u hin _ _ _ _
        j.1.isLt mask j.2.2.1.isLt j.2.2.2.1.isLt)
  · intro r o outside
    exact frame r o (IOFacts.outside realOps c valid cu64 off64 id vals r o outside)

theorem native_result (c : Config) (valid : c.Valid) (p : Native.Instructions)
    (cu64 off64 : Bool) (vals : Inputs Native.Node c.shape) (ready : Ready c vals)
    (u : State Native.Node)
    (hin : (io (Native.ops p) c cu64 off64 observeNative).InputBinding vals u) :
    ∃ final, runSchedule (Interface.schedule (Native.ops p) c cu64 off64) u=some final ∧
      (io (Native.ops p) c cu64 off64 observeNative).OutputReadback vals (nativeReference c p vals) final ∧
      (io (Native.ops p) c cu64 off64 observeNative).Frame vals u final := by
  obtain ⟨final,ran,values,frame⟩ := KernelEndpoints.native c valid p u
    (IOFacts.metadata (Native.ops p) c cu64 off64 observeNative vals ready u hin)
    (IOFacts.suffix (Native.ops p) c cu64 off64 observeNative vals ready u hin) cu64 off64
  refine ⟨final,?_,?_,?_⟩
  · rw [IOFacts.schedule_eq]
    exact ran
  · intro port j mask
    fin_cases port
    exact congrArg Native.atom ((values j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
      j.1.isLt mask j.2.2.1.isLt j.2.2.2.1.isLt).trans
      (IOFacts.native_formula c valid p cu64 off64 vals u hin _ _ _ _
        j.1.isLt mask j.2.2.1.isLt j.2.2.2.1.isLt))
  · intro r o outside
    exact frame r o (IOFacts.outside (Native.ops p) c valid cu64 off64 observeNative vals r o outside)

end
end VeriTile.Bench.Repairs.RotaryTransformOps.IOResult
