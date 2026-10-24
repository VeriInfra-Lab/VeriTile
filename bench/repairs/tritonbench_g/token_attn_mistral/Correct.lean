import bench.repairs.tritonbench_g.token_attn_mistral.Proofs.Checked
import bench.repairs.tritonbench_g.token_attn_mistral.Proofs.Mathematical
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 4000000

/-- Complete checked call: independent gathered weighted sum, every output,
None return, supplied output identity, active storage bounds, padded integer
bounds and the full current-memory frame. Floating rounding is erased only here. -/
theorem correct (c : Call) (raw : SignedMetadata) (m : Memory ℝ) (flags : Flags)
    (accepted : check c raw m flags=Bool.true) (binding : SignedBinding c raw m)
    (work : List PID) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ Formed c m ∧ ∃ result,
      invokeActual Real.algebra Real.sum c work m=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        (result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)).read .fp32=
          target c m b.val h.val d.val) ∧ Frame c m result.memory := by
  have a := (check_iff c raw m flags).mp accepted
  refine ⟨accesses c m a.layout a.data a.storage,formed c m a.data a.integers,?_⟩
  rw [invokeActual_eq _ _ _ _ _ a.layout legal]
  exact whole_real c m a.layout

/-- Successful checked empty and launch branches share the public endpoint. -/
theorem checked_call (c : Call) (raw : SignedMetadata) (m : Memory ℝ) (flags : Flags)
    (launch : Bool) (accepted : validate c raw m flags=.ok launch) (binding : SignedBinding c raw m)
    (work : List PID) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    ∃ result, dispatch Real.algebra Real.sum c raw m flags work m=.ok (some result) ∧
      result.returned=none ∧ result.output=c.output ∧
      ((launch=Bool.false ∧ result.memory=m) ∨
       (launch=Bool.true ∧ Accesses c m ∧ Formed c m ∧
        (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
          (result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)).read .fp32=
            target c m b.val h.val d.val) ∧ Frame c m result.memory)) := by
  cases launch
  · exact ⟨⟨none,c.output,m⟩,dispatch_empty _ _ _ _ _ _ _ _ accepted,rfl,rfl,Or.inl ⟨rfl,rfl⟩⟩
  · have a := ((validate_launch c raw m flags).mp accepted).2
    obtain ⟨access,formed,result,run,ret,out,read,frame⟩ :=
      correct c raw m flags ((check_iff c raw m flags).mpr a) binding work legal
    refine ⟨result,?_,ret,out,Or.inr ⟨rfl,access,formed,read,frame⟩⟩
    rw [dispatch_launch _ _ _ _ _ _ _ _ accepted]
    exact congrArg Except.ok run

#axiomsClean checked_call
#axiomsClean correct
#auditStmt correct
#specNonCircular target avoiding [FP.Structural.exec,Grid.run,invoke,invokeActual]
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
