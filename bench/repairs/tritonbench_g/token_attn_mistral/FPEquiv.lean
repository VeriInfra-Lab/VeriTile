import bench.repairs.tritonbench_g.token_attn_mistral.Proofs.Checked
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent initial-word target. All padded products and 128-lane tree
nodes, loop additions, admissible FMA contractions and the output cast remain
present. A reduction plan describes ordered scalar nodes; it cannot supply a
floating result. No real arithmetic or whole reduction oracle is assumed. -/
def roundedTarget (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (c : Call) (m : Memory Native.Word) (b h d : Nat) (hd : d<c.value.width) :=
  Native.eval I S (values Native.algebra (Native.reduction R) (params c) (Native.entry m)
    ((b,h),(⟨d,hd⟩,PUnit.unit)))

theorem checked_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (c : Call) (raw : SignedMetadata) (m : Memory Native.Word) (flags : Flags)
    (accepted : check c raw m flags=Bool.true) (binding : SignedBinding c raw m)
    (work : List PID) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ Formed c m ∧ ∃ result,
      invokeActual Native.algebra (Native.reduction R) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        Native.observe I S result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)=
          roundedTarget I R S c m b.val h.val d.val d.isLt) ∧
      Frame c (Native.entry m) result.memory := by
  have a := (check_iff c raw m flags).mp accepted
  refine ⟨accesses c m a.layout a.data a.storage,formed c m a.data a.integers,?_⟩
  rw [invokeActual_eq _ _ _ _ _ a.layout legal]
  exact whole_native I R S c m a.layout

/-- Only rounded scalar add/mul/FMA instruction agreement is deferred.
Reduction trees, masks, metadata gathers, loop order and stores are proved. -/
theorem fpequiv (actual standard : Native.Instructions)
    (agreement : Native.InstructionAgreement actual standard)
    (R : Native.Reductions) (S : Native.Schedule)
    (c : Call) (raw : SignedMetadata) (m : Memory Native.Word) (flags : Flags)
    (accepted : check c raw m flags=Bool.true) (binding : SignedBinding c raw m)
    (work : List PID) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ Formed c m ∧ ∃ result,
      invokeActual Native.algebra (Native.reduction R) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        Native.observe actual S result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)=
          roundedTarget standard R S c m b.val h.val d.val d.isLt) ∧
      Frame c (Native.entry m) result.memory := by
  rw [Native.instruction_equality actual standard agreement]
  exact checked_native standard R S c raw m flags accepted binding work legal

/-- Full FP wrapper dispatch, including a successful byte-preserving empty return. -/
theorem checked_fp_call (actual standard : Native.Instructions)
    (agreement : Native.InstructionAgreement actual standard)
    (R : Native.Reductions) (S : Native.Schedule)
    (c : Call) (raw : SignedMetadata) (m : Memory Native.Word) (flags : Flags)
    (launch : Bool) (accepted : validate c raw m flags=.ok launch) (binding : SignedBinding c raw m)
    (work : List PID) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    ∃ result, dispatch Native.algebra (Native.reduction R) c raw m flags work (Native.entry m)=.ok (some result) ∧
      result.returned=none ∧ result.output=c.output ∧
      ((launch=Bool.false ∧ result.memory=Native.entry m) ∨
       (launch=Bool.true ∧ Accesses c m ∧ Formed c m ∧
        (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
          Native.observe actual S result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)=
            roundedTarget standard R S c m b.val h.val d.val d.isLt) ∧ Frame c (Native.entry m) result.memory)) := by
  cases launch
  · exact ⟨⟨none,c.output,Native.entry m⟩,dispatch_empty _ _ _ _ _ _ _ _ accepted,rfl,rfl,Or.inl ⟨rfl,rfl⟩⟩
  · have a := ((validate_launch c raw m flags).mp accepted).2
    obtain ⟨access,formed,result,run,ret,out,read,frame⟩ :=
      fpequiv actual standard agreement R S c raw m flags ((check_iff c raw m flags).mpr a) binding work legal
    refine ⟨result,?_,ret,out,Or.inr ⟨rfl,access,formed,read,frame⟩⟩
    rw [dispatch_launch _ _ _ _ _ _ _ _ accepted]
    exact congrArg Except.ok run

#axiomsClean checked_fp_call
#axiomsClean checked_native
#axiomsClean fpequiv
#auditStmt checked_native
#auditStmt fpequiv
#specNonCircular roundedTarget avoiding [FP.Structural.exec,Grid.run,invoke,invokeActual]
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
