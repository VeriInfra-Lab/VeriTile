import bench.tritonbench_g.pow_scalar_tensor.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.FlatIO
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Transport any proved regional real output and full cell frame through a
safe disjoint allocation. The input cell types are unrestricted. -/
theorem transport {α : Type} (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (k : Kernel) (hok : k.FlattenOk)
    (s : BlockState) (hu : s.undef = (fun _ _ => 0)) (hsafe : k.TraceSafe A.extent s)
    (Out : RegionName) (hout : Out ∈ A.regions) (write : α → Nat) (active : α → Prop)
    (expected : α → ℝ) (hb : ∀ i, active i → write i < A.extent Out)
    (hrun : ∃ t, exec k s = some t ∧ (∀ i, active i → t.readMem Out (write i) = expected i) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i, active i → o ≠ write i) → t.mem r o = s.mem r o)) :
    ∃ t, exec (A.flattenKernel k) (A.flattenState s) = some t ∧
      (∀ i, active i → t.readMem A.flat (A.addr Out (write i)) = expected i) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i, active i → o ≠ A.addr Out (write i)) →
        t.mem r o = (A.flattenState s).mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := hrun
  refine ⟨A.flattenState t, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hcov k s hsafe hok hu, he, Option.map_some]
  · intro i hi
    have h := A.flattenState_readMemAs hd t hout (hb i hi) .real
    change (A.flattenState t).readMemAs .real A.flat (A.addr Out (write i)) =
      t.readMemAs .real Out (write i) at h
    rw [BlockState.readMemAs_real, BlockState.readMemAs_real] at h
    exact (Option.some.inj h).trans (hv i hi)
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
        obtain ⟨r, o⟩ := p
        obtain ⟨_, hoeq, _⟩ := A.decode_sound hdec
        show A.trCell (t.mem r o) = A.trCell (s.mem r o)
        refine congrArg A.trCell (hf r o ?_)
        by_cases hro : r = Out
        · subst hro
          refine Or.inr fun i hi => ?_
          rcases hcond with hflat | hn
          · exact absurd rfl hflat
          · intro heq
            exact hn i hi (by rw [hoeq, heq])
        · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.FlatIO
