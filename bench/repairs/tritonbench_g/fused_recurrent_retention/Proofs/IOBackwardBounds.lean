import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.IOGeometry

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

theorem backward_bounds (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (xs : (io .backward a s p incoming finalGradient seed final).Inputs)
    (bounds : RegionName → Nat) (hb : (io .backward a s p incoming finalGradient seed final).InBounds bounds xs)
    (pid : ProgramId) (hp : pid.Valid s) :
    SourceSafety.BackwardBounds s (KernelInterface.buffers s incoming finalGradient) pid seed final bounds := by
  constructor
  · intro t ht lane _ hl
    have h := hb.2.1 1 (⟨pid.head,hp.2.2⟩,⟨t,ht⟩,⟨pid.key s lane,hl⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DK pid.head t (pid.key s lane)<bounds "k" at h
    exact key_window s incoming finalGradient pid t lane ht (by omega) ▸ h
  · intro t ht lane _ hl
    have h := hb.2.1 2 (⟨pid.head,hp.2.2⟩,⟨t,ht⟩,⟨pid.value s lane,hl⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DV pid.head t (pid.value s lane)<bounds "v" at h
    exact value_window s incoming finalGradient pid t lane ht (by omega) ▸ h
  · intro t ht lane _ hl
    have h := hb.2.1 3 (⟨pid.head,hp.2.2⟩,⟨t,ht⟩,⟨pid.value s lane,hl⟩,PUnit.unit) (by simp [inputActive])
    change (pid.head/s.H)*incoming 0+(pid.head%s.H)*incoming 1+pid.value s lane*incoming 3+t*incoming 2<bounds "do" at h
    simpa [doOffset,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,ProgramId.value] using h
  · intro t ht lane _ hl
    have hj := partialOffset_bound s s.NV s.DK pid.iv pid.head t (pid.key s lane) hp.1 hp.2.2 ht hl
    have h := hb.2.2.1 1 (⟨_,hj⟩,PUnit.unit) (by simp [outputActive])
    change partialOffset s s.DK pid.iv pid.head t (pid.key s lane)<bounds "dq" at h
    exact gradient_window s incoming finalGradient pid t lane ht (by omega) ▸ h
  · intro t ht lane _ hl
    rw [reverseKey_at _ _ _ _ ht]
    have timeBound : s.T-1-t<s.T := by omega
    have h := hb.2.1 0 (⟨pid.head,hp.2.2⟩,⟨(s.T-1-t),timeBound⟩,⟨pid.key s lane,hl⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DK pid.head (s.T-1-t) (pid.key s lane)<bounds "q" at h
    exact key_window s incoming finalGradient pid (s.T-1-t) lane timeBound (by omega) ▸ h
  · intro t ht lane _ hl
    rw [reverseKey_at _ _ _ _ ht]
    have timeBound : s.T-1-t<s.T := by omega
    have h := hb.2.1 1 (⟨pid.head,hp.2.2⟩,⟨(s.T-1-t),timeBound⟩,⟨pid.key s lane,hl⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DK pid.head (s.T-1-t) (pid.key s lane)<bounds "k" at h
    exact key_window s incoming finalGradient pid (s.T-1-t) lane timeBound (by omega) ▸ h
  · intro t ht lane _ hl
    rw [reverseValue_at _ _ _ _ ht]
    have timeBound : s.T-1-t<s.T := by omega
    have h := hb.2.1 2 (⟨pid.head,hp.2.2⟩,⟨(s.T-1-t),timeBound⟩,⟨pid.value s lane,hl⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DV pid.head (s.T-1-t) (pid.value s lane)<bounds "v" at h
    exact value_window s incoming finalGradient pid (s.T-1-t) lane timeBound (by omega) ▸ h
  · intro t ht lane _ hl
    rw [reverseDo_at _ _ _ _ ht]
    have timeBound : s.T-1-t<s.T := by omega
    have h := hb.2.1 3 (⟨pid.head,hp.2.2⟩,⟨(s.T-1-t),timeBound⟩,⟨pid.value s lane,hl⟩,PUnit.unit) (by simp [inputActive])
    change (pid.head/s.H)*incoming 0+(pid.head%s.H)*incoming 1+pid.value s lane*incoming 3+(s.T-1-t)*incoming 2<bounds "do" at h
    simpa [doOffset,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,ProgramId.value] using h
  · intro t ht lane _ hl
    rw [reverseGradient_at _ _ _ _ ht]
    have timeBound : s.T-1-t<s.T := by omega
    have hj := partialOffset_bound s s.NV s.DK pid.iv pid.head (s.T-1-t) (pid.key s lane) hp.1 hp.2.2 timeBound hl
    have h := hb.2.2.1 2 (⟨_,hj⟩,PUnit.unit) (by simp [outputActive])
    change partialOffset s s.DK pid.iv pid.head (s.T-1-t) (pid.key s lane)<bounds "dk" at h
    exact gradient_window s incoming finalGradient pid (s.T-1-t) lane timeBound (by omega) ▸ h
  · intro t ht lane _ hl
    rw [reverseOutput_at _ _ _ _ ht]
    have timeBound : s.T-1-t<s.T := by omega
    have hj := partialOffset_bound s s.NK s.DV pid.ik pid.head (s.T-1-t) (pid.value s lane) hp.2.1 hp.2.2 timeBound hl
    have h := hb.2.2.1 3 (⟨_,hj⟩,PUnit.unit) (by simp [outputActive])
    change partialOffset s s.DV pid.ik pid.head (s.T-1-t) (pid.value s lane)<bounds "dv" at h
    exact output_window s incoming finalGradient pid (s.T-1-t) lane timeBound (by omega) ▸ h
  · intro hs k _ v _ hk hv
    have h := hb.2.1 4 (⟨pid.head,hp.2.2⟩,⟨pid.key s k,hk⟩,⟨pid.value s v,hv⟩,PUnit.unit)
      (by simp [inputActive,hs])
    change stateOffset s pid.head (pid.key s k) (pid.value s v)<bounds "initial_state" at h
    exact seed_window s incoming finalGradient pid k v ▸ h
  · intro hf k _ v _ hk hv
    have h := hb.2.1 5 (⟨pid.head,hp.2.2⟩,⟨pid.key s k,hk⟩,⟨pid.value s v,hv⟩,PUnit.unit)
      (by simp [inputActive,hf])
    change (pid.head/s.H)*finalGradient 0+(pid.head%s.H)*finalGradient 1+
      pid.key s k*finalGradient 2+pid.value s v*finalGradient 3<bounds "d_final_state" at h
    simpa [SourceSafety.finalOffset,KernelInterface.layout,KernelInterface.buffers,
      KernelInterface.descriptor,layout,ProgramId.key,ProgramId.value] using h

#axiomsClean backward_bounds
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
