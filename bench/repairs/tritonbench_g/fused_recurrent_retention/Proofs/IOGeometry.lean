import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.PublicData
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.SourceSafetyLoops

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

theorem key_window (s : Shape) (incoming finalGradient : Fin 4 → Nat) (pid : ProgramId) (t k : Nat) (ht : t<s.T) (hw : 0<s.DK) :
    keyOffset (KernelInterface.layout s incoming finalGradient) pid t k=
      rowAddress s s.DK pid.head t (pid.key s k) := by
  simp [KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,contiguous4,
    max_eq_left (by omega : 1 ≤ s.T),max_eq_left (by omega : 1 ≤ s.DK),
    keyOffset,rowAddress,ProgramId.key]
  ring

theorem value_window (s : Shape) (incoming finalGradient : Fin 4 → Nat) (pid : ProgramId) (t v : Nat) (ht : t<s.T) (hw : 0<s.DV) :
    valueOffset (KernelInterface.layout s incoming finalGradient) pid t v=
      rowAddress s s.DV pid.head t (pid.value s v) := by
  simp [KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,contiguous4,
    max_eq_left (by omega : 1 ≤ s.T),max_eq_left (by omega : 1 ≤ s.DV),
    valueOffset,rowAddress,ProgramId.value]
  ring

theorem output_window (s : Shape) (incoming finalGradient : Fin 4 → Nat) (pid : ProgramId) (t v : Nat) (ht : t<s.T) (hw : 0<s.DV) :
    outputOffset (KernelInterface.layout s incoming finalGradient) pid t v=
      partialOffset s s.DV pid.ik pid.head t (pid.value s v) := by
  simp [KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,contiguous4,
    max_eq_left (by omega : 1 ≤ s.T),max_eq_left (by omega : 1 ≤ s.DV),
    outputOffset,partialOffset,ProgramId.value,Shape.BH]
  ring

theorem gradient_window (s : Shape) (incoming finalGradient : Fin 4 → Nat) (pid : ProgramId) (t k : Nat) (ht : t<s.T) (hw : 0<s.DK) :
    gradientOffset (KernelInterface.layout s incoming finalGradient) pid t k=
      partialOffset s s.DK pid.iv pid.head t (pid.key s k) := by
  simp [KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,contiguous4,
    max_eq_left (by omega : 1 ≤ s.T),max_eq_left (by omega : 1 ≤ s.DK),
    gradientOffset,partialOffset,ProgramId.key,Shape.BH]
  ring

theorem seed_window (s : Shape) (incoming finalGradient : Fin 4 → Nat) (pid : ProgramId) (k v : Nat) :
    seedOffset (KernelInterface.layout s incoming finalGradient) pid k v=
      stateOffset s pid.head (pid.key s k) (pid.value s v) := by
  simp [KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,
    seedOffset,stateOffset,ProgramId.key,ProgramId.value]
  ring

theorem forward_bounds (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (xs : (io .forward a s p incoming finalGradient seed final).Inputs)
    (bounds : RegionName → Nat) (hb : (io .forward a s p incoming finalGradient seed final).InBounds bounds xs)
    (pid : ProgramId) (hp : pid.Valid s) :
    SourceSafety.ForwardBounds s (KernelInterface.buffers s incoming finalGradient) pid seed final bounds := by
  constructor
  · intro t ht k _ hk
    have h := hb.2.1 0 (⟨pid.head,hp.2.2⟩,⟨t,ht⟩,⟨pid.key s k,hk⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DK pid.head t (pid.key s k)<bounds "q" at h
    exact key_window s incoming finalGradient pid t k ht (by omega) ▸ h
  · intro t ht k _ hk
    have h := hb.2.1 1 (⟨pid.head,hp.2.2⟩,⟨t,ht⟩,⟨pid.key s k,hk⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DK pid.head t (pid.key s k)<bounds "k" at h
    exact key_window s incoming finalGradient pid t k ht (by omega) ▸ h
  · intro t ht v _ hv
    have h := hb.2.1 2 (⟨pid.head,hp.2.2⟩,⟨t,ht⟩,⟨pid.value s v,hv⟩,PUnit.unit) (by simp [inputActive])
    change rowAddress s s.DV pid.head t (pid.value s v)<bounds "v" at h
    exact value_window s incoming finalGradient pid t v ht (by omega) ▸ h
  · intro t ht v _ hv
    have hj := partialOffset_bound s s.NK s.DV pid.ik pid.head t (pid.value s v) hp.2.1 hp.2.2 ht hv
    have h := hb.2.2.1 0 (⟨_,hj⟩,PUnit.unit) (by simp [outputActive])
    change partialOffset s s.DV pid.ik pid.head t (pid.value s v)<bounds "o" at h
    exact output_window s incoming finalGradient pid t v ht (by omega) ▸ h
  · intro hs k _ v _ hk hv
    have h := hb.2.1 4 (⟨pid.head,hp.2.2⟩,⟨pid.key s k,hk⟩,⟨pid.value s v,hv⟩,PUnit.unit)
      (by simp [inputActive,hs])
    change stateOffset s pid.head (pid.key s k) (pid.value s v)<bounds "initial_state" at h
    exact seed_window s incoming finalGradient pid k v ▸ h
  · intro hf k _ v _ hk hv
    have hj := stateOffset_bound s pid.head (pid.key s k) (pid.value s v) hp.2.2 hk hv
    have h := hb.2.2.1 4 (⟨_,hj⟩,PUnit.unit) (by simp [outputActive,hf])
    change stateOffset s pid.head (pid.key s k) (pid.value s v)<bounds "final_state" at h
    exact seed_window s incoming finalGradient pid k v ▸ h

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
