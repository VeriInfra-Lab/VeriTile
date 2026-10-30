import bench.tritonbench_g.lightning_attention.Proofs.InputBinding
import bench.tritonbench_g.lightning_attention.Proofs.IOExecution
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000

theorem forward_outputs (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (forwardIO d).Inputs) (s t : State ℝ) (bound : (forwardIO d).InputBinding x s)
    (mem : t.mem = Real.afterForward d s.mem) :
    (forwardIO d).OutputBinding x (forwardFormula d x) t := by
  intro ⟨port,hport⟩ j active
  have hp : port = 0 := by change port < 1 at hport; omega
  subst port
  rcases j with ⟨head,time,feature,⟨⟩⟩
  change mapCell id (t.mem "output" (head.val*d.N*d.E+time.val*d.E+feature.val)) = _
  rw [mem,Real.forward_output d geometry]
  obtain ⟨hq,hk,hv⟩ := forward_inputs d x s bound head
  change Cell.mk .real _ = Cell.mk .real _
  rw [hq,hk,hv]
  rfl

theorem backward_outputs (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (backwardIO d).Inputs) (s t : State ℝ) (bound : (backwardIO d).InputBinding x s)
    (mem : t.mem = Real.backwardMemory d s.mem) :
    (backwardIO d).OutputBinding x (backwardFormula d x) t := by
  intro ⟨port,hport⟩ j active
  have hp : port < 3 := hport
  interval_cases port
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    change mapCell id (t.mem "queryGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)) = _
    rw [mem,Real.backward_query d geometry]
    obtain ⟨hq,hk,hv,hg⟩ := backward_inputs d x s bound head
    change Cell.mk .real _ = Cell.mk .real _
    rw [hk,hv,hg]
    rfl
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    change mapCell id (t.mem "keyGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)) = _
    rw [mem,Real.backward_key d geometry]
    obtain ⟨hq,hk,hv,hg⟩ := backward_inputs d x s bound head
    change Cell.mk .real _ = Cell.mk .real _
    rw [hq,hv,hg]
    rfl
  · rcases j with ⟨head,time,feature,⟨⟩⟩
    change mapCell id (t.mem "valueGradient" (head.val*d.N*d.E+time.val*d.E+feature.val)) = _
    rw [mem,Real.backward_value d geometry]
    obtain ⟨hq,hk,hv,hg⟩ := backward_inputs d x s bound head
    change Cell.mk .real _ = Cell.mk .real _
    rw [hq,hk,hg]
    rfl

theorem forward_outside (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (forwardIO d).Inputs) (r : RegionName) (o : Nat)
    (outside : (forwardIO d).Outside x r o) (pid : PID) (valid : ValidPID .forward d pid) :
    ¬ Footprint .forward d pid r o := by
  rintro ⟨rfl,time,ht,feature,hf,rfl⟩
  have fc := Geometry.channel_active d geometry pid.channel valid.2 ⟨feature,hf⟩
  have bad := outside.1 (0 : Fin 1) (⟨pid.head,valid.1⟩,⟨time,ht⟩,⟨pid.channel*d.BM+feature,fc⟩,()) trivial
  exact bad.elim (fun h => h rfl) (fun h => h (by
    simp [Geometry.dense,forwardIO,backwardIO,backwardIndex,backwardChannels,denseIndex,
      Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc]))

theorem backward_outside (d : Host.Dimensions) (x : (backwardIO d).Inputs)
    (r : RegionName) (o : Nat) (outside : (backwardIO d).Outside x r o)
    (head : Nat) (hh : head < d.B*d.H) : ¬InterFoot d.parameters head r o := by
  rintro (hw | hw | hw)
  · obtain ⟨rfl,time,ht,feature,hf,rfl⟩ := hw
    have bad := outside.1 (0 : Fin 3) (⟨head,hh⟩,⟨time,ht⟩,⟨feature,hf⟩,()) trivial
    exact bad.elim (fun h => h rfl) (fun h => h (by
    simp [Geometry.dense,forwardIO,backwardIO,backwardIndex,backwardChannels,denseIndex,
      Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc]))
  · obtain ⟨rfl,time,ht,feature,hf,rfl⟩ := hw
    have bad := outside.1 (1 : Fin 3) (⟨head,hh⟩,⟨time,ht⟩,⟨feature,hf⟩,()) trivial
    exact bad.elim (fun h => h rfl) (fun h => h (by
    simp [Geometry.dense,forwardIO,backwardIO,backwardIndex,backwardChannels,denseIndex,
      Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc]))
  · obtain ⟨rfl,time,ht,feature,hf,rfl⟩ := hw
    have bad := outside.1 (2 : Fin 3) (⟨head,hh⟩,⟨time,ht⟩,⟨feature,hf⟩,()) trivial
    exact bad.elim (fun h => h rfl) (fun h => h (by
    simp [Geometry.dense,forwardIO,backwardIO,backwardIndex,backwardChannels,denseIndex,
      Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc]))

theorem forward_frame (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (forwardIO d).Inputs) (s t : State ℝ) (mem : t.mem = Real.afterForward d s.mem) :
    (forwardIO d).Frame x s t := by
  intro r o outside
  rw [mem]
  apply grid_frame
  intro pid hp
  exact forward_outside d geometry x r o outside pid ((mem_grid _ _ _).mp hp)

theorem backward_frame (d : Host.Dimensions) (x : (backwardIO d).Inputs)
    (s t : State ℝ) (mem : t.mem = Real.backwardMemory d s.mem) :
    (backwardIO d).Frame x s t := by
  intro r o outside
  rw [mem]
  unfold Real.backwardMemory Real.backwardIntra
  rw [grid_frame]
  · apply grid_frame
    intro pid hp hw
    apply backward_outside d x r o outside pid.head ((mem_grid _ _ _).mp hp).1
    rcases hw with hw | hw | hw
    · exact Or.inl (rows_head _ _ _ _ _ _ _ _ hw)
    · exact Or.inr (Or.inl (rows_head _ _ _ _ _ _ _ _ hw))
    · exact Or.inr (Or.inr (rows_head _ _ _ _ _ _ _ _ hw))
  · intro pid hp
    exact backward_outside d x r o outside pid.head ((mem_grid _ _ _).mp hp).1

theorem forward_result (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (forwardIO d).Inputs) (s : State ℝ) (bound : (forwardIO d).InputBinding x s) :
    ∃ t, runSchedule ((forwardIO d).schedule x) s = some t ∧
      (forwardIO d).OutputBinding x (forwardFormula d x) t ∧ (forwardIO d).Frame x s t := by
  obtain ⟨t,ran,mem⟩ := IOExecution.forward d s
  exact ⟨t,ran,forward_outputs d geometry x s t bound mem,forward_frame d geometry x s t mem⟩

theorem backward_result (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (backwardIO d).Inputs) (s : State ℝ) (bound : (backwardIO d).InputBinding x s) :
    ∃ t, runSchedule ((backwardIO d).schedule x) s = some t ∧
      (backwardIO d).OutputBinding x (backwardFormula d x) t ∧ (backwardIO d).Frame x s t := by
  obtain ⟨t,ran,mem⟩ := IOExecution.backward d s
  exact ⟨t,ran,backward_outputs d geometry x s t bound mem,backward_frame d x s t mem⟩

#axiomsClean forward_result
#axiomsClean backward_result
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
