import bench.tritonbench_g.rms_matmul_rbe.Proofs.Epilogue
import bench.tritonbench_g.rms_matmul_rbe.Proofs.NativeEpilogue
import bench.tritonbench_g.rms_matmul_rbe.Proofs.StoreFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

theorem normalization_pids {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (epsilon : α)
    (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (hm : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (normalizationExecution a p s epsilon moments accumulator he hm ha).val.pids=s.pids := rfl

theorem rotation_pids {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (rotationExecution a p s theta pm pn accumulator ht hm hn ha).val.pids=s.pids := rfl

theorem native_rotation_pids {α : Type} [Inhabited α] (a : Arithmetic α)
    (plan : RotaryPlan) (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (nativeRotationExecution a plan p s theta pm pn accumulator ht hm hn ha).val.pids=s.pids := by
  rcases plan with ⟨swap,fuse⟩
  cases swap <;> cases fuse <;> rfl

theorem epilogueExecutionFull {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (pb pm pn : Nat) (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (hmom : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    ∃ final, run a.model (epilogue fp8 outputFP8 p) s=some final ∧
      final.mem=(tileWrite a outputFP8 p pb pm pn
        (epilogueValues a p epsilon theta pm pn moments accumulator) s).mem ∧
      HeaderPreserved s final ∧ final.pids=s.pids := by
  let norm := normalizationExecution a p s epsilon moments accumulator he hmom ha
  have nh : HeaderPreserved s norm.val :=
    normalization_header a p s epsilon moments accumulator he hmom ha
  have nv := normalization_value a p s epsilon moments accumulator he hmom ha
  have nf : norm.val.mem=s.mem := normalization_frame a p s epsilon moments accumulator he hmom ha
  have np : norm.val.pids=s.pids := rfl
  have nt : norm.val.regs .real [] "THETA"=some (fun _ => theta) :=
    (nh _ _ _ (by simp [headerNames])).trans ht
  have nm : norm.val.regs .nat [] "pid_m"=some (fun _ => pm) :=
    (nh _ _ _ (by simp [headerNames])).trans hm
  have nn : norm.val.regs .nat [] "pid_n"=some (fun _ => pn) :=
    (nh _ _ _ (by simp [headerNames])).trans hn
  have choice : ∃ next,
      FP.Structural.step a.model (.ifThen (.constBool p.rotary) (rotation p)) norm.val=some next ∧
      next.regs .real [16,64] "accumulator"=
        some (epilogueValues a p epsilon theta pm pn moments accumulator) ∧
      next.mem=s.mem ∧ HeaderPreserved s next ∧ next.pids=s.pids := by
    cases hr : p.rotary with
    | «false» =>
      refine ⟨norm.val,?_,?_,nf,nh,np⟩
      · simp [FP.Structural.step,evalOp_unfold,hr]
      · simpa only [epilogueValues,hr,Bool.false_eq_true,if_false] using nv
    | «true» =>
      let rot := rotationExecution a p norm.val theta pm pn
        (normalized a p epsilon moments accumulator) nt nm nn nv
      refine ⟨rot.val,?_,?_,?_,HeaderPreserved.trans s norm.val rot.val nh
        (rotation_header a p norm.val theta pm pn _ nt nm nn nv),
        (rotation_pids a p norm.val theta pm pn _ nt nm nn nv).trans np⟩
      · simpa [FP.Structural.step,evalOp_unfold,hr] using rot.property
      · simpa only [epilogueValues,hr,if_true] using
          rotation_value a p norm.val theta pm pn _ nt nm nn nv
      · exact (rotation_frame a p norm.val theta pm pn _ nt nm nn nv).trans nf
  obtain ⟨next,hchoice,hv,hf,hh,hp⟩ := choice
  have nb : next.regs .nat [] "pid_batch"=some (fun _ => pb) :=
    (hh _ _ _ (by simp [headerNames])).trans hb
  have nm : next.regs .nat [] "pid_m"=some (fun _ => pm) :=
    (hh _ _ _ (by simp [headerNames])).trans hm
  have nn : next.regs .nat [] "pid_n"=some (fun _ => pn) :=
    (hh _ _ _ (by simp [headerNames])).trans hn
  let stored := storingExecution a outputFP8 p next pb pm pn _ nb nm nn hv
  refine ⟨stored.val,?_,?_,HeaderPreserved.trans s next stored.val hh
    (storing_header a outputFP8 p next pb pm pn _ nb nm nn hv),
    (storing_pids a outputFP8 p next pb pm pn _ nb nm nn hv).trans hp⟩
  · rw [epilogue_partition,List.append_assoc,run_append,norm.property]
    simp only [Option.bind_some,run_append,run,hchoice,Option.bind_some]
    exact stored.property
  · exact (storing_values a outputFP8 p next pb pm pn _ nb nm nn hv).trans
      (tileWrite_memory a outputFP8 p pb pm pn _ next s hf)

#axiomsClean epilogueExecutionFull

theorem nativeEpilogueExecutionFull {α : Type} [Inhabited α] (a : Arithmetic α)
    (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (pb pm pn : Nat) (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (hmom : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    ∃ final, run a.model (selectedEpilogue plan fp8 outputFP8 p) s=some final ∧
      final.mem=(tileWrite a outputFP8 p pb pm pn
        (nativeEpilogueValues a plan p epsilon theta pm pn moments accumulator) s).mem ∧
      HeaderPreserved s final ∧ final.pids=s.pids := by
  let norm := normalizationExecution a p s epsilon moments accumulator he hmom ha
  have nh : HeaderPreserved s norm.val :=
    normalization_header a p s epsilon moments accumulator he hmom ha
  have nv := normalization_value a p s epsilon moments accumulator he hmom ha
  have nf : norm.val.mem=s.mem := normalization_frame a p s epsilon moments accumulator he hmom ha
  have np : norm.val.pids=s.pids := rfl
  have nt : norm.val.regs .real [] "THETA"=some (fun _ => theta) :=
    (nh _ _ _ (by simp [headerNames])).trans ht
  have nm : norm.val.regs .nat [] "pid_m"=some (fun _ => pm) :=
    (nh _ _ _ (by simp [headerNames])).trans hm
  have nn : norm.val.regs .nat [] "pid_n"=some (fun _ => pn) :=
    (nh _ _ _ (by simp [headerNames])).trans hn
  have choice : ∃ next,
      FP.Structural.step a.model (.ifThen (.constBool p.rotary) (nativeRotation plan p)) norm.val=some next ∧
      next.regs .real [16,64] "accumulator"=
        some (nativeEpilogueValues a plan p epsilon theta pm pn moments accumulator) ∧
      next.mem=s.mem ∧ HeaderPreserved s next ∧ next.pids=s.pids := by
    cases hr : p.rotary with
    | «false» =>
      refine ⟨norm.val,?_,?_,nf,nh,np⟩
      · simp [FP.Structural.step,evalOp_unfold,hr]
      · simpa only [nativeEpilogueValues,hr,Bool.false_eq_true,if_false] using nv
    | «true» =>
      let rot := nativeRotationExecution a plan p norm.val theta pm pn
        (normalized a p epsilon moments accumulator) nt nm nn nv
      refine ⟨rot.val,?_,?_,?_,HeaderPreserved.trans s norm.val rot.val nh
        (native_rotation_header a plan p norm.val theta pm pn _ nt nm nn nv),
        (native_rotation_pids a plan p norm.val theta pm pn _ nt nm nn nv).trans np⟩
      · simpa [FP.Structural.step,evalOp_unfold,hr] using rot.property
      · simpa only [nativeEpilogueValues,hr,if_true] using
          native_rotation_value a plan p norm.val theta pm pn _ nt nm nn nv
      · exact (native_rotation_frame a plan p norm.val theta pm pn _ nt nm nn nv).trans nf
  obtain ⟨next,hchoice,hv,hf,hh,hp⟩ := choice
  have nb : next.regs .nat [] "pid_batch"=some (fun _ => pb) :=
    (hh _ _ _ (by simp [headerNames])).trans hb
  have nm : next.regs .nat [] "pid_m"=some (fun _ => pm) :=
    (hh _ _ _ (by simp [headerNames])).trans hm
  have nn : next.regs .nat [] "pid_n"=some (fun _ => pn) :=
    (hh _ _ _ (by simp [headerNames])).trans hn
  let stored := storingExecution a outputFP8 p next pb pm pn _ nb nm nn hv
  refine ⟨stored.val,?_,?_,HeaderPreserved.trans s next stored.val hh
    (storing_header a outputFP8 p next pb pm pn _ nb nm nn hv),
    (storing_pids a outputFP8 p next pb pm pn _ nb nm nn hv).trans hp⟩
  · rw [native_epilogue_partition,List.append_assoc,run_append,norm.property]
    simp only [Option.bind_some,run_append,run,hchoice,Option.bind_some]
    exact stored.property
  · exact (storing_values a outputFP8 p next pb pm pn _ nb nm nn hv).trans
      (tileWrite_memory a outputFP8 p pb pm pn _ next s hf)

#axiomsClean nativeEpilogueExecutionFull

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
