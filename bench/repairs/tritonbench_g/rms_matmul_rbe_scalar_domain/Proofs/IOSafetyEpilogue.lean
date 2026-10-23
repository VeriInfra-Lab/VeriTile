import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOSafetyStore

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem normalization_safe (M : Algebra α) (p : Arguments) (capacity : RegionName → Nat) (s : State α) :
    traceSafeList M capacity (normalization p) s := by
  apply pure_trace
  intro st member t
  simp only [normalization,epilogue,source,TypedSource.kernel,ComputeKernel.surfaceBody,
    List.take,List.drop,List.getElem?_cons_zero,List.getElem?_cons_succ,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl
  all_goals simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem rotation_safe (M : Algebra α) (p : Arguments) (capacity : RegionName → Nat) (s : State α) :
    traceSafeList M capacity (rotation p) s := by
  apply pure_trace
  intro st member t
  simp only [rotation,rotaryBody,epilogue,source,TypedSource.kernel,ComputeKernel.surfaceBody,
    List.take,List.drop,List.getElem?_cons_zero,List.getElem?_cons_succ,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem native_rotation_safe (M : Algebra α) (plan : RotaryPlan) (p : Arguments)
    (capacity : RegionName → Nat) (s : State α) :
    traceSafeList M capacity (nativeRotation plan p) s := by
  apply pure_trace
  intro st member t
  simp only [nativeRotation,selectedRotary,rotaryBody,epilogue,source,TypedSource.kernel,ComputeKernel.surfaceBody,
    List.take,List.drop,List.getElem?_cons_zero,List.getElem?_cons_succ,List.cons_append,List.nil_append,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [nativeRealNode,nativeImagNode,fp,pairRef,pairMul,
    stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem source_epilogue_safe (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (pb pm pn : Nat) (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (hmom : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator)
    (capacity : RegionName → Nat)
    (bound : ∀ i : TileIndex [16,64],outputActive p pm pn i → outputOffset p pb pm pn i<capacity p.OUT) :
    traceSafeList a.model capacity (epilogue fp8 outputFP8 p) s := by
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
  have rotationSafe : stepSafe a.model capacity (.ifThen (.constBool p.rotary) (rotation p)) norm.val := by
    simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,true_and]
    split
    · exact rotation_safe a.model p capacity norm.val
    · trivial
  rw [epilogue_partition,List.append_assoc,traceSafeList_append]
  refine ⟨normalization_safe a.model p capacity s,?_⟩
  intro nstate hnstate
  have equal : nstate=norm.val := Option.some.inj (hnstate.symm.trans norm.property)
  subst nstate
  apply traceSafeList_cons_of_step rotationSafe hchoice
  exact storing_safe a outputFP8 p next pb pm pn _ nb nm nn hv capacity bound

#axiomsClean source_epilogue_safe

theorem native_epilogue_safe (a : Arithmetic α)
    (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (pb pm pn : Nat) (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hb : s.regs .nat [] "pid_batch"=some (fun _ => pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (hmom : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator)
    (capacity : RegionName → Nat)
    (bound : ∀ i : TileIndex [16,64],outputActive p pm pn i → outputOffset p pb pm pn i<capacity p.OUT) :
    traceSafeList a.model capacity (selectedEpilogue plan fp8 outputFP8 p) s := by
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
  have rotationSafe : stepSafe a.model capacity (.ifThen (.constBool p.rotary) (nativeRotation plan p)) norm.val := by
    simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,true_and]
    split
    · exact native_rotation_safe a.model plan p capacity norm.val
    · trivial
  rw [native_epilogue_partition,List.append_assoc,traceSafeList_append]
  refine ⟨normalization_safe a.model p capacity s,?_⟩
  intro nstate hnstate
  have equal : nstate=norm.val := Option.some.inj (hnstate.symm.trans norm.property)
  subst nstate
  apply traceSafeList_cons_of_step rotationSafe hchoice
  exact storing_safe a outputFP8 p next pb pm pn _ nb nm nn hv capacity bound

#axiomsClean native_epilogue_safe

theorem epilogue_safe (a : Arithmetic α) (native : Bool) (plan : RotaryPlan)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (pb pm pn : Nat) (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _=>epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _=>theta))
    (hb : s.regs .nat [] "pid_batch"=some (fun _=>pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _=>pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _=>pn))
    (hmom : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator)
    (capacity : RegionName → Nat)
    (bound : ∀ i : TileIndex [16,64],outputActive p pm pn i → outputOffset p pb pm pn i<capacity p.OUT) :
    traceSafeList a.model capacity
      (if native then selectedEpilogue plan fp8 outputFP8 p else epilogue fp8 outputFP8 p) s := by
  cases native
  · exact source_epilogue_safe a fp8 outputFP8 p s epsilon theta pb pm pn moments accumulator he ht hb hm hn hmom ha capacity bound
  · exact native_epilogue_safe a plan fp8 outputFP8 p s epsilon theta pb pm pn moments accumulator he ht hb hm hn hmom ha capacity bound

#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
