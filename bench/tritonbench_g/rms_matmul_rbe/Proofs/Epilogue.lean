import bench.tritonbench_g.rms_matmul_rbe.Proofs.EpilogueFrame
import bench.tritonbench_g.rms_matmul_rbe.Proofs.RotateValues
import bench.tritonbench_g.rms_matmul_rbe.Proofs.StoreValues

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def epilogueValues (a : Arithmetic α) (p : Arguments) (epsilon theta : α) (pm pn : Nat)
    (moments accumulator : Values α .real [16,64]) : Values α .real [16,64] :=
  let norm := normalized a p epsilon moments accumulator
  if p.rotary then rotated a p theta pm pn norm else norm

theorem epilogue_partition (fp8 outputFP8 : Bool) (p : Arguments) :
    epilogue fp8 outputFP8 p=normalization p ++
      [.ifThen (.constBool p.rotary) (rotation p)] ++ storing outputFP8 p := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem tileWrite_memory (a : Arithmetic α) (outputFP8 : Bool) (p : Arguments)
    (pb pm pn : Nat) (values : Values α .real [16,64]) (s t : State α)
    (h : s.mem=t.mem) :
    (tileWrite a outputFP8 p pb pm pn values s).mem=
      (tileWrite a outputFP8 p pb pm pn values t).mem :=
  maskedScatter_memory _ _ _ _ _ s t h

/-- Constructive execution of normalization, optional pair rotation, the
output cast and every active store, with a separate value-and-write target. -/
theorem epilogueExecution {α : Type} [Inhabited α] (a : Arithmetic α)
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
        (epilogueValues a p epsilon theta pm pn moments accumulator) s).mem := by
  let norm := normalizationExecution a p s epsilon moments accumulator he hmom ha
  have nh : HeaderPreserved s norm.val :=
    normalization_header a p s epsilon moments accumulator he hmom ha
  have nv := normalization_value a p s epsilon moments accumulator he hmom ha
  have nf : norm.val.mem=s.mem := normalization_frame a p s epsilon moments accumulator he hmom ha
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
      next.mem=s.mem ∧ HeaderPreserved s next := by
    cases hr : p.rotary with
    | «false» =>
      refine ⟨norm.val,?_,?_,nf,nh⟩
      · simp [FP.Structural.step,evalOp_unfold,hr]
      · simpa only [epilogueValues,hr,Bool.false_eq_true,if_false] using nv
    | «true» =>
      let rot := rotationExecution a p norm.val theta pm pn
        (normalized a p epsilon moments accumulator) nt nm nn nv
      refine ⟨rot.val,?_,?_,?_,HeaderPreserved.trans s norm.val rot.val nh
        (rotation_header a p norm.val theta pm pn _ nt nm nn nv)⟩
      · simpa [FP.Structural.step,evalOp_unfold,hr] using rot.property
      · simpa only [epilogueValues,hr,if_true] using
          rotation_value a p norm.val theta pm pn _ nt nm nn nv
      · exact (rotation_frame a p norm.val theta pm pn _ nt nm nn nv).trans nf
  obtain ⟨next,hchoice,hv,hf,hh⟩ := choice
  have nb : next.regs .nat [] "pid_batch"=some (fun _ => pb) :=
    (hh _ _ _ (by simp [headerNames])).trans hb
  have nm : next.regs .nat [] "pid_m"=some (fun _ => pm) :=
    (hh _ _ _ (by simp [headerNames])).trans hm
  have nn : next.regs .nat [] "pid_n"=some (fun _ => pn) :=
    (hh _ _ _ (by simp [headerNames])).trans hn
  let stored := storingExecution a outputFP8 p next pb pm pn _ nb nm nn hv
  refine ⟨stored.val,?_,?_⟩
  · rw [epilogue_partition,List.append_assoc,run_append,norm.property]
    simp only [Option.bind_some,run_append,run,hchoice,Option.bind_some]
    exact stored.property
  · exact (storing_values a outputFP8 p next pb pm pn _ nb nm nn hv).trans
      (tileWrite_memory a outputFP8 p pb pm pn _ next s hf)

#axiomsClean epilogue_partition
#axiomsClean epilogueExecution
#specNonCircular epilogueValues avoiding [epilogueExecution,run,rotationExecution,normalizationExecution]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
