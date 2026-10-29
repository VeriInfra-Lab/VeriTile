import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumeric
namespace VeriTile.Bench.Repairs.Conv2d.KernelNumeric
open VeriTile.Triton FP.Structural KernelBinding KernelGrid
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16384
set_option linter.unusedSimpArgs false
variable {α ι : Type} [Inhabited α]

/-- Lift the scalar recurrence through an arbitrary source loop, retaining
its exact trip sequence. This includes the zero-trip case. -/
theorem foldRange_lane (o : Operations α)
    (G : Nat → (ι → Value α) → ι → Value α) (f : Nat → ι → α → α)
    (step : ∀ (q : Nat) (acc : ι → α),G q (fun i=>.scalar (acc i))=fun i=>.scalar (f q i (acc i)))
    (cur stop stride : Nat) (initial : ι → α) :
    foldRange G cur stop stride (fun i=>.scalar (initial i)) =
      fun i=>.scalar (foldRange (fun q x=>f q i x) cur stop stride (initial i)) := by
  by_cases hz : stride=0
  · simp [foldRange,hz]
  by_cases hc : cur<stop
  · conv_lhs => rw [foldRange]
    simp only [dif_neg hz,dif_pos hc,step]
    rw [foldRange_lane o G f step (cur+stride) stop stride]
    funext i
    conv_rhs => rw [foldRange]
    simp only [dif_neg hz,dif_pos hc]
  · simp [foldRange,hz,hc]
termination_by stop-cur
decreasing_by omega

def laneStep (o : Operations α) (half : Bool) (s : Shape) (f : Flags) (pid : Program)
    (t : State (Value α)) (h w c : Nat) (i : Index) (acc : α) : α :=
  o.lower32 (fun e=>input o half (parameters s f) (atProgram pid t) h w c (i.1,e,PUnit.unit))
    (fun e=>weight o half (parameters s f) (atProgram pid t) h w c (e,i.2.1,PUnit.unit)) acc

theorem half_lane (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (pid : Program) (t : State (Value α)) (i : Index) :
    KernelHalf.outputValue (algebra o) (halfDot o) (parameters s f) (atProgram pid t) i=
      .scalar (foldRange (fun h acc=>foldRange (fun w acc=>foldRange
        (fun c acc=>laneStep o Bool.true s f pid t h w c i acc)
        0 s.cg 32 acc) 0 s.kw 1 acc) 0 s.kh 1 o.zero) := by
  have gd : KernelHalf.igd (parameters s f)=s.cg := (group_division s v).1
  have update (h w c : Nat) (acc : Index → α) :
      KernelHalf.update (algebra o) (halfDot o) (parameters s f) (atProgram pid t) h w c
        (fun i=>.scalar (acc i))=fun i=>.scalar (laneStep o Bool.true s f pid t h w c i (acc i)) := by
    funext i
    simp [KernelHalf.update,KernelHalf.inputValue,KernelHalf.weightValue,halfDot,dot,algebra,add,scalar,
      laneStep,input,weight,KernelOperands.input,KernelOperands.weight,KernelHalf.BIN,KernelDefault.BIN,parameters]
    all_goals rfl
  have channels (h w : Nat) (acc : Index → α) :
      KernelHalf.channels (algebra o) (halfDot o) (parameters s f) (atProgram pid t) h w
        (fun i=>.scalar (acc i))=fun i=>.scalar (foldRange
          (fun c x=>laneStep o Bool.true s f pid t h w c i x) 0 s.cg 32 (acc i)) := by
    unfold KernelHalf.channels
    rw [gd]
    exact foldRange_lane o _ _ (update h w) 0 s.cg 32 acc
  have widths (h : Nat) (acc : Index → α) :
      KernelHalf.widths (algebra o) (halfDot o) (parameters s f) (atProgram pid t) h
        (fun i=>.scalar (acc i))=fun i=>.scalar (foldRange (fun w x=>foldRange
          (fun c x=>laneStep o Bool.true s f pid t h w c i x) 0 s.cg 32 x) 0 s.kw 1 (acc i)) := by
    unfold KernelHalf.widths
    exact foldRange_lane o _ _ (channels h) 0 s.kw 1 acc
  unfold KernelHalf.outputValue KernelHalf.accumulation
  change (algebra o).cast none .real .fp32 (foldRange _ 0 s.kh 1 (fun _=>.scalar o.zero) i)=_
  rw [foldRange_lane o _ _ widths]
  rfl

theorem default_lane (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (pid : Program) (t : State (Value α)) (i : Index) :
    KernelDefault.outputValue (algebra o) (defaultDot o) (parameters s f) (atProgram pid t) i=
      .scalar (foldRange (fun h acc=>foldRange (fun w acc=>foldRange
        (fun c acc=>laneStep o Bool.false s f pid t h w c i acc)
        0 s.cg 32 acc) 0 s.kw 1 acc) 0 s.kh 1 o.zero) := by
  have gd : KernelDefault.igd (parameters s f)=s.cg := (group_division s v).1
  have update (h w c : Nat) (acc : Index → α) :
      KernelDefault.update (algebra o) (defaultDot o) (parameters s f) (atProgram pid t) h w c
        (fun i=>.scalar (acc i))=fun i=>.scalar (laneStep o Bool.false s f pid t h w c i (acc i)) := by
    funext i
    simp [KernelDefault.update,KernelDefault.inputValue,KernelDefault.weightValue,defaultDot,dot,algebra,add,scalar,
      laneStep,input,weight,KernelOperands.input,KernelOperands.weight,KernelHalf.BIN,KernelDefault.BIN,parameters]
    all_goals rfl
  have channels (h w : Nat) (acc : Index → α) :
      KernelDefault.channels (algebra o) (defaultDot o) (parameters s f) (atProgram pid t) h w
        (fun i=>.scalar (acc i))=fun i=>.scalar (foldRange
          (fun c x=>laneStep o Bool.false s f pid t h w c i x) 0 s.cg 32 (acc i)) := by
    unfold KernelDefault.channels
    rw [gd]
    exact foldRange_lane o _ _ (update h w) 0 s.cg 32 acc
  have widths (h : Nat) (acc : Index → α) :
      KernelDefault.widths (algebra o) (defaultDot o) (parameters s f) (atProgram pid t) h
        (fun i=>.scalar (acc i))=fun i=>.scalar (foldRange (fun w x=>foldRange
          (fun c x=>laneStep o Bool.false s f pid t h w c i x) 0 s.cg 32 x) 0 s.kw 1 (acc i)) := by
    unfold KernelDefault.widths
    exact foldRange_lane o _ _ (channels h) 0 s.kw 1 acc
  unfold KernelDefault.outputValue KernelDefault.accumulation
  change (algebra o).cast none .real .fp32 (foldRange _ 0 s.kh 1 (fun _=>.scalar o.zero) i)=_
  rw [foldRange_lane o _ _ widths]
  rfl

theorem source_reference (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (pid : Program) (t : State (Value α)) (i : Index) :
    KernelGrid.value (algebra o) (halfDot o) (defaultDot o) s f pid t i=
      .scalar (laneReference o s f pid t i) := by
  unfold KernelGrid.value
  cases hf:f.fp16
  · simp only [hf,Bool.false_eq_true,if_false]
    rw [default_lane o s v f]
    simp only [laneReference,laneStep,hf]
  · simp only [hf,if_true]
    rw [half_lane o s v f]
    simp only [laneReference,laneStep,hf]

#axiomsClean half_lane
#axiomsClean default_lane
#axiomsClean source_reference
end
end VeriTile.Bench.Repairs.Conv2d.KernelNumeric
