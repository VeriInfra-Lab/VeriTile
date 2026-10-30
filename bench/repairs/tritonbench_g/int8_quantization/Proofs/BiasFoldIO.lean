import bench.repairs.tritonbench_g.int8_quantization.Proofs.BiasFold

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural Revision002
noncomputable section
variable {α : Type} [Inhabited α]


theorem bias_initial (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (w : Schedule.Program) (s : State α) : (biasInvocation M ops p w).initial s = Grid.atProgram s w :=
  invocation_initial M ops p w s

theorem bias_run (M : Algebra α) (signs : BiasFold.HalfSigns M) (ops : Model.Primitives α)
    (p : Parameters) (positive : 0<p.width) (w : Schedule.Program) (s : State α) :
    (biasInvocation M ops p w).run s = (invocation M ops p w).run s := by
  unfold Invocation.run
  rw [bias_initial,invocation_initial]
  exact (BiasFold.source_run M signs ops (Grid.parameters p w) positive _).trans
    (Revision002.source_run M ops (Grid.parameters p w) positive _).symm

theorem bias_safe (M : Algebra α) (signs : BiasFold.HalfSigns M) (ops : Model.Primitives α)
    (p : Parameters) (positive : 0<p.width) (values : Inputs α p) (bounds : RegionName → Nat)
    (hb : (quantizationIO M ops p).InBounds bounds values) (w : Schedule.Program)
    (hw : w.index < count p) (s : State α) : (biasInvocation M ops p w).Safe bounds s := by
  unfold Invocation.Safe
  rw [bias_initial]
  apply BiasFold.source_safe M signs ops (Grid.parameters p w) positive (Grid.atProgram s w)
  · intro i hi
    exact hb.2.1 ⟨0,by simp [quantizationIO,memory,inputChannels]⟩ (⟨w.index,hw⟩,i.1,PUnit.unit) hi
  · intro i hi
    exact hb.2.2.1 ⟨0,by simp [quantizationIO,memory,outputChannels]⟩ (⟨w.index,hw⟩,i.1,PUnit.unit) hi
  · exact hb.2.2.1 ⟨1,by simp [quantizationIO,memory,outputChannels]⟩ (⟨w.index,hw⟩,PUnit.unit) trivial


theorem bias_schedule_run (M : Algebra α) (signs : BiasFold.HalfSigns M) (ops : Model.Primitives α)
    (p : Parameters) (positive : 0<p.width) (order : List Schedule.Program) (s : State α) :
    runSchedule (order.map (biasInvocation M ops p)) s =
      runSchedule (order.map (invocation M ops p)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons w rest ih => simp only [List.map_cons,runSchedule,bias_run M signs ops p positive,ih]

theorem bias_schedule_safe (M : Algebra α) (signs : BiasFold.HalfSigns M) (ops : Model.Primitives α)
    (p : Parameters) (positive : 0<p.width) (values : Inputs α p) (bounds : RegionName → Nat)
    (hb : (quantizationIO M ops p).InBounds bounds values) (order : List Schedule.Program)
    (members : ∀ w∈order,w.index<count p) (s : State α) :
    ScheduleSafe bounds (order.map (biasInvocation M ops p)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons w rest ih =>
    constructor
    · exact bias_safe M signs ops p positive values bounds hb w (members w (by simp)) s
    · intro t _
      exact ih (fun w hw=>members w (by simp [hw])) t

/-- Complete Q/K block schedules, both typed outputs and exact exterior cells.
The two local constant equations impose no restriction on input payloads. -/
theorem bias_equivalent (M : Algebra α) (signs : BiasFold.HalfSigns M) (ops : Model.Primitives α)
    (p : Parameters) (positive : 0<p.width) (block : 0<p.blockRows) :
    (quantizationIO M ops p).Equivalent (biasVariant M ops p) := by
  intro values _ bounds hb initial hin
  have members (w : Schedule.Program) (hw : w∈Schedule.programs 0 (count p)) : w.index<count p :=
    by simpa using (Schedule.program_bounds 0 (count p) w hw).2.2
  obtain ⟨final,ran,_,_,exterior⟩ := RawKernel.complete M ops p positive block initial
  have leftRun : runSchedule ((quantizationIO M ops p).schedule values) initial=some final := by
    change runSchedule ((Schedule.programs 0 (count p)).map (invocation M ops p)) initial = _
    rw [schedule_run]
    exact ran
  have rightRun : runSchedule ((biasVariant M ops p).schedule values) initial=some final := by
    change runSchedule ((Schedule.programs 0 (count p)).map (biasInvocation M ops p)) initial = _
    rw [bias_schedule_run M signs ops p positive]
    exact leftRun
  have framed := frame M ops p values initial final exterior
  exact ⟨schedule_safe M ops p positive values bounds hb _ members initial,
    bias_schedule_safe M signs ops p positive values bounds hb _ members initial,
    final,final,leftRun,rightRun,fun _ _ _=>rfl,framed,framed⟩



#axiomsClean bias_equivalent
end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
