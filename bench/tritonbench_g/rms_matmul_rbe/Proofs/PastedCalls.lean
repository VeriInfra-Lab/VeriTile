import bench.tritonbench_g.rms_matmul_rbe.Proofs.Pair

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Pasted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

theorem original_body (fp8 : Bool) (c : ParentArguments) :
    (pastedQKV Bool.false fp8 c).surfaceBody=qkvCode Bool.false fp8 c.phase := by
  cases fp8 <;> rfl

theorem candidate_body (fp8 : Bool) (c : ParentArguments) :
    (pastedQKV Bool.true fp8 c).surfaceBody=Pair.code fp8 c.phase := by
  cases fp8 <;> rfl

/-- Exact full-state execution equality, including failure, for the complete
three-call source. Candidate equality uses only signed-index commutation. -/
theorem call_run {α : Type} [Inhabited α] (a : Arithmetic α)
    (candidate fp8 : Bool) (c : ParentArguments) (s : State α) :
    run a.model (pastedQKV candidate fp8 c).surfaceBody s=
      run a.model (qkvCode Bool.false fp8 c.phase) s := by
  cases candidate
  · rw [original_body]
  · rw [candidate_body]
    exact Pair.qkv_run a.model fp8 c.phase s

theorem call_safe {α : Type} [Inhabited α] (a : Arithmetic α)
    (candidate fp8 : Bool) (c : ParentArguments) (capacity : RegionName → Nat) (s : State α) :
    Safety.traceSafeList a.model capacity (pastedQKV candidate fp8 c).surfaceBody s ↔
      Safety.traceSafeList a.model capacity (qkvCode Bool.false fp8 c.phase) s := by
  cases candidate
  · rw [original_body]
  · rw [candidate_body]
    exact Pair.qkv_safe a.model fp8 c.phase capacity s

theorem schedule_run {α : Type} [Inhabited α] (a : Arithmetic α)
    (candidate fp8 : Bool) (c : ParentArguments) (epsilon theta : α)
    (order : List (Nat → Nat)) (s : State α) :
    runSchedule (order.map (job a candidate fp8 c epsilon theta)) s=
      runSchedule (order.map (IO.job a Bool.false fp8 c.phase epsilon theta)) s := by
  induction order generalizing s with
  | nil=>rfl
  | cons pid rest ih=>
    have same (s : State α) : (job a candidate fp8 c epsilon theta pid).run s=
        (IO.job a Bool.false fp8 c.phase epsilon theta pid).run s := call_run a candidate fp8 c _
    simp only [List.map_cons,runSchedule,same,ih]

theorem schedule_safe {α : Type} [Inhabited α] (a : Arithmetic α)
    (candidate fp8 : Bool) (c : ParentArguments) (epsilon theta : α)
    (order : List (Nat → Nat)) (capacity : RegionName → Nat) (s : State α) :
    ScheduleSafe capacity (order.map (job a candidate fp8 c epsilon theta)) s ↔
      ScheduleSafe capacity (order.map (IO.job a Bool.false fp8 c.phase epsilon theta)) s := by
  induction order generalizing s with
  | nil=>rfl
  | cons pid rest ih=>
    have safe (s : State α) : (job a candidate fp8 c epsilon theta pid).Safe capacity s ↔
        (IO.job a Bool.false fp8 c.phase epsilon theta pid).Safe capacity s := call_safe a candidate fp8 c capacity _
    have ran (s : State α) : (job a candidate fp8 c epsilon theta pid).run s=
        (IO.job a Bool.false fp8 c.phase epsilon theta pid).run s := call_run a candidate fp8 c _
    simp only [List.map_cons,ScheduleSafe,safe,ran,ih]

/-- Transport the complete IO judgment, preserving input domain, output
observations, memory frame and trace safety for every initial state. -/
theorem implements_iff {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (c : ParentArguments) (epsilon theta : α) (order : List (Nat → Nat))
    (domain : ArrayValues α (IO.inputs fp8 c.phase) → Prop)
    (f : ArrayValues α (IO.inputs fp8 c.phase) → ChannelValues α (IO.outputs fp8 c.phase order)) :
    (arrayIO a fp8 c epsilon theta order).ImplementsOn domain f ↔
      (IO.qkvIO a Bool.false fp8 c.phase epsilon theta order).ImplementsOn domain f := by
  constructor
  · intro proof vals hd initial binding
    obtain ⟨safe,final,ran,output,frame⟩:=proof vals hd initial binding
    refine ⟨?_,final,?_,output,frame⟩
    · intro capacity bound
      exact (schedule_safe a Bool.false fp8 c epsilon theta order capacity initial).mp (safe capacity bound)
    · exact (schedule_run a Bool.false fp8 c epsilon theta order initial).symm.trans ran
  · intro proof vals hd initial binding
    obtain ⟨safe,final,ran,output,frame⟩:=proof vals hd initial binding
    refine ⟨?_,final,?_,output,frame⟩
    · intro capacity bound
      exact (schedule_safe a Bool.false fp8 c epsilon theta order capacity initial).mpr (safe capacity bound)
    · exact (schedule_run a Bool.false fp8 c epsilon theta order initial).trans ran

theorem equivalent {α : Type} [Inhabited α] (a : Arithmetic α) (zero : ZeroLaws a)
    (fp8 : Bool) (c : ParentArguments) (epsilon theta : α) (order : List (Nat → Nat))
    (even : ∀ phase,(c.phase phase).N%2=0)
    (apart : ∀ writer reader,(c.phase reader).X≠(c.phase writer).OUT ∧
      (c.phase reader).W≠(c.phase writer).OUT ∧ (c.phase reader).RMS≠(c.phase writer).OUT) :
    (arrayIO a fp8 c epsilon theta order).EquivalentOn
      (variant a fp8 c epsilon theta order) (IO.InputDomain fp8 c.phase) := by
  intro vals domain capacity bound initial binding
  obtain ⟨safe,final,ran,output,frame⟩:=
    IOResults.implements a zero Bool.false fp8 c.phase epsilon theta order even apart vals domain initial binding
  refine ⟨?_,?_,final,final,?_,?_,?_,frame,frame⟩
  · exact (schedule_safe a Bool.false fp8 c epsilon theta order capacity initial).mpr (safe capacity bound)
  · exact (schedule_safe a Bool.true fp8 c epsilon theta order capacity initial).mpr (safe capacity bound)
  · exact (schedule_run a Bool.false fp8 c epsilon theta order initial).trans ran
  · exact (schedule_run a Bool.true fp8 c epsilon theta order initial).trans ran
  · intro o j active
    rfl

#axiomsClean original_body
#axiomsClean candidate_body
#axiomsClean call_run
#axiomsClean call_safe
#axiomsClean implements_iff
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Pasted
