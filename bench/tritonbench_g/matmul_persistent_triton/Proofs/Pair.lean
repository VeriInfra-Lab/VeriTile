import bench.tritonbench_g.matmul_persistent_triton.IO
import bench.tritonbench_g.matmul_persistent_triton.Proofs.IOBounds
import bench.tritonbench_g.matmul_persistent_triton.Proofs.Kernels
import VeriTile.Triton.Float.Loop
import VeriTile.Triton.Float.Equivalence

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Pair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open Core
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Expanding the task macro into ordinary quotations preserved the complete
selected original AST, for both storage profiles and every parameter. -/
theorem selected_original_preserved (storage : TypedSource.Storage) :
    TypedSource.kernel storage=
      VeriTile.Bench.Repairs.MatmulPersistentTriton.TypedSource.kernel storage := by
  cases storage <;> rfl

/-- Proof-only extraction from the complete already-declared quotations. -/
def candidateOuter (p : Parameters) : List ComputeStmt :=
  match (candidate p).surfaceBody[15]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def row (p : Parameters) : ComputeStmt :=
  ((outer p)[6]?).getD (.ifThen (.constBool Bool.false) [])
def candidateRow (p : Parameters) : ComputeStmt :=
  ((candidateOuter p)[6]?).getD (.ifThen (.constBool Bool.false) [])
def tilePrefix (p : Parameters) := (outer p).take 6
def tileTail (p : Parameters) := (outer p).drop 7

theorem original_outer_decomposition (p : Parameters) :
    outer p=tilePrefix p++row p::tileTail p := by
  rcases p with ⟨⟨m,n,k,sms,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile <;> rfl
theorem candidate_outer_decomposition (p : Parameters) :
    candidateOuter p=tilePrefix p++candidateRow p::tileTail p := by
  rcases p with ⟨⟨m,n,k,sms,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile <;> rfl

/-- Exact signed integer multiplication, independent of all floating primitives. -/
theorem row_step_equal (M : Algebra α) (p : Parameters) (s : State α) :
    step M (row p) s=step M (candidateRow p) s := by
  cases hp : p.problem.profile <;>
    simp [row,candidateRow,outer,source,candidateOuter,candidate,
      TypedSource.kernel,TypedSource.reordered,TypedSource.fp16,TypedSource.f8e4,
      TypedSource.fp16Reordered,TypedSource.f8e4Reordered,ComputeKernel.surfaceBody,
      step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,Int.mul_comm,hp]
  all_goals cases hs : s.regs .int [] "pid_m" <;>
    simp [hs,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]
  all_goals
    apply congrArg (fun v => s.setReg "start_m" .int [] v)
    funext i
    exact Int.mul_comm _ _

theorem row_safety_equal (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.stepSafe M bounds (row p) s ↔ Safety.stepSafe M bounds (candidateRow p) s := by
  cases hp : p.problem.profile <;>
    simp [row,candidateRow,outer,source,candidateOuter,candidate,
      TypedSource.kernel,TypedSource.reordered,TypedSource.fp16,TypedSource.f8e4,
      TypedSource.fp16Reordered,TypedSource.f8e4Reordered,ComputeKernel.surfaceBody,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,hp]

theorem tile_equal (M : Algebra α) (p : Parameters) (s : State α) :
    run M (outer p) s=run M (candidateOuter p) s := by
  rw [original_outer_decomposition,candidate_outer_decomposition,run_append,run_append]
  cases hr : run M (tilePrefix p) s with
  | none => rfl
  | some t =>
    rw [Option.bind_some,Option.bind_some,run,run]
    rw [row_step_equal]

theorem tile_safety_equal (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (outer p) s ↔
      Safety.traceSafeList M bounds (candidateOuter p) s := by
  rw [original_outer_decomposition,candidate_outer_decomposition,
    Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons,row_safety_equal,row_step_equal]

theorem range_equal (M : Algebra α) (p : Parameters)
    (start stop stride : Nat) (s : State α) :
    FP.Structural.range M "tile_ordinal" start stop stride (outer p) s=
      FP.Structural.range M "tile_ordinal" start stop stride (candidateOuter p) s := by
  apply range_congr_on_reachable
  intro cur t _ _ _
  exact tile_equal M p _

theorem range_safety_equal (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat)
    (start stop stride : Nat) (s : State α) :
    Safety.rangeSafe M bounds "tile_ordinal" start stop stride (outer p) s ↔
      Safety.rangeSafe M bounds "tile_ordinal" start stop stride (candidateOuter p) s := by
  induction start using (measure (fun i => stop-i)).wf.induction generalizing s with
  | h cur ih =>
    rw [Safety.rangeSafe_unfold,Safety.rangeSafe_unfold]
    by_cases hz : stride=0
    · simp [hz]
    simp only [hz,if_false]
    by_cases hc : cur<stop
    · simp only [hc,if_true,tile_safety_equal,← tile_equal]
      cases hr : run M (outer p) (s.setReg "tile_ordinal" .nat [] (fun _ => cur)) with
      | none => rfl
      | some t => exact and_congr Iff.rfl (ih (cur+stride) (by simp_wf;omega) t)
    · simp [hc]

theorem outer_step_equal (M : Algebra α) (p : Parameters) (s : State α) :
    step M (outerStatement p) s=
      step M (.forRangeDyn "tile_ordinal" (.constNat 0) (.ref .nat [] "tiles_per_SM")
        (.constNat 1) (candidateOuter p)) s := by
  have hs : outerStatement p=.forRangeDyn "tile_ordinal" (.constNat 0)
      (.ref .nat [] "tiles_per_SM") (.constNat 1) (outer p) := by
    rcases p with ⟨⟨m,n,k,sms,profile⟩,am,ak,bk,bn,cm,cn⟩
    cases profile <;> rfl
  rw [hs]
  simp only [step,evalOp_unfold,Option.bind_some]
  cases hr : s.regs .nat [] "tiles_per_SM" with
  | none => rfl
  | some count => exact range_equal M p 0 (count ()) 1 s

theorem outer_safety_equal (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.stepSafe M bounds (outerStatement p) s ↔
      Safety.stepSafe M bounds
        (.forRangeDyn "tile_ordinal" (.constNat 0) (.ref .nat [] "tiles_per_SM")
          (.constNat 1) (candidateOuter p)) s := by
  have hs : outerStatement p=.forRangeDyn "tile_ordinal" (.constNat 0)
      (.ref .nat [] "tiles_per_SM") (.constNat 1) (outer p) := by
    rcases p with ⟨⟨m,n,k,sms,profile⟩,am,ak,bk,bn,cm,cn⟩
    cases profile <;> rfl
  rw [hs]
  simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,Option.bind_some]
  cases hr : s.regs .nat [] "tiles_per_SM" with
  | none => rfl
  | some count => simpa using range_safety_equal M p bounds 0 (count ()) 1 s

theorem candidate_decomposition (p : Parameters) :
    (candidate p).surfaceBody=prelude p++
      [.forRangeDyn "tile_ordinal" (.constNat 0) (.ref .nat [] "tiles_per_SM")
        (.constNat 1) (candidateOuter p)] := by
  rcases p with ⟨⟨m,n,k,sms,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile <;> rfl

theorem kernel_equal (M : Algebra α) (p : Parameters) (s : State α) :
    exec M (source p) s=exec M (candidate p) s := by
  change run M (source p).surfaceBody s=run M (candidate p).surfaceBody s
  rw [source_decomposition,candidate_decomposition,run_append,run_append]
  cases hr : run M (prelude p) s with
  | none => rfl
  | some t =>
    rw [Option.bind_some,Option.bind_some,run,run]
    rw [outer_step_equal]

theorem kernel_safety_equal (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (source p) s ↔ TraceSafe M bounds (candidate p) s := by
  unfold TraceSafe
  rw [source_decomposition,candidate_decomposition,
    Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons,outer_safety_equal,outer_step_equal]

/-- Construct a successful tile execution independently of any output formula
or numerical zero/conversion identity. The loop and register invariants are
the existing source execution invariants. -/
theorem tile_terminates (P : Source.Arithmetic α α) (p : Parameters)
    (initial : State α) (pid ordinal : Nat) (regs : Registers p pid initial)
    (tileBound : p.problem.tile pid ordinal<p.problem.tiles) :
    ∃ final,run (backend P) (outer p)
        (initial.setReg "tile_ordinal" .nat [] (fun _ => ordinal))=some final ∧
      Registers p pid final := by
  let setup := tilePreludeExecution P p pid ordinal initial regs
  have invariant := tile_initial P p pid ordinal initial regs tileBound
  have registers := tile_registers P p pid ordinal initial regs
  obtain ⟨last,ran,inv,reg⟩ := innerLoop P p initial setup.val pid (p.problem.tile pid ordinal) 0
    (Nat.zero_le _) invariant registers
  refine ⟨suffixResult P p initial last (p.problem.tile pid ordinal),?_,
    suffix_registers P p initial last pid (p.problem.tile pid ordinal) reg⟩
  rw [outer_decomposition,run_append,setup.property,Option.bind_some,run,
    dynamicInner P p setup.val pid registers,ran]
  exact suffixExecution P p initial last _ inv

private theorem sequence_terminates (one : Nat → State α → Option (State α))
    (ready : State α → Prop) (ids : List Nat) (legal : Nat → Prop)
    (valid : ∀ id∈ids,legal id)
    (single : ∀ id,legal id → ∀ s,ready s → ∃ t,one id s=some t ∧ ready t)
    (s : State α) (hs : ready s) :
    ∃ t,VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence one ids s=
      some t ∧ ready t := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl,hs⟩
  | cons id rest ih =>
    obtain ⟨mid,hm,hr⟩ := single id (valid id (by simp)) s hs
    obtain ⟨final,hf,fr⟩ := ih (fun i hi => valid i (by simp [hi])) mid hr
    refine ⟨final,?_,fr⟩
    simp only [VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence,hm,Option.bind_some,hf]

theorem program_terminates (P : Source.Arithmetic α α) (p : Parameters)
    (sms : 0<p.problem.SMS) (pid : Nat) (valid : pid<p.problem.grid) (initial : State α) :
    ∃ final,program P p pid initial=some final := by
  open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution in
  let prepared := preludeExecution P p (atProgram initial pid)
  have regs : Registers p pid prepared.val := by
    simpa [VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.atProgram]
      using (prelude_invariant P p
        (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.atProgram initial pid)).1
  have single : ∀ ordinal,ordinal<p.problem.work pid → ∀ current,Registers p pid current →
      ∃ final,Compose.ordinal (backend P) "tile_ordinal" (outer p) ordinal current=some final ∧
        Registers p pid final := by
    intro ordinal ho current ready
    have tileBound : p.problem.tile pid ordinal<p.problem.tiles :=
      (Source.tiles_exact p.problem sms _).mp
        ((Source.tile_mem p.problem _).mpr ⟨pid,valid,ordinal,ho,rfl⟩)
    exact tile_terminates P p current pid ordinal ready tileBound
  obtain ⟨final,ran,_⟩ := sequence_terminates
    (Compose.ordinal (backend P) "tile_ordinal" (outer p)) (Registers p pid)
    (List.range' 0 (p.problem.work pid)) (fun ordinal => ordinal<p.problem.work pid)
    (by intro ordinal ho;simpa [List.mem_range'] using ho) single prepared.val regs
  refine ⟨final,?_⟩
  unfold program
  change run (backend P) (source p).surfaceBody _=some final
  rw [source_decomposition,run_append,prepared.property,Option.bind_some,run,
    dynamicOuter P p prepared.val pid regs,Compose.range_sequence]
  simp only [Nat.sub_zero,ran,run]
  rfl

theorem job_equal (P : Source.Arithmetic α α) (p : Parameters) (pid : Nat) (s : State α) :
    (PublicData.job P p pid).run s=(job P p pid).run s :=
  kernel_equal (backend P) p _

theorem job_safety_equal (P : Source.Arithmetic α α) (p : Parameters)
    (pid : Nat) (bounds : RegionName → Nat) (s : State α) :
    (PublicData.job P p pid).Safe bounds s ↔ (job P p pid).Safe bounds s :=
  kernel_safety_equal (backend P) p bounds _

theorem schedule_equal (P : Source.Arithmetic α α) (p : Parameters)
    (ids : List Nat) (s : State α) :
    runSchedule (ids.map (PublicData.job P p)) s=runSchedule (ids.map (job P p)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,job_equal]
    cases hr : (job P p pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safety_equal (P : Source.Arithmetic α α) (p : Parameters)
    (ids : List Nat) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (ids.map (PublicData.job P p)) s ↔
      ScheduleSafe bounds (ids.map (job P p)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,ScheduleSafe,job_equal]
    exact and_congr (job_safety_equal P p pid bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

theorem schedule_terminates (P : Source.Arithmetic α α) (p : Parameters)
    (sms : 0<p.problem.SMS) (ids : List Nat) (valid : ∀ pid∈ids,pid<p.problem.grid)
    (s : State α) : ∃ final,runSchedule (ids.map (PublicData.job P p)) s=some final := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons pid rest ih =>
    obtain ⟨mid,hm⟩ := program_terminates P p sms pid (valid pid (by simp)) s
    obtain ⟨final,hf⟩ := ih (fun id hi => valid id (by simp [hi])) mid
    refine ⟨final,?_⟩
    change (Invocation.run (PublicData.job P p pid) s).bind
      (runSchedule (rest.map (PublicData.job P p)))=some final
    rw [show (PublicData.job P p pid).run s=some mid from hm]
    exact hf

theorem implements_transfer (P : Source.Arithmetic α α) (p : Parameters)
    (f : ChannelValues α (PublicData.inputs p) → ChannelValues α (PublicData.outputs p))
    (original : (PublicData.io P p).Implements f) : (candidateIO P p).Implements f := by
  intro vals hv bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ := original vals hv bounds hb initial hin
  refine ⟨(schedule_safety_equal P p _ bounds initial).mp safe,final,?_,outputs,frame⟩
  exact (schedule_equal P p _ initial).symm.trans ran

theorem Program.schedule_io (p : Program α) (vals : p.io.Inputs) :
    p.io.schedule vals=p.schedule := by
  cases p with | mk P p sms reordered => cases reordered <;> rfl

def Program.run (p : Program α) (s : State α) : Option (State α) :=
  runSchedule p.schedule s

def Program.Safe (p : Program α) (bounds : RegionName → Nat) (s : State α) : Prop :=
  ScheduleSafe bounds p.schedule s

/-- Both actual complete persistent launches succeed to the same typed state,
and have identical reached access-safety conditions for every region bound. -/
def Equivalent (lhs rhs : Program α) : Prop := ∀ initial,∃ final,
  lhs.run initial=some final ∧ rhs.run initial=some final ∧
    ∀ bounds,lhs.Safe bounds initial ↔ rhs.Safe bounds initial

instance : Spec.ProgramSyntax (Program α) where
  Statement := ComputeStmt
  Signature := Source.Arithmetic α α × Parameters
  signature p := (p.arithmetic,p.parameters)
  body p := p.kernel.surfaceBody
  structural := some Equivalent

def original (P : Source.Arithmetic α α) (p : Parameters) (sms : 0<p.problem.SMS) : Program α :=
  ⟨P,p,sms,Bool.false⟩
def reordered (P : Source.Arithmetic α α) (p : Parameters) (sms : 0<p.problem.SMS) : Program α :=
  ⟨P,p,sms,Bool.true⟩

theorem equivalent (P : Source.Arithmetic α α) (p : Parameters) (sms : 0<p.problem.SMS) :
    Equivalent (original P p sms) (reordered P p sms) := by
  intro initial
  obtain ⟨final,ran⟩ := schedule_terminates P p sms (List.range p.problem.grid)
    (fun pid hp => List.mem_range.mp hp) initial
  refine ⟨final,ran,(schedule_equal P p _ initial).symm.trans ran,?_⟩
  intro bounds
  exact schedule_safety_equal P p _ bounds initial

#axiomsClean kernel_equal
#axiomsClean kernel_safety_equal
#axiomsClean schedule_terminates
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Pair
