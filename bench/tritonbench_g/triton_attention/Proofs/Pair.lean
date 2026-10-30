import bench.tritonbench_g.triton_attention.PairIO
import bench.tritonbench_g.triton_attention.Proofs.PublicCorrectness
import VeriTile.Spec
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Pair
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 32000

def after (c : Shape) := (preprocessKernel c).surfaceBody.drop 1
def originalSite (c : Shape) :=
  ((preprocessKernel c).surfaceBody[0]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (c : Shape) :=
  ((CandidateSource.preprocessKernel c).surfaceBody[0]? : Option ComputeStmt).getD (.effectMarker "unreachable")
theorem original_code (c : Shape) :
    (preprocessKernel c).surfaceBody = originalSite c :: after c := rfl
theorem shifted_code (c : Shape) :
    (CandidateSource.preprocessKernel c).surfaceBody = shiftedSite c :: after c := rfl

theorem site_run {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α) :
    step M (originalSite c) s=step M (shiftedSite c) s := by
  simp only [originalSite,shiftedSite,preprocessKernel,preParameters,preSource,preprocessTyped,
    CandidateSource.preprocessKernel,ComputeKernel.surfaceBody,List.getElem?_cons_zero,Option.getD_some]
  simp [step,evalExpr,evalOp_unfold,numeric,FP.Structural.bop,Nat.shiftLeft_eq,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals unfold FP.Structural.bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.shiftLeft_eq]

theorem site_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (c : Shape) (s : State α) :
    Safety.stepSafe M bounds (originalSite c) s ∧ Safety.stepSafe M bounds (shiftedSite c) s := by
  simp only [originalSite,shiftedSite,preprocessKernel,preParameters,preSource,preprocessTyped,
    CandidateSource.preprocessKernel,ComputeKernel.surfaceBody,List.getElem?_cons_zero,Option.getD_some]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,and_true,true_and]

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α) :
    exec M (preprocessKernel c) s=exec M (CandidateSource.preprocessKernel c) s := by
  change run M (preprocessKernel c).surfaceBody s=run M (CandidateSource.preprocessKernel c).surfaceBody s
  rw [original_code,shifted_code]
  simp only [run,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (c : Shape) (s : State α) :
    TraceSafe M bounds (preprocessKernel c) s ↔ TraceSafe M bounds (CandidateSource.preprocessKernel c) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe M bounds c s).1,(site_safe M bounds c s).2,true_and,site_run]

theorem source_run {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (kind : Native.Kind) (c : Shape) (scale : ℝ) (pid : Grid.Program) (s : State α) :
    (PublicIO.sourceInvocation rest P kind c scale pid).run s=
      (sourceInvocation rest P kind c scale pid).run s := by
  cases kind
  · rfl
  · change exec (backend rest P) (preprocessKernel c) _=exec (backend rest P) (CandidateSource.preprocessKernel c) _
    exact kernel_run _ c _
  · rfl

theorem source_safe {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (kind : Native.Kind) (c : Shape) (scale : ℝ) (pid : Grid.Program) (bounds : RegionName → Nat) (s : State α) :
    (PublicIO.sourceInvocation rest P kind c scale pid).Safe bounds s ↔
      (sourceInvocation rest P kind c scale pid).Safe bounds s := by
  cases kind
  · rfl
  · change TraceSafe (backend rest P) bounds (preprocessKernel c) _ ↔
      TraceSafe (backend rest P) bounds (CandidateSource.preprocessKernel c) _
    exact kernel_safe _ bounds c _
  · rfl

theorem invocation_run (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program)
    (mem : RegionName → Nat → Cell Native.Graph) :
    (PublicIO.nativeInvocation one kind c scale owners pid).run mem=
      (invocation one kind c scale owners pid).run mem := by
  unfold MaterializedInvocation.run
  change ((PublicIO.sourceInvocation _ _ kind c scale pid).run _).map _ =
    ((sourceInvocation _ _ kind c scale pid).run _).map _
  rw [source_run]
  rfl

theorem invocation_safe (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program)
    (bounds : RegionName → Nat) (mem : RegionName → Nat → Cell Native.Graph) :
    (PublicIO.nativeInvocation one kind c scale owners pid).Safe bounds mem ↔
      (invocation one kind c scale owners pid).Safe bounds mem := by
  change (PublicIO.sourceInvocation _ _ kind c scale pid).Safe bounds _ ↔
    (sourceInvocation _ _ kind c scale pid).Safe bounds _
  exact source_safe _ _ _ _ _ _ _ _

theorem stage_run (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (work : List Grid.Program)
    (mem : RegionName → Nat → Cell Native.Graph) :
    runMaterializedSchedule (work.map (PublicIO.nativeInvocation one kind c scale owners)) mem=
      runMaterializedSchedule (work.map (invocation one kind c scale owners)) mem := by
  induction work generalizing mem with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runMaterializedSchedule,←invocation_run]
    cases he : (PublicIO.nativeInvocation one kind c scale owners pid).run mem with
    | none => rfl
    | some next => exact ih next

theorem stage_safe (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (work : List Grid.Program)
    (bounds : RegionName → Nat) (mem : RegionName → Nat → Cell Native.Graph) :
    MaterializedScheduleSafe bounds (work.map (PublicIO.nativeInvocation one kind c scale owners)) mem ↔
      MaterializedScheduleSafe bounds (work.map (invocation one kind c scale owners)) mem := by
  induction work generalizing mem with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,MaterializedScheduleSafe,←invocation_run,←invocation_safe]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)

theorem schedule_run (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (mem : RegionName → Nat → Cell Native.Graph) (xs : PublicIO.Inputs Native.Word c) :
    runMaterializedSchedule ((PublicIO.nativeIO scalar c owners I literal fw pw bw).schedule xs) mem=
      runMaterializedSchedule (schedule scalar c owners fw pw bw) mem := by
  simp only [PublicIO.nativeIO,PublicIO.layout,schedule,runMaterializedSchedule_append,stage_run]

theorem schedule_safe (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (bounds : RegionName → Nat) (mem : RegionName → Nat → Cell Native.Graph) (xs : PublicIO.Inputs Native.Word c) :
    MaterializedScheduleSafe bounds ((PublicIO.nativeIO scalar c owners I literal fw pw bw).schedule xs) mem ↔
      MaterializedScheduleSafe bounds (schedule scalar c owners fw pw bw) mem := by
  simp only [PublicIO.nativeIO,PublicIO.layout,schedule,materializedScheduleSafe_append,runMaterializedSchedule_append,stage_safe,stage_run]

theorem equivalent (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    Implementation.Equivalent (original scalar c owners I literal fw pw bw)
      (shifted scalar c owners I literal fw pw bw) := by
  intro xs admitted initial pinned
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    PublicIO.native_implements scalar c owners I literal fw pw bw hf hp hb xs admitted initial pinned
  refine ⟨?_,final,final,ran,?_,?_,frame,frame⟩
  · intro bounds backed
    exact ⟨safe bounds backed,(schedule_safe scalar c owners I literal fw pw bw bounds initial xs).mp (safe bounds backed)⟩
  · change runMaterializedSchedule (schedule scalar c owners fw pw bw) initial=some final
    rw [←schedule_run scalar c owners I literal fw pw bw initial xs]
    exact ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Pair
