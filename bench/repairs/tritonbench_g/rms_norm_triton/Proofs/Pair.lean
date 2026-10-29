import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.SourcePreservation
import bench.repairs.tritonbench_g.rms_norm_triton.IO
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.PublicIO
namespace VeriTile.Bench.Repairs.RmsNormTriton.Pair
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole Observed
noncomputable section
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

abbrev sourceCandidate (c : Layout) (eps : ℝ) : ComputeKernel :=
  runtimeCandidate c.Y c.X c.W c.N c.B

theorem source_id (c : Layout) : ActualRow.source c = runtimeKernel c.Y c.X c.W c.N c.B := rfl

theorem run_eq {α : Type} [Inhabited α] (M : Algebra α)
    (c : Layout) (eps : ℝ) (s : State α) :
    FP.Structural.exec (engine M) (sourceCandidate c eps) s =
      FP.Structural.exec (engine M) (ActualRow.source c) s := by
  change run (engine M)
      ((sourceCandidate c eps).surfaceBody.take 3 ++ (ActualRow.source c).surfaceBody.drop 3) s =
    run (engine M)
      ((ActualRow.source c).surfaceBody.take 3 ++ (ActualRow.source c).surfaceBody.drop 3) s
  rw [run_append, run_append]
  congr 1
  simp [sourceCandidate,runtimeCandidate,ActualRow.source,runtimeKernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,bop,store,Option.bind,numeric,
    Region.cast,State.setReg,Nat.mul_comm]
  funext d sh n
  split_ifs <;> subst_vars <;> simp [FP.Structural.bop, Nat.mul_comm]
  all_goals
    funext i
    change ((⟨_⟩ : RegionName), 0+c.N*s.pids 0) = ((⟨_⟩ : RegionName), 0+s.pids 0*c.N)
    rw [Nat.mul_comm c.N (s.pids 0)]


/-- Complete state equality also holds for the checked literal-call forms. -/
theorem literal_run_eq {α : Type} [Inhabited α] (M : Algebra α)
    (c : Layout) (eps : ℝ) (s : State α) :
    FP.Structural.exec (engine M) (indexCommutedKernel c.Y c.X c.W c.N c.B eps) s =
      FP.Structural.exec (engine M) (kernel c.Y c.X c.W c.N c.B eps) s := by
  change run (engine M)
      ((indexCommutedKernel c.Y c.X c.W c.N c.B eps).surfaceBody.take 3 ++ (kernel c.Y c.X c.W c.N c.B eps).surfaceBody.drop 3) s =
    run (engine M)
      ((kernel c.Y c.X c.W c.N c.B eps).surfaceBody.take 3 ++ (kernel c.Y c.X c.W c.N c.B eps).surfaceBody.drop 3) s
  rw [run_append, run_append]
  congr 1
  simp [indexCommutedKernel,kernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,bop,store,Option.bind,numeric,
    Region.cast,State.setReg,Nat.mul_comm]
  funext d sh n
  split_ifs <;> subst_vars <;> simp [FP.Structural.bop, Nat.mul_comm]
  all_goals
    funext i
    change ((⟨_⟩ : RegionName), 0+c.N*s.pids 0) = ((⟨_⟩ : RegionName), 0+s.pids 0*c.N)
    rw [Nat.mul_comm c.N (s.pids 0)]



theorem row_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i : Fin c.B,i.val<c.N → pid*c.N+i.val<bounds c.X)
    (weight : ∀ i : Fin c.B,i.val<c.N → i.val<bounds c.W)
    (output : ∀ i : Fin c.B,i.val<c.N → pid*c.N+i.val<bounds c.Y)
    (epsilon : α := ActualRow.argument M eps) :
    TraceSafe (engine M) bounds (sourceCandidate c eps) (ActualRow.entry epsilon pid s) := by
  simp [TraceSafe,sourceCandidate,runtimeCandidate,ActualRow.entry,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,bop,natLt,rowState,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  simpa only [Nat.mul_comm c.N pid] using And.intro input (And.intro weight output)


theorem schedule_run (c : Layout) (eps : ℝ) (rows : List Nat)
    (s : State (Term (BitVec 32))) :
    runSchedule (rows.map (FloatingIO.candidateInvocation c eps)) s =
      runSchedule (rows.map (fun pid => KernelSafety.invocation graphAlgebra c eps pid (.parameter "eps"))) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule]
    have same : (FloatingIO.candidateInvocation c eps pid).run s =
        (KernelSafety.invocation graphAlgebra c eps pid (.parameter "eps")).run s :=
      run_eq graphAlgebra c eps (ActualRow.entry (.parameter "eps") pid s)
    rw [same]
    congr 1
    funext next
    exact ih next

theorem schedule_safe (c : Layout) (eps : ℝ) (M : NativeModel (BitVec 32))
    (profile : ReductionFamily) (rows : List Nat) (schedule : rows.Perm (List.range c.M))
    (vals : (FloatingIO.nativeIO c eps M profile rows).Inputs) (bounds : RegionName → Nat)
    (bound : (FloatingIO.nativeIO c eps M profile rows).InBounds bounds vals)
    (s : State (Term (BitVec 32))) :
    ScheduleSafe bounds (rows.map (FloatingIO.candidateInvocation c eps)) s := by
  have input (pid : Nat) (hp : pid∈rows) (i : Fin c.B) (hi : i.val<c.N) :
      pid*c.N+i.val<bounds c.X :=
    bound.2.1 0 (⟨pid,List.mem_range.mp (schedule.mem_iff.mp hp)⟩,⟨i.val,hi⟩,()) trivial
  have weight (i : Fin c.B) (hi : i.val<c.N) : i.val<bounds c.W :=
    bound.2.1 1 (⟨i.val,hi⟩,()) trivial
  have output (pid : Nat) (hp : pid∈rows) (i : Fin c.B) (hi : i.val<c.N) :
      pid*c.N+i.val<bounds c.Y :=
    bound.2.2.1 0 (⟨pid,List.mem_range.mp (schedule.mem_iff.mp hp)⟩,⟨i.val,hi⟩,()) trivial
  clear schedule bound vals
  induction rows generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    exact ⟨row_safe graphAlgebra c eps pid s bounds (input pid (by simp)) weight
      (output pid (by simp)) (.parameter "eps"), fun next _ =>
        ih next (fun p hp => input p (by simp [hp])) (fun p hp => output p (by simp [hp]))⟩

theorem equivalent (c : Layout) (eps : ℝ) (M : NativeModel (BitVec 32))
    (profile : ReductionFamily) (xSeparate : c.X≠c.Y) (wSeparate : c.W≠c.Y)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M)) :
    (FloatingIO.nativeIO c eps M profile rows).Equivalent
      (FloatingIO.candidate c eps M profile rows) := by
  intro vals _ bounds bound initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    GridPublic.implements_argument graphAlgebra Term.input (interpret M profile) c eps (.parameter "eps") rows schedule
      xSeparate wSeparate (reference M profile c.N c.B) (RuntimeObserved.source_graph_lowering M profile c)
      vals trivial bounds bound initial binding
  refine ⟨safe,schedule_safe c eps M profile rows schedule vals bounds bound initial,
    final,final,ran,?_,fun _ _ _ => rfl,frame,frame⟩
  exact (schedule_run c eps rows initial).trans ran

#axiomsClean literal_run_eq
#axiomsClean run_eq
#axiomsClean row_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RmsNormTriton.Pair
