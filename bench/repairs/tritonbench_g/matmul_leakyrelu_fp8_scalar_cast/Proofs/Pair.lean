import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.IOResults
import VeriTile.Triton.Float.ScalarOpsLaws
import VeriTile.Triton.Float.Equivalence

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Pair
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]


/-- Proof-only decomposition of two already complete ordinary quotations. -/
def commonPrefix (p : Problem) (b : Buffers) := (source p b).surfaceBody.take 21
def suffix (p : Problem) (b : Buffers) := (candidate p b).surfaceBody.drop 21

theorem original_decomposition (p : Problem) (b : Buffers) :
    (source p b).surfaceBody = commonPrefix p b ++ epilogue p b := rfl

theorem candidate_decomposition (p : Problem) (b : Buffers) :
    (candidate p b).surfaceBody = commonPrefix p b ++ suffix p b := rfl

/-- Only the multiplication primitive is constrained. This includes all
exceptional FP32 words and imposes no input or whole-program equality premise. -/
def MulCommutes (M : Algebra α) : Prop := ∀ x y,
  M.binary (some .fp32) .real .mul x y = M.binary (some .fp32) .real .mul y x

theorem mul_eval_equal (M : Algebra α) (D : DotAcc α) (h : MulCommutes M)
    (s : State α) :
    evalOp (backend M D) (some .fp32)
      (.mul .real .scalarL (.const 0.01) (.ref .real [128,256] "accumulator")) s =
    evalOp (backend M D) (some .fp32)
      (.mul .real .scalarR (.ref .real [128,256] "accumulator") (.const 0.01)) s := by
  simp only [evalOp_unfold]
  cases ha : s.regs .real [128,256] "accumulator" with
  | none => rfl
  | some a =>
    try simp only [ha,Option.bind_some]
    change some (fun i : CIndex => M.binary (some .fp32) .real .mul
      (M.literal (some .fp32) .real 0.01) (a i)) =
      some (fun i : CIndex => M.binary (some .fp32) .real .mul
        (a i) (M.literal (some .fp32) .real 0.01))
    exact congrArg some (funext (fun i => h _ _))

theorem epilogue_equal (M : Algebra α) (D : DotAcc α) (h : MulCommutes M)
    (p : Problem) (b : Buffers) (s : State α) :
    run (backend M D) (epilogue p b) s = run (backend M D) (suffix p b) s := by
  have hm := mul_eval_equal M D h s
  simp only [epilogue,suffix,source,candidate,TypedSource.kernel,TypedSource.rightSlope,
    ComputeKernel.surfaceBody,List.drop]
  simp only [run,step,evalExpr,evalComputeOp,evalOp_unfold]
  simp only [evalOp_unfold] at hm
  erw [hm]
  rfl

theorem epilogue_safety_equal (M : Algebra α) (D : DotAcc α) (h : MulCommutes M)
    (p : Problem) (b : Buffers) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend M D) bounds (epilogue p b) s ↔
      Safety.traceSafeList (backend M D) bounds (suffix p b) s := by
  have hm := mul_eval_equal M D h s
  simp only [epilogue,suffix,source,candidate,TypedSource.kernel,TypedSource.rightSlope,
    ComputeKernel.surfaceBody,List.drop]
  simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold]
  simp only [evalOp_unfold] at hm
  erw [hm]
  rfl

/-- Complete kernel execution, including every K iteration and both branches. -/
theorem kernel_equal (M : Algebra α) (D : DotAcc α) (h : MulCommutes M)
    (p : Problem) (b : Buffers) (s : State α) :
    exec (backend M D) (source p b) s = exec (backend M D) (candidate p b) s := by
  change run (backend M D) (source p b).surfaceBody s =
    run (backend M D) (candidate p b).surfaceBody s
  rw [original_decomposition,candidate_decomposition,run_append,run_append]
  cases hr : run (backend M D) (commonPrefix p b) s with
  | none => rfl
  | some t => exact epilogue_equal M D h p b t

theorem kernel_safety_equal (M : Algebra α) (D : DotAcc α) (h : MulCommutes M)
    (p : Problem) (b : Buffers) (bounds : RegionName → Nat) (s : State α) :
    TraceSafe (backend M D) bounds (source p b) s ↔
      TraceSafe (backend M D) bounds (candidate p b) s := by
  unfold TraceSafe
  rw [original_decomposition,candidate_decomposition,
    Safety.traceSafeList_append,Safety.traceSafeList_append]
  exact and_congr_right (fun _ => forall_congr' (fun t =>
    imp_congr_right (fun _ => epilogue_safety_equal M D h p b bounds t)))


/-- All public channels, storage, masks and the complete grouped launch order
are inherited verbatim; only the declared source kernel changes. -/

theorem invocation_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (pid : Nat) (s : State α) :
    (IO.invocation M ops g b pid).run s = (invocation M ops g b pid).run s :=
  kernel_equal M (selectedDot ops) h g.problem (b.views g) _

theorem invocation_safety_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (pid : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    (IO.invocation M ops g b pid).Safe bounds s ↔
      (invocation M ops g b pid).Safe bounds s :=
  kernel_safety_equal M (selectedDot ops) h g.problem (b.views g) bounds _

theorem schedule_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (ids : List Nat) (s : State α) :
    runSchedule (ids.map (IO.invocation M ops g b)) s =
      runSchedule (ids.map (invocation M ops g b)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [List.map_cons,runSchedule,invocation_equal M ops h]
    cases hr : (invocation M ops g b pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safety_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (ids : List Nat)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (ids.map (IO.invocation M ops g b)) s ↔
      ScheduleSafe bounds (ids.map (invocation M ops g b)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_equal M ops h]
    exact and_congr (invocation_safety_equal M ops h g b pid bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Success is constructed from the complete source execution theorem. It does
not follow merely from equal failed runs, nor require a numerical formula law. -/
theorem schedule_terminates (M : Algebra α) (ops : Operations α)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b)
    (ids : List Nat) (legal : ∀ pid∈ids,pid<g.problem.tiles) (s : State α) :
    ∃ final, runSchedule (ids.map (IO.invocation M ops g b)) s=some final := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons pid tail ih =>
    let start := (IO.invocation M ops g b pid).initial s
    have hp : start.pids 0=pid := by simp [start,Invocation.initial,IO.invocation,bindRegisters]
    have ht : start.pids 0<g.problem.tiles := hp ▸ legal pid (by simp)
    obtain ⟨next,ran,_⟩ := programExecution M (selectedDot ops) g.problem (b.views g) start
      (active_injective g.problem (b.views g) L start ht)
    obtain ⟨final,hf⟩ := ih (fun p hp => legal p (by simp [hp])) next
    refine ⟨final,?_⟩
    change (Invocation.run (IO.invocation M ops g b pid) s).bind
      (runSchedule (tail.map (IO.invocation M ops g b)))=some final
    rw [show Invocation.run (IO.invocation M ops g b pid) s=some next from ran]
    exact hf

/-- The equality transports every typed logical output and exact raw frame,
as well as trace safety. It introduces no target-specific numerical premise. -/
theorem implements_transfer (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers)
    (f : ChannelValues α (IO.inputs g) → ChannelValues α (IO.outputs g))
    (original : (IO.matmulIO M ops g b).Implements f) :
    (candidateIO M ops g b).Implements f := by
  intro vals hv bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ := original vals hv bounds hb initial hin
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (schedule_safety_equal M ops h g b _ bounds initial).mp safe
  · exact (schedule_equal M ops h g b _ initial).symm.trans ran

/-- The strict concrete multiplier is shared software theory. All unchanged
primitive implementations, ordered dot instructions and casts stay explicit. -/
abbrev Word := VeriTile.Triton.FP.Value .fp32

def fp32Algebra (cfg : FP.Config) (rest : Algebra Word) : Algebra Word :=
  { rest with binary := fun precision dtype op x y =>
      match precision,dtype,op with
      | some .fp32,.real,.mul => FP.mul cfg x y
      | _,_,_ => rest.binary precision dtype op x y }

theorem fp32_mul_commutes (cfg : FP.Config) (rest : Algebra Word) :
    MulCommutes (fp32Algebra cfg rest) := by
  intro x y
  exact FP.ScalarOpsLaws.mul_comm cfg x y

structure Program (α : Type) where
  algebra : Algebra α
  operations : Operations α
  geometry : KernelBuffers.Geometry
  buffers : KernelBuffers.Buffers
  layout : KernelBuffers.Layout geometry buffers
  rightSlope : Bool

def Program.kernel (p : Program α) : ComputeKernel :=
  if p.rightSlope then candidate p.geometry.problem (p.buffers.views p.geometry)
  else source p.geometry.problem (p.buffers.views p.geometry)

def Program.io (p : Program α) : ProgramIO α α :=
  if p.rightSlope then candidateIO p.algebra p.operations p.geometry p.buffers
  else IO.matmulIO p.algebra p.operations p.geometry p.buffers

def Program.schedule (p : Program α) : List (Invocation α) :=
  if p.rightSlope then
    (Source.tiles p.geometry.problem).map (invocation p.algebra p.operations p.geometry p.buffers)
  else (Source.tiles p.geometry.problem).map (IO.invocation p.algebra p.operations p.geometry p.buffers)

theorem Program.schedule_io (p : Program α) (vals : p.io.Inputs) :
    p.io.schedule vals=p.schedule := by
  cases p with
  | mk M ops g b L right => cases right <;> rfl

def Program.run (p : Program α) (s : State α) : Option (State α) :=
  runSchedule p.schedule s

def Program.Safe (p : Program α) (bounds : RegionName → Nat) (s : State α) : Prop :=
  ScheduleSafe bounds p.schedule s

/-- Actual full-grid successful execution to the same entire typed state,
and the same reached access-safety obligations for every memory window. -/
def Equivalent (lhs rhs : Program α) : Prop := ∀ initial, ∃ final,
  lhs.run initial=some final ∧ rhs.run initial=some final ∧
    ∀ bounds, lhs.Safe bounds initial ↔ rhs.Safe bounds initial

instance : Spec.ProgramSyntax (Program α) where
  Statement := ComputeStmt
  Signature := Algebra α × Operations α × KernelBuffers.Geometry × KernelBuffers.Buffers
  signature p := (p.algebra,p.operations,p.geometry,p.buffers)
  body p := p.kernel.surfaceBody
  structural := some Equivalent

def original (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b) : Program α :=
  ⟨M,ops,g,b,L,Bool.false⟩

def rightSlope (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b) : Program α :=
  ⟨M,ops,g,b,L,Bool.true⟩

theorem equivalent (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b) :
    Equivalent (original M ops g b L) (rightSlope M ops g b L) := by
  intro initial
  obtain ⟨final,ran⟩ := schedule_terminates M ops g b L (Source.tiles g.problem)
    (fun pid hp => (Source.tiles_exact g.problem pid).mp hp) initial
  refine ⟨final,ran,?_,?_⟩
  · exact (schedule_equal M ops h g b _ initial).symm.trans ran
  · intro bounds
    exact schedule_safety_equal M ops h g b _ bounds initial

#axiomsClean kernel_equal
#axiomsClean kernel_safety_equal
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Pair
