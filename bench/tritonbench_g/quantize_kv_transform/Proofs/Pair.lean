import bench.tritonbench_g.quantize_kv_transform.Proofs.NativeIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.Pair
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- The two actual programs have identical complete successful states. -/
theorem kernel_run {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (width : 0<p.width) :
    exec (KernelExecution.backend rest convert equal) (KernelSource.addressEarly p) s =
      exec (KernelExecution.backend rest convert equal) (kernel p) s := by
  simp [kernel, KernelSource.addressEarly, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.natLt, FP.Structural.bop, KernelExecution.backend,
    ComputeDType.eraseDType, ofFloat, toFloat, Region.cast, Option.bind,
    TileShape.axisDim, width]
  congr 2
  simp only [State.setReg]
  congr 1
  funext d sh name
  by_cases ho : name="o_ptrs"
  · subst name; simp only [String.reduceEq, if_false, if_true]
  by_cases hs : name="os_ptrs"
  · subst name; simp only [String.reduceEq, if_false, if_true]
  simp only [ho, hs, if_false]

theorem kernel_safe {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (width : 0<p.width)
    (bounds : RegionName → Nat) :
    TraceSafe (KernelExecution.backend rest convert equal) bounds (KernelSource.addressEarly p) s ↔
      TraceSafe (KernelExecution.backend rest convert equal) bounds (kernel p) s := by
  simp (maxSteps := 1600000) [FP.Structural.TraceSafe, kernel, KernelSource.addressEarly,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, store, Option.bind,
    ComputeDType.eraseDType, numeric, numericLt, numericEq, bop, natLt, KernelExecution.backend,
    ofFloat, toFloat, Region.cast, State.setReg, TileShape.axisDim, width]

def invocation {α : Type} (A : Algebra α) (p : Parameters) (token : Nat) : Invocation α :=
  { IO.launch A p token with kernel := KernelSource.addressEarly p }

theorem invocation_run {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (width : 0<p.width) (token : Nat) (s : State α) :
    (invocation (KernelExecution.backend rest convert equal) p token).run s =
      (IO.launch (KernelExecution.backend rest convert equal) p token).run s :=
  kernel_run rest convert equal p _ width

theorem invocation_safe {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (width : 0<p.width) (token : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    (invocation (KernelExecution.backend rest convert equal) p token).Safe bounds s ↔
      (IO.launch (KernelExecution.backend rest convert equal) p token).Safe bounds s :=
  kernel_safe rest convert equal p _ width bounds

theorem schedule_run {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (width : 0<p.width) (order : List Nat) (s : State α) :
    runSchedule (order.map (invocation (KernelExecution.backend rest convert equal) p)) s =
      runSchedule (IO.schedule (KernelExecution.backend rest convert equal) p order) s := by
  induction order generalizing s with
  | nil => rfl
  | cons token rest ih =>
    simp only [IO.schedule, List.map_cons, runSchedule, invocation_run, width, ih]

theorem schedule_safe {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (width : 0<p.width) (order : List Nat)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (order.map (invocation (KernelExecution.backend rest convert equal) p)) s ↔
      ScheduleSafe bounds (IO.schedule (KernelExecution.backend rest convert equal) p order) s := by
  induction order generalizing s with
  | nil => rfl
  | cons token rest ih =>
    simp only [IO.schedule, List.map_cons, ScheduleSafe, invocation_safe, invocation_run, width, ih]

def variant {α : Type} (A : Algebra α) (p : Parameters) (count : Nat) (order : List Nat) :
    ProgramIO.Variant (IO.quantizeIO A p count order) where
  schedule := fun _ => order.map (invocation A p)
  observe := (IO.quantizeIO A p count order).observe

open KernelNativeProfile
local instance instPairWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def original (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (count : Nat) (order : List Nat) :=
  ProgramIO.Implementation.original (IO.quantizeIO (backend a rest) p count order)
    (IO.Destinations p count)

def addressEarly (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (count : Nat) (order : List Nat) :=
  ProgramIO.Implementation.ofVariant (variant (backend a rest) p count order) (IO.Destinations p count)

/-- Every launch, output channel and exterior cell is retained. -/
theorem equivalent (a : Arithmetic) (rest : Algebra Word) (p : Parameters)
    (width : 0 < p.width) (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count)) :
    (IO.quantizeIO (backend a rest) p count order).EquivalentOn
      (variant (backend a rest) p count order) (IO.Destinations p count) := by
  intro values domain bounds bound initial binding
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    NativeIO.implements a rest p width headCover colCover count order coverage
      values domain bounds bound initial binding
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_safe (algebra a rest) PtxSigned.signedByte a.equal p width order bounds initial).mpr safe
  · exact (schedule_run (algebra a rest) PtxSigned.signedByte a.equal p width order initial).trans ran
  · intro o j active
    rfl

#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.QuantizeKvTransform.Pair
