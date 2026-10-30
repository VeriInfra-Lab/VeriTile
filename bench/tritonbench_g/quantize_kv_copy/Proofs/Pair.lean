import bench.tritonbench_g.quantize_kv_copy.PairIO
import bench.tritonbench_g.quantize_kv_copy.Proofs.IOGrid
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.Pair
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
set_option autoImplicit true
set_option maxHeartbeats 18000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Complete successful states agree: only independent pure assignments moved. -/
theorem kernel_run [Inhabited α] (op : Ops α) (p : Parameters) (s : State α) :
    FP.Structural.exec (backend op) (KernelSyntax.addressEarly p.groups p.block p.heads) s =
      FP.Structural.exec (backend op) (kernel p) s := by
  simp [kernel, KernelSyntax.source, KernelSyntax.addressEarly, FP.Structural.exec,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.store, FP.Structural.numeric, FP.Structural.bop, FP.Structural.natLt,
    FP.Structural.numericEq, backend, ComputeDType.eraseDType, FP.Structural.ofFloat,
    FP.Structural.toFloat, Region.cast, Option.bind, TileShape.axisDim]
  congr 2
  simp only [State.setReg]
  congr 1
  funext d sh name
  by_cases ho : name="o_ptrs"
  · subst name; simp only [String.reduceEq, if_false, if_true]
  by_cases hs : name="os_ptrs"
  · subst name; simp only [String.reduceEq, if_false, if_true]
  simp only [ho, hs, if_false]

/-- Reordering pure address calculations preserves the actual access trace. -/
theorem kernel_safe [Inhabited α] (op : Ops α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    FP.Structural.TraceSafe (backend op) bounds
      (KernelSyntax.addressEarly p.groups p.block p.heads) s ↔
    FP.Structural.TraceSafe (backend op) bounds (kernel p) s := by
  simp (maxSteps := 1600000) [FP.Structural.TraceSafe, kernel, KernelSyntax.source,
    KernelSyntax.addressEarly, ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
    Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
    Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
    Safety.addressSafe_unfold, Safety.maskActive_unfold, step, evalExpr, evalComputeOp,
    evalOp_unfold, store, Option.bind, numeric, bop, natLt, numericLt, numericEq,
    ofFloat, toFloat, Region.cast, State.setReg, backend, ComputeDType.eraseDType,
    reduction, TileShape.axisDim]


theorem invocation_run [Inhabited α] (op : Ops α) (p : Parameters) (pid : Nat) (s : State α) :
    (invocation op p pid).run s = (PublicIO.invocation op p pid).run s :=
  kernel_run op p _

theorem invocation_safe [Inhabited α] (op : Ops α) (p : Parameters) (pid : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    (invocation op p pid).Safe bounds s ↔ (PublicIO.invocation op p pid).Safe bounds s :=
  kernel_safe op p bounds _

theorem schedule_run [Inhabited α] (op : Ops α) (p : Parameters) (order : List Nat) (s : State α) :
    runSchedule (order.map (invocation op p)) s =
      runSchedule (order.map (PublicIO.invocation op p)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons, runSchedule, invocation_run, ih]

theorem schedule_safe [Inhabited α] (op : Ops α) (p : Parameters) (order : List Nat)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (order.map (invocation op p)) s ↔
      ScheduleSafe bounds (order.map (PublicIO.invocation op p)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons, ScheduleSafe, invocation_safe, invocation_run, ih]




/-- Full-grid equivalence, with both successful runs, both safe traces,
all output cells and both exact exterior frames. -/
theorem equivalent (instructions : NativeInstructions) (tree : ReductionTree)
    (p : Parameters) (n : Nat) (dest : Nat → Nat) (covers : p.groups ≤ p.block)
    (unique : ∀ a b, a<n → b<n →
      KernelGridResult.target p dest a = KernelGridResult.target p dest b → a=b) :
    (PublicIO.sourceIO (KernelModels.nativeOps instructions tree) p n dest).EquivalentOn
      (variant (KernelModels.nativeOps instructions tree) p n dest) (PublicIO.Metadata dest) := by
  have left := PublicIO.implements (KernelModels.nativeOps instructions tree) p n dest
    (fun values pid g c => (KernelModels.nativeCode instructions tree (PublicIO.elements values pid g) c).toInt)
    (fun values pid g => KernelModels.nativeScale instructions tree (PublicIO.elements values pid g))
    (fun values s reads => KernelRaw.native instructions tree p n dest covers unique
      (PublicIO.elements values) s reads)
  intro values domain bounds bound initial binding
  obtain ⟨safe,final,ran,outputs,frame⟩ := left values domain bounds bound initial binding
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_safe _ p (List.range n) bounds initial).mpr safe
  · exact (schedule_run _ p (List.range n) initial).trans ran
  · intro o j active
    rfl

#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.QuantizeKvCopy.Pair
