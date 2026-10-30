import bench.tritonbench_g.decay_cumsum.Proofs.IOResults
import bench.tritonbench_g.decay_cumsum.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.DecayCumsum.IOCorrect
open VeriTile.Triton FP.Structural Geometry DecayIO
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

theorem real (rest : Algebra ℝ) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : ℝ) :
    (io (KernelReal.algebra rest) kind h p prog scale).Implements (realTarget kind h p scale) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial _
    exact schedule_safe _ kind h p hp prog scale vals bounds bounded initial
  · intro vals initial inputs
    obtain ⟨final,ran,values,frame⟩ := source_run (KernelReal.algebra rest) kind h p hp prog scale vals initial inputs
    refine ⟨final,ran,?_,frame⟩
    intro o j _
    cases kind <;> fin_cases o
    · change mapCell id (final.mem "go" (FPPrefix.off h p prog j.1.val j.2.1))=
        Cell.mk .real (KernelReal.evaluate .forward p.span scale
          (fun r n => logicalRead .forward h p vals r n j.2.1) "go" j.1.val)
      rw [values "go" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),KernelReal.evaluate_value]
      rfl
    · change mapCell id (final.mem "qg" (FPPrefix.off h p prog j.1.val j.2.1))=
        Cell.mk .real (KernelReal.evaluate .prepare p.span scale
          (fun r n => logicalRead .prepare h p vals r n j.2.1) "qg" j.1.val)
      rw [values "qg" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),KernelReal.evaluate_value]
      rfl
    · change mapCell id (final.mem "kg" (FPPrefix.off h p prog j.1.val j.2.1))=
        Cell.mk .real (KernelReal.evaluate .prepare p.span scale
          (fun r n => logicalRead .prepare h p vals r n j.2.1) "kg" j.1.val)
      rw [values "kg" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),KernelReal.evaluate_value]
      rfl
    · change mapCell id (final.mem "dqo" (FPPrefix.off h p prog j.1.val j.2.1))=
        Cell.mk .real (KernelReal.evaluate .backward p.span scale
          (fun r n => logicalRead .backward h p vals r n j.2.1) "dqo" j.1.val)
      rw [values "dqo" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),KernelReal.evaluate_value]
      rfl
    · change mapCell id (final.mem "dko" (FPPrefix.off h p prog j.1.val j.2.1))=
        Cell.mk .real (KernelReal.evaluate .backward p.span scale
          (fun r n => logicalRead .backward h p vals r n j.2.1) "dko" j.1.val)
      rw [values "dko" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),KernelReal.evaluate_value]
      rfl
    · change mapCell id (final.mem "dg" (FPPrefix.off h p prog j.1.val j.2.1))=
        Cell.mk .real (KernelReal.evaluate .backward p.span scale
          (fun r n => logicalRead .backward h p vals r n j.2.1) "dg" j.1.val)
      rw [values "dg" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),KernelReal.evaluate_value]
      rfl

theorem native (instructions : Native.Instructions) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : Native.Word) :
    (io (Native.algebra instructions Native.selected kind scale) kind h p prog Native.scaleSlot).Observes
      (nativeObservation kind h p) (nativeTarget instructions kind h p scale) := by
  intro vals _ bounds bounded initial inputs
  refine ⟨schedule_safe _ kind h p hp prog Native.scaleSlot vals bounds bounded initial,?_⟩
  obtain ⟨final,ran,values,frame⟩ := source_run
    (Native.algebra instructions Native.selected kind scale) kind h p hp prog Native.scaleSlot vals initial inputs
  refine ⟨final,ran,?_,frame⟩
  intro o j _
  cases kind <;> fin_cases o
  · change mapCell (fun node : Native.Node => node.value) (mapCell id
        (final.mem "go" (FPPrefix.off h p prog j.1.val j.2.1)))=
      Cell.mk .real (Native.evaluateWords instructions .forward p.span scale
        (fun r n => (logicalRead .forward h p vals r n j.2.1).value) "go" j.1.val)
    rw [values "go" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),Native.evaluate_lowering]
    rfl
  · change mapCell (fun node : Native.Node => node.value) (mapCell id
        (final.mem "qg" (FPPrefix.off h p prog j.1.val j.2.1)))=
      Cell.mk .real (Native.evaluateWords instructions .prepare p.span scale
        (fun r n => (logicalRead .prepare h p vals r n j.2.1).value) "qg" j.1.val)
    rw [values "qg" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),Native.evaluate_lowering]
    rfl
  · change mapCell (fun node : Native.Node => node.value) (mapCell id
        (final.mem "kg" (FPPrefix.off h p prog j.1.val j.2.1)))=
      Cell.mk .real (Native.evaluateWords instructions .prepare p.span scale
        (fun r n => (logicalRead .prepare h p vals r n j.2.1).value) "kg" j.1.val)
    rw [values "kg" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),Native.evaluate_lowering]
    rfl
  · change mapCell (fun node : Native.Node => node.value) (mapCell id
        (final.mem "dqo" (FPPrefix.off h p prog j.1.val j.2.1)))=
      Cell.mk .real (Native.evaluateWords instructions .backward p.span scale
        (fun r n => (logicalRead .backward h p vals r n j.2.1).value) "dqo" j.1.val)
    rw [values "dqo" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),Native.evaluate_lowering]
    rfl
  · change mapCell (fun node : Native.Node => node.value) (mapCell id
        (final.mem "dko" (FPPrefix.off h p prog j.1.val j.2.1)))=
      Cell.mk .real (Native.evaluateWords instructions .backward p.span scale
        (fun r n => (logicalRead .backward h p vals r n j.2.1).value) "dko" j.1.val)
    rw [values "dko" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),Native.evaluate_lowering]
    rfl
  · change mapCell (fun node : Native.Node => node.value) (mapCell id
        (final.mem "dg" (FPPrefix.off h p prog j.1.val j.2.1)))=
      Cell.mk .real (Native.evaluateWords instructions .backward p.span scale
        (fun r n => (logicalRead .backward h p vals r n j.2.1).value) "dg" j.1.val)
    rw [values "dg" (by simp [FPTile.outputs,FPBackwardBase.outputRegions]),Native.evaluate_lowering]
    rfl

/-- The selected object/ABI specialization is the same source IO and carries
its original runtime binary64-to-FP32 scalar conversion without specialization. -/
theorem compiled (platform : KernelCompiler.Platform) (kind : Kind) (h : Call) (p : Phase)
    (binding : KernelCompiler.Binding platform kind h p) (host : UInt64)
    (hp : p∈phases h) (prog : Program h p) :
    (io (KernelCompiler.algebra platform binding host) kind h p prog Native.scaleSlot).Observes
      (nativeObservation kind h p)
      (nativeTarget (platform.instructions binding.object.primitives) kind h p
        (platform.scale binding.object host)) := by
  have graphs := funext binding.graphs
  have same : KernelCompiler.algebra platform binding host=
      Native.algebra (platform.instructions binding.object.primitives) Native.selected kind
        (platform.scale binding.object host) := by
    simp only [KernelCompiler.algebra,graphs]
  rw [same]
  exact native _ kind h p hp prog _

#axiomsClean real
#axiomsClean native
#axiomsClean compiled
end
end VeriTile.Bench.Repairs.DecayCumsum.IOCorrect
