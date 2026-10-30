import bench.tritonbench_g.int4_matmul.Proofs.Kernel.SourceGraph
import bench.tritonbench_g.int4_matmul.Proofs.Kernel.Complete

namespace VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- A compositional equality rule, not an executor field in the IO. -/
theorem run_congr (p : Ops α) (rest : Algebra α) (code : List ComputeStmt)
    (steps : ∀ st ∈ code, ∀ s : FP.Structural.State (Node α),
      KernelExecution.step (ops p) (base p rest) st s = FP.Structural.step (engine p rest) st s)
    (s : FP.Structural.State (Node α)) :
    KernelExecution.run (ops p) (base p rest) code s = FP.Structural.run (engine p rest) code s := by
  induction code generalizing s with
  | nil => rw [KernelExecution.run, FP.Structural.run]
  | cons st tail ih =>
    rw [KernelExecution.run, FP.Structural.run, steps st (by simp)]
    cases h : FP.Structural.step (engine p rest) st s with
    | none => rfl
    | some next => exact ih (fun st hs => steps st (by simp [hs])) next

/-- Exact transport of the actual source body, including all masked operands
and its one selected K8 accumulation node. -/
theorem body_equal (p : Ops α) (rest : Algebra α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) :
    KernelExecution.run (ops p) (base p rest) (KernelExecution.body cfg) s =
      FP.Structural.run (engine p rest) (KernelExecution.body cfg) s := by
  apply run_congr
  intro st member state
  simp [KernelExecution.body,KernelSource.kernel,ComputeKernel.surfaceBody] at member
  repeat' first
    | (simpa [KernelExecution.step] using (accumulating_step p rest _ _ state).symm)
    | (simp only [KernelExecution.step,KernelExecution.localStep,engine]; done)
    | rcases member with rfl | member

theorem prelude_equal (p : Ops α) (rest : Algebra α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) :
    KernelExecution.run (ops p) (base p rest) (KernelExecution.prelude cfg) s =
      FP.Structural.run (engine p rest) (KernelExecution.prelude cfg) s := by
  simp [KernelExecution.prelude,KernelSource.kernel,ComputeKernel.surfaceBody]
  simp only [KernelExecution.run,KernelExecution.step,KernelExecution.localStep,
    FP.Structural.run,FP.Structural.step,engine]
  rfl

theorem epilogue_equal (p : Ops α) (rest : Algebra α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) :
    KernelExecution.run (ops p) (base p rest) (KernelExecution.epilogue cfg) s =
      FP.Structural.run (engine p rest) (KernelExecution.epilogue cfg) s := by
  simp [KernelExecution.epilogue,KernelSource.kernel,ComputeKernel.surfaceBody]
  simp only [KernelExecution.run,KernelExecution.step,KernelExecution.localStep,
    FP.Structural.run,FP.Structural.step,engine]
  rfl

theorem range_equal (p : Ops α) (rest : Algebra α) (cfg : KernelSource.Parameters)
    (name : RegName) (start stop stride : Nat) (s : FP.Structural.State (Node α)) :
    KernelExecution.range (ops p) (base p rest) name start stop stride (KernelExecution.body cfg) s =
      FP.Structural.range (engine p rest) name start stop stride (KernelExecution.body cfg) s := by
  rw [KernelExecution.range,FP.Structural.range]
  by_cases zero : stride = 0
  · simp only [zero,ite_true]
  · simp only [zero,ite_false]
    by_cases active : start < stop
    · simp only [active,ite_true,body_equal]
      cases h : FP.Structural.run (engine p rest) (KernelExecution.body cfg)
          (s.setReg name .nat [] (fun _ => start)) with
      | none => rfl
      | some next => exact range_equal p rest cfg name (start+stride) stop stride next
    · simp only [active,ite_false]
termination_by stop-start
decreasing_by omega

/-- Complete original kernel, with its source loop and both store/atomic
branches, executes through the standard Structural interpreter unchanged. -/
theorem source_equal (p : Ops α) (rest : Algebra α) (cfg : KernelSource.Parameters)
    (s : FP.Structural.State (Node α)) :
    KernelExecution.exec (ops p) (base p rest) cfg s =
      FP.Structural.exec (engine p rest) (KernelSource.kernel cfg) s := by
  change KernelExecution.run (ops p) (base p rest) (KernelSource.kernel cfg).surfaceBody s =
    FP.Structural.run (engine p rest) (KernelSource.kernel cfg).surfaceBody s
  rw [KernelExecution.source_decomposition]
  simp only [KernelExecution.run_append,FP.Structural.run_append,
    prelude_equal,epilogue_equal,KernelExecution.run,KernelExecution.step,
    FP.Structural.run,FP.Structural.step,range_equal]
  rfl

#axiomsClean body_equal
#axiomsClean prelude_equal
#axiomsClean epilogue_equal
#axiomsClean range_equal
#axiomsClean source_equal
end
end VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
