import bench.tritonbench_g.isfinite_kernel.Proofs.Source
import bench.tritonbench_g.isfinite_kernel.Proofs.BitBackend

/-! Actual header execution from arbitrary states and arbitrary algebras.
Readiness is established by source assignments, not assumed of an initial
state. The unchanged unused cdiv assignment remains part of the execution. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural
noncomputable section

def count (p : Parameters) : Nat := (p.size + p.block - 1) / p.block

def beforeState {α : Type} (p : Parameters) (s : State α) : State α :=
  (s.setReg "pid" .nat [] (fun _ => s.pids 0)).setReg
    "num_tiles0" .nat [] (fun _ => count p)

def headerState {α : Type} (p : Parameters) (s : State α) : State α :=
  (beforeState p s).setReg "num_ctas" .nat [] (fun _ => s.numPids 0)

def OneReady {α : Type} (pid : Nat) (s : State α) : Prop :=
  s.regs .nat [] "pid" = some (fun _ => pid)

def LoopReady {α : Type} (pid numCtas : Nat) (s : State α) : Prop :=
  OneReady pid s ∧ s.regs .nat [] "num_ctas" = some (fun _ => numCtas)

theorem count_eval {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    FP.Structural.evalOp M none (countOp p) s = some (fun _ => count p) := by
  simp only [countOp, FP.Structural.evalOp_unfold, bind, Option.bind, Option.pure_def,
    FP.Structural.numeric]
  apply congrArg some
  funext i
  cases i
  rfl

theorem before_eval {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    FP.Structural.run M (before p) s = some (beforeState p s) := by
  have hpid : FP.Structural.step M
      (.assign .nat [] "pid" (.alg (.programId 0))) s =
      some (s.setReg "pid" .nat [] (fun _ => s.pids 0)) := by
    rw [FP.Structural.step]
    simp only [FP.Structural.evalExpr, FP.Structural.evalOp_unfold,
      bind, Option.bind, Option.pure_def]
  have hcount (t : State α) : FP.Structural.step M
      (.assign .nat [] "num_tiles0" (.alg (countOp p))) t =
      some (t.setReg "num_tiles0" .nat [] (fun _ => count p)) := by
    rw [FP.Structural.step]
    simp only [FP.Structural.evalExpr, count_eval,
      bind, Option.bind, Option.pure_def]
  rw [before_exact]
  simp only [FP.Structural.run, hpid, hcount, bind, Option.bind,
    Option.pure_def, beforeState]

theorem header_eval {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    FP.Structural.run M (header p) s = some (headerState p s) := by
  rw [header, FP.Structural.run_append, before_eval]
  simp [numCtas, headerState, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp_unfold, Option.bind] <;> rfl

theorem before_ready {α : Type} (p : Parameters) (s : State α) :
    OneReady (s.pids 0) (beforeState p s) := by
  simp [OneReady, beforeState]

theorem header_ready {α : Type} (p : Parameters) (s : State α) :
    LoopReady (s.pids 0) (s.numPids 0) (headerState p s) := by
  simp [LoopReady, OneReady, headerState, beforeState]

theorem before_memory {α : Type} (p : Parameters) (s : State α) :
    (beforeState p s).mem = s.mem := rfl

theorem header_memory {α : Type} (p : Parameters) (s : State α) :
    (headerState p s).mem = s.mem := rfl

#axiomsClean count_eval
#axiomsClean before_eval
#axiomsClean header_eval
#axiomsClean before_ready
#axiomsClean header_ready
#axiomsClean before_memory
#axiomsClean header_memory
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
