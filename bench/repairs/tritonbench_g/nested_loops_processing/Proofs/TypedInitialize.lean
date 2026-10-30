import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedStep
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural Kernels.LegacyPastedSurface
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def prefixBody (count cols : Nat) : List ComputeStmt :=
  (source count cols).surfaceBody.take 7

def initialA (cols : Nat) : OffsetTile := fun i ↦ i.1.val * cols + i.2.1.val

def initialC (cols : Nat) : OffsetTile := fun i ↦ cols * i.1.val + i.2.1.val

def unique (cols : Nat) : Unique :=
  fun i ↦ decide (i.1.val = 0) || decide (1 < cols) || decide (i.2.1.val = 1)

def prefixState (s : State α) (cols : Nat) : State α :=
  let s := s.setReg "offs_am" .nat [2] (fun i ↦ i.1.val)
  let s := s.setReg "offs_an" .nat [2] (fun i ↦ i.1.val)
  let s := s.setReg "a_offsets" .nat [2,2] (initialA cols)
  let s := s.setReg "offs_cm" .nat [2] (fun i ↦ i.1.val)
  let s := s.setReg "offs_cn" .nat [2] (fun i ↦ i.1.val)
  let s := s.setReg "c_offsets" .nat [2,2] (initialC cols)
  s.setReg "unique" .bool [2,2] (unique cols)

theorem prefix_step (M : Algebra α) (count cols : Nat) (s : State α) :
    run M (prefixBody count cols) s = some (prefixState s cols) := by
  simp [ComputeDType.eraseDType, Option.bind, prefixBody, source, kernel,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, numericEq, numericLt, natLt, natLe, prefixState, initialA, initialC, unique]
  unfold FP.Structural.bop
  simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex,
    prefixState, initialA, initialC, unique]
  congr 1
  funext i
  rcases i with ⟨r,c,rest⟩
  fin_cases r <;> fin_cases c <;> rfl

theorem source_decomposition (count cols : Nat) :
    (source count cols).surfaceBody = prefixBody count cols ++
      [.forRange "i" 0 2 1 (outer count cols)] := by rfl

end VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
