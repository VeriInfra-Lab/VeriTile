import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Native
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.NativeStages
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

/-- Reification of the already selected three scalar gate instructions. This
proof fragment is not a second authoritative Python/kernel implementation.
The stage-local `sigmoid` port denotes the existing `sigmoidNeg` instruction.
Only `accumulator` is written, exactly as in `gateExecution`. -/
def gateCode : List ComputeStmt := [
  .assign .real [16,16] "accumulator" (.fma .fp32 (.consSame .scalarR) .scalarR
    (.compute (.alg .fp32 (.negFloat .real (.ref .real [16,16] "acc1"))))
    (.alg (.ref .real [16] "a_norm"))
    (.compute (.alg .fp32 (.const 0)))),
  .assign .real [16,16] "accumulator" (.compute (.alg .fp32
    (.sigmoid (.ref .real [16,16] "accumulator")))),
  .assign .real [16,16] "accumulator" (.compute (.alg .fp32
    (.mul .real (.consSame (.consSame .nil))
      (.mul .real (.consSame .scalarR) (.ref .real [16,16] "acc2") (.ref .real [16] "a_norm"))
      (.mul .real (.consSame (.consSame .nil))
        (.mul .real (.consSame .scalarR) (.ref .real [16,16] "acc1") (.ref .real [16] "a_norm"))
        (.ref .real [16,16] "accumulator"))))) ]

def gateBackend {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α) : Algebra α :=
  { a.model with
    fma := fun _ => some g.fma
    unary := fun p op => match op with
      | .neg _ => g.neg
      | .sigmoid => g.sigmoidNeg
      | _ => a.unary p op }

theorem gate_exact {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (s : State α) : run (gateBackend a g) gateCode s = gateExecution a g s := by
  cases h1 : s.regs .real [16,16] "acc1" <;>
  cases h3 : s.regs .real [16,16] "acc2" <;>
  cases hr : s.regs .real [16] "a_norm" <;>
    simp [gateCode,gateBackend,gateExecution,gateValue,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,ComputeDType.eraseDType,numeric,Arithmetic.model,h1,h3,hr,ofFloat,toFloat,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,State.setReg]
  funext d sh n
  by_cases hn : n="accumulator"
  · subst n
    by_cases hd : TileDType.real=d
    · subst d
      by_cases hs : [16,16]=sh
      · subst sh; rfl
      · simp [hs]
    · simp [hd]
  · simp [hn]

/-- Exact original three-statement terminal activation site. The selected
lowering consumes these same raw acc1/acc2 and normalization operands; its
FMA/order are the pre-existing compiler boundary, not a numerical rewrite. -/
theorem gate_site (p : Parameters) :
    ((epilogue p).drop 2).take 3 =
      ((epilogue {p with byte := Bool.false}).drop 2).take 3 := by
  rcases p with ⟨M,N,K,b,e⟩
  cases b <;> rfl

theorem gate_safe {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (gateBackend a g) bounds gateCode s := by
  simp [gateCode,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold]
  repeat' first | exact True.intro | constructor | split

#axiomsClean gate_exact
#axiomsClean gate_site
#axiomsClean gate_safe
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.NativeStages
