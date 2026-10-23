import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Arithmetic
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Lowering

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- Three scalar instructions at the compiler's terminal activation site.
`sigmoidNeg z` is the selected scalar `1/(1+exp(z))` instruction sequence.
The negative normalization product is a fused FMA, including signed zeros. -/
structure GateInstructions (α : Type) where
  neg : α → α
  fma : α → α → α → α
  sigmoidNeg : α → α

def gateValue {α : Type} (a : Arithmetic α) (g : GateInstructions α)
    (raw1 raw3 rho : α) : α :=
  let mul := a.binary (some .fp32) .real .mul
  let u := mul raw1 rho
  let v := mul raw3 rho
  mul v (mul u (g.sigmoidNeg (g.fma (g.neg raw1) rho (a.literal (some .fp32) .real 0))))

/-- Local scalar lowering of the two normalization multiplies and terminal
activation. Address generation and the actual typed masked store follow via
the unchanged shared interpreter. No whole result or program is supplied. -/
def gateExecution {α : Type} (a : Arithmetic α) (g : GateInstructions α)
    (current : State α) : Option (State α) := do
  let raw1 ← current.regs .real [16,16] "acc1"
  let raw3 ← current.regs .real [16,16] "acc2"
  let rho ← current.regs .real [16] "a_norm"
  pure (current.setReg "accumulator" .real [16,16]
    (fun i => gateValue a g (raw1 i) (raw3 i) (rho (i.1,PUnit.unit))))

def nativeExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (initial : State α) : Option (State α) := do
  let start ← run a.model (prelude p) initial
  let mid ← step a.model (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)) start
  let norm ← run a.model ((epilogue p).take 2) mid
  let gate ← gateExecution a g norm
  run a.model ((epilogue p).drop 5) gate

theorem epilogue_partition (p : Parameters) :
    epilogue p = (epilogue p).take 2 ++ ((epilogue p).drop 2).take 3 ++ (epilogue p).drop 5 := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  cases fp8 <;> rfl

#axiomsClean epilogue_partition
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
