import bench.tritonbench_g.rope_transform.Proofs.TypedSoftware
namespace VeriTile.Bench.TritonBenchG.RopeTransform.Typed
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessSmall : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem witness_arithmetic :
    FP.mul .ieeeValue witnessOne witnessOne = witnessOne ∧
    FP.mul .ieeeValue witnessSmall witnessOne = witnessSmall ∧
    FP.add .ieeeValue witnessSmall witnessOne = witnessOne ∧
    FP.sub .ieeeValue witnessOne witnessSmall = witnessOne ∧
    witnessSmall.toRat? = some (1 / 33554432) ∧
    (1 - 1 / 33554432 : Rat) ≠ 1 := by decide +kernel

def witnessParameters : Parameters where
  q := "Q"
  k := "K"
  cos := "cos"
  sin := "sin"
  qStride := 4
  kStride := 4
  cosStride := 2
  sinStride := 2
  length := 1
  qHeads := 1
  kHeads := 1
  headDim := 2
  paddedQHeads := 2
  paddedKHeads := 2
  paddedHeadDim := 4
  backward := Bool.false

def witnessState [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun r o => .mk .fp32 (if r = "cos" ∨ r = "sin" ∨ o = 0 then witnessOne else witnessSmall)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem bit_success [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (kernel .fp32 .fp32 witnessParameters) witnessState = some t ∧
      t.mem "Q" 0 = .mk .fp32 witnessOne ∧
      t.mem "Q" 1 = .mk .fp32 witnessOne ∧
      t.mem "K" 0 = .mk .fp32 witnessOne ∧
      t.mem "K" 1 = .mk .fp32 witnessOne ∧
      t.mem "Q" 2 = .mk .fp32 witnessSmall ∧
      (1 - 1 / 33554432 : Rat) ≠ 1 := by
  let t := (execution (FP.Software.scalarFP32 rest .ieeeValue) witnessParameters witnessState).val
  have hm : t.mem = roundedMemory .ieeeValue witnessParameters witnessState :=
    (execution_memory _ _ _).trans (reference_software _ _ _ _)
  refine ⟨t, (execution _ _ _).property, ?_, ?_, ?_, ?_, ?_, witness_arithmetic.2.2.2.2.2⟩
  all_goals rw [hm]
  all_goals simp [roundedMemory, scatter, witnessParameters, witnessState,
    TileShape.allIndices, List.finRange_succ, active, address, roundedLane, readFloat, toFloat, Cell.read,
    witness_arithmetic.1, witness_arithmetic.2.1, witness_arithmetic.2.2.1,
    witness_arithmetic.2.2.2.1]

#axiomsClean witness_arithmetic
#axiomsClean bit_success
end
end VeriTile.Bench.TritonBenchG.RopeTransform.Typed
