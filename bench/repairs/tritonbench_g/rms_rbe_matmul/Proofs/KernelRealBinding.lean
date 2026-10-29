import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelRealBinding
open VeriTile.Triton FP.Structural
open KernelBinding (Bytes Cast Preparation parameters fp8 readWord signedByte inputTerm)
open Contract (Inputs InputContract)
open KernelNative (Word Term)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Mathematical interpretation of a physical input word only. Runtime
expressions and native outputs are never decoded by this function. -/
def inputValue (decode : Word → ℝ) : Term → ℝ
  | .leaf r _ word => if r="" then 0 else decode word
  | _ => 0
theorem inputValue_default (decode : Word → ℝ) : inputValue decode (default : Term)=0 := rfl
def values (decode32 decode16 : Word → ℝ) (v : KernelGraph.Input Term) : KernelGraph.Input ℝ where
  x := fun b m k => inputValue decode32 (v.x b m k)
  weightHalf := fun n k => inputValue decode16 (v.weightHalf n k)
  rmsHalf := fun k => inputValue decode16 (v.rmsHalf k)
  weightByte := v.weightByte
  rmsByte := v.rmsByte
def originalInputs (decode32 decode16 : Word → ℝ) (i : Inputs) (a : Checked.Allocation i)
    (cast : Cast) (before : Bytes) : KernelGraph.Input ℝ :=
  values decode32 decode16 (KernelBinding.originalInputs i a cast before)

def state (decode32 decode16 : Word → ℝ) (i : Inputs) (a : Checked.Allocation i)
    (heap : Bytes) (scratch : State ℝ) : State ℝ :=
  {scratch with mem:=fun r off =>
    if r="X" then .mk .real (inputValue decode32 (inputTerm "X" i.x off (readWord heap i.x off)))
    else if r="W" then
      (if fp8 i then Cell.mk TileDType.int (signedByte (readWord heap i.weight off))
       else Cell.mk TileDType.fp16 (inputValue decode16 (inputTerm "W" i.weight off (readWord heap i.weight off))))
    else if r="RMS" then
      (if fp8 i then Cell.mk TileDType.int (signedByte (readWord heap a.rms off))
       else Cell.mk TileDType.fp16 (inputValue decode16 (inputTerm "RMS" a.rms off (readWord heap a.rms off))))
    else scratch.mem r off}

theorem state_inputs (d32 d16 : Word → ℝ) (i : Inputs) (a : Checked.Allocation i)
    (heap : Bytes) (scratch : State ℝ) :
    KernelSource.inputs (parameters i) (state d32 d16 i a heap scratch)=
      values d32 d16 (KernelBinding.preparedInputs i a heap) := by
  unfold KernelSource.inputs values KernelBinding.preparedInputs KernelBinding.inputWords
  congr 1
  all_goals funext
  all_goals cases hf : fp8 i <;> simp only [state,hf,Bool.false_eq_true,if_true,if_false]
  all_goals simp [Cell.read,defaultValue,inputValue_default,inputTerm]
  all_goals rfl

theorem state_bytes (d32 d16 : Word → ℝ) (i : Inputs) (a : Checked.Allocation i)
    (heap : Bytes) (scratch : State ℝ) : KernelSource.ByteMemory (state d32 d16 i a heap scratch) := by
  intro r hr off
  rcases hr with rfl|rfl <;> cases hf : fp8 i <;>
    simp only [state,hf,Bool.false_eq_true,if_true,if_false]
  all_goals simp [Cell.read,defaultValue,KernelBinding.signedByte_valid]
  all_goals decide

theorem values_agreement (d32 d16 : Word → ℝ)
    (h : KernelGraph.Agreement p batch v w) :
    KernelGraph.Agreement p batch (values d32 d16 v) (values d32 d16 w) := by
  constructor
  · intro m k hm hk; exact congrArg (inputValue d32) (h.x m k hm hk)
  · intro k hk; exact congrArg (inputValue d16) (h.rmsHalf k hk)
  · exact h.rmsByte
  · intro n k hn hk; exact congrArg (inputValue d16) (h.weightHalf n k hn hk)
  · exact h.weightByte

#axiomsClean state_inputs
#axiomsClean state_bytes
#axiomsClean values_agreement
#specNonCircular originalInputs avoiding [FP.Structural.exec,KernelGraph.output,KernelSource.scatter]
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelRealBinding
