import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedNative
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Real

namespace VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
open Contract
noncomputable section
set_option maxHeartbeats 5000000

/-- Ideal arithmetic is independent of native bit arithmetic. Half values
are decoded Reals; E5M2 decode/encode are explicit scalar representation
ports. No native floating operation is identified with a Real operation. -/
def mathematicalStorage (decode : Byte → ℝ) (encodeByte : ℝ → Byte) : StorageOps ℝ ℝ where
  toOps := realOps
  hzero := 0
  widen := id
  narrow := id
  e5Half := decode
  e5Single := decode
  e5Encode := encodeByte
def mathematicalTarget (decode : Byte → ℝ) (encodeByte : ℝ → Byte)
    (i : Inputs) (a : Allocation i) (caller : Memory ℝ ℝ) (x : Cell (shape i)) : Stored ℝ ℝ :=
  let ops := mathematicalStorage decode encodeByte
  encode ops (buffers i a.q) x.1 (realTarget (parameters i id)
    (rowInput ops (buffers i a.q) caller x.1 x.2.1 x.2.2.1) (rotary x.1) x.2.2.1 x.2.2.2)
def mathematicalEntry (decode : Byte → ℝ) (encodeByte : ℝ → Byte)
    (i : Inputs) (a : Allocation i) (returns : Returns i a) (caller : Memory ℝ ℝ) :=
  (returns.tensor,run (mathematicalStorage decode encodeByte) (parameters i id) (buffers i a.q)
    selectedPlans selectedReductions (schedule (shape i)).events caller)

theorem mathematical_returns (decode : Byte → ℝ) (encodeByte : ℝ → Byte)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (returns : Returns i a) (caller : Memory ℝ ℝ) :
    (∀ x : Cell (shape i),tensorWord (mathematicalEntry decode encodeByte i a returns caller).2
      ((mathematicalEntry decode encodeByte i a returns caller).1 x.1)
      (RmsMatmulRbe.Checked.flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2)=
      mathematicalTarget decode encodeByte i a caller x) ∧
    (∀ q,(∀ x : Cell (shape i),q≠outputAddress (buffers i a.q) x) →
      (mathematicalEntry decode encodeByte i a returns caller).2 q=caller q) := by
  constructor
  · intro x
    have active : 0<(shape i).B ∧ 0<(shape i).M := ⟨by have := x.2.1.isLt; omega,by have := x.2.2.1.isLt; omega⟩
    change run _ _ _ _ _ _ caller ((returns.tensor x.1).base,(returns.tensor x.1).offset+_)=_
    rw [return_address returns x]
    rw [(whole_word_and_frame (mathematicalStorage decode encodeByte) (parameters i id) (buffers i a.q)
      (buffers_separate i h a active) (output_injective i h a) selectedPlans selectedReductions (schedule (shape i)) caller).1 x]
    unfold wordTarget mathematicalTarget
    rw [←value_reference,show (mathematicalStorage decode encodeByte).toOps=realOps from rfl,value_real]
  · intro q outside
    exact run_frame _ _ _ _ _ _ _ q outside

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
