import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.CheckedNative
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Real

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
open VeriTile.Bench.Repairs.RmsMatmulRbe (realOps realTarget value_real value_reference)
open VeriTile.Bench.Repairs.RmsMatmulRbe.Checked (flat3)
open VeriTile.Bench.Repairs.RmsRbeMatmul.Contract (Inputs InputContract)
noncomputable section
set_option maxHeartbeats 5000000

def mathematicalStorage (decode : Byte → ℝ) (encodeByte : ℝ → Byte) : StorageOps ℝ ℝ where
  toOps := realOps
  hzero := 0
  widen := id
  narrow := id
  e5Half := decode
  e5Single := decode
  e5Encode := encodeByte
/-- Original-input RMS-normalized matmul and optional adjacent-pair rotation.
The original Tensor.to scalar conversion is retained explicitly before E5
payload decoding; it is not replaced by a whole-output accuracy premise. -/
def mathematicalTarget (decode : Byte → ℝ) (encodeByte : ℝ → Byte) (cast : HostCast ℝ ℝ)
    (i : Inputs) (a : Allocation i) (caller : Memory ℝ ℝ) (x : Cell (shape i)) : Stored ℝ ℝ :=
  let ops := mathematicalStorage decode encodeByte
  encode ops (buffers i a) (realTarget (parameters i id)
    (hostRowInput ops cast (sourceDType i) (dense1 i.rms) (buffers i a) caller x.1 x.2.1)
    i.rotary x.2.1 x.2.2)
def mathematicalEntry (decode : Byte → ℝ) (encodeByte : ℝ → Byte)
    (i : Inputs) (a : Allocation i) (returns : Returns i a) (prepared : Memory ℝ ℝ) :=
  (returns.tensor,run (mathematicalStorage decode encodeByte) (parameters i id) (buffers i a)
    selectedPlans selectedReductions (schedule (shape i)).events prepared)

theorem mathematical_prepared (decode : Byte → ℝ) (encodeByte : ℝ → Byte) (cast : HostCast ℝ ℝ)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (caller prepared : Memory ℝ ℝ)
    (prep : HostPreparation i a cast caller prepared) (x : Cell (shape i)) :
    wordTarget (mathematicalStorage decode encodeByte) (parameters i id) (buffers i a)
      selectedPlans selectedReductions prepared x=mathematicalTarget decode encodeByte cast i a caller x := by
  let ops := mathematicalStorage decode encodeByte
  have p := HostPreparation.legacy i h a cast caller prepared prep
  have he : wordTarget ops (parameters i id) (buffers i a) selectedPlans selectedReductions prepared x=
      hostTarget ops cast (sourceDType i) (dense1 i.rms) (parameters i id) (buffers i a) caller x := by
    unfold wordTarget hostTarget
    congr 1
    apply reference_rms_congr
    · funext k
      simp only [rowInput,hostRowInput,read32,p.x]
    · funext n k
      simp only [rowInput,hostRowInput,loadWeight,read8,read16,p.weight]
    · intro k hk
      simp only [rowInput,hostRowInput,rmsFromWord,loadRms,read8,read16,p.rms k hk]
  rw [he]
  unfold hostTarget mathematicalTarget
  rw [←value_reference,show ops.toOps=realOps from rfl,value_real]
  rfl

theorem mathematical_returns (decode : Byte → ℝ) (encodeByte : ℝ → Byte) (cast : HostCast ℝ ℝ)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (returns : Returns i a)
    (caller prepared : Memory ℝ ℝ) (prep : HostPreparation i a cast caller prepared) :
    (∀ x : Cell (shape i),tensorWord (mathematicalEntry decode encodeByte i a returns prepared).2
      (mathematicalEntry decode encodeByte i a returns prepared).1
      (flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2)=mathematicalTarget decode encodeByte cast i a caller x) ∧
    (∀ q,q.1≠a.out.base → (sourceDType i=weightDType (buffers i a) ∨ q.1≠a.rms.base) →
      (mathematicalEntry decode encodeByte i a returns prepared).2 q=caller q) := by
  constructor
  · intro x
    have active : 0<(shape i).B ∧ 0<(shape i).M := ⟨by have := x.1.isLt; omega,by have := x.2.1.isLt; omega⟩
    change run _ _ _ _ _ _ prepared (returns.tensor.base,returns.tensor.offset+_)=_
    rw [return_address returns x]
    rw [(whole_word_and_frame (mathematicalStorage decode encodeByte) (parameters i id) (buffers i a)
      (buffers_separate i h a active) (output_injective i a) selectedPlans selectedReductions (schedule (shape i)) prepared).1 x]
    exact mathematical_prepared decode encodeByte cast i h a caller prepared prep x
  · intro q fresh original
    rw [show (mathematicalEntry decode encodeByte i a returns prepared).2 q=prepared q from
      run_frame _ _ _ _ _ _ _ q (fun x he => fresh (congrArg Prod.fst he))]
    exact prep.frame q fresh original

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
