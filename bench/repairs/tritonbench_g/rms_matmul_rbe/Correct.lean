import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Program
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Real
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Mathematical
namespace VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section
set_option maxHeartbeats 5000000

/-- Mathematical storage contains decoded Real input values; all representation
casts are erased. It is not an assertion that native floating arithmetic is exact. -/
def idealStorage : StorageOps ℝ ℝ where
  toOps := realOps
  hzero := 0
  widen := id
  narrow := id
  e5Half := fun _ => 0
  e5Single := fun _ => 0
  e5Encode := fun _ => 0

def targetAt (p : Parameters ℝ) (b : Buffers) (initial : Memory ℝ ℝ) (x : Cell s) :=
  realTarget p (rowInput idealStorage b initial x.1 x.2.1 x.2.2.1) (rotary x.1) x.2.2.1 x.2.2.2

theorem callee_correct (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (rotary : Bool) (row n : Nat) :
    rms_matmul_rbe realOps p x plans reduction rotary row n=realTarget p x rotary row n :=
  value_real p x plans reduction rotary row n

theorem correct (p : Parameters ℝ) (b : Buffers) (decoded : b.useByte=Bool.false)
    (sep : b.Separate) (injective : Function.Injective (outputAddress b : Cell s → Address))
    (schedule : Schedule s) (initial : Memory ℝ ℝ) (x : Cell s) :
    run idealStorage p b selectedPlans selectedReductions schedule.events initial (outputAddress b x)=
      .inr (.inr (targetAt p b initial x)) := by
  rw [(whole_word_and_frame idealStorage p b sep injective selectedPlans selectedReductions schedule initial).1 x]
  unfold wordTarget
  rw [←value_reference,show idealStorage.toOps=realOps from rfl,value_real]
  simp [encode,decoded,idealStorage,targetAt]

theorem correct_frame (p : Parameters ℝ) (b : Buffers) (schedule : Schedule s)
    (initial : Memory ℝ ℝ) (q : Address) (outside : ∀ x : Cell s,q≠outputAddress b x) :
    run idealStorage p b selectedPlans selectedReductions schedule.events initial q=initial q :=
  run_frame idealStorage p b selectedPlans selectedReductions schedule.events initial q outside

#axiomsClean callee_correct
#axiomsClean correct
#axiomsClean correct_frame

theorem python_checks_iff_input_contract (i : Contract.Inputs) :
    Contract.checkInputs i=.ok () ↔ Contract.InputContract i := Contract.checkInputs_iff i

/-- Whole checked QKV API in the independent Real arithmetic interpretation.
The target is RMS-normalized GEMM followed by head-local adjacent-pair rotary
for Q/K, with the original input storage and explicit output representation. -/
specification rms_matmul_rbe_repaired_real_returns
    (decode : Byte → ℝ) (encodeByte : ℝ → Byte)
    (i : Contract.Inputs) (h : Contract.InputContract i) (a : Checked.Allocation i)
    (returns : Checked.Returns i a) (caller : Memory ℝ ℝ) :
    (∀ phase,((Checked.mathematicalEntry decode encodeByte i a returns caller).1 phase).metadata.shape=
      [(Checked.shape i).B,(Checked.shape i).M,(Checked.shape i).heads,(Checked.shape i).headDim]) ∧
    (∀ x : Cell (Checked.shape i),Checked.tensorWord (Checked.mathematicalEntry decode encodeByte i a returns caller).2
      ((Checked.mathematicalEntry decode encodeByte i a returns caller).1 x.1)
      (Checked.flat3 (Checked.shape i).M (Checked.shape i).N x.2.1 x.2.2.1 x.2.2.2)=
      Checked.mathematicalTarget decode encodeByte i a caller x) ∧
    (∀ q,(∀ x : Cell (Checked.shape i),q≠outputAddress (Checked.buffers i a.q) x) →
      (Checked.mathematicalEntry decode encodeByte i a returns caller).2 q=caller q) := by
  exact ⟨fun phase => (returns.views phase).shape,Checked.mathematical_returns decode encodeByte i h a returns caller⟩

#axiomsClean rms_matmul_rbe_repaired_real_returns
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbe
