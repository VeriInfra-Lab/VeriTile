import bench.repairs.tritonbench_g.llama_ff_triton.Kernels
import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Real
import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.CheckedHost
namespace VeriTile.Bench.Repairs.LlamaFF
noncomputable section

/-- Ideal decoded storage erases representation casts. This does not identify
native floating arithmetic, half rounding or nonfinite byte payloads with ℝ. -/
def idealStorage : StorageOps ℝ ℝ where
  toOps := realOps
  hzero := 0
  widen := id
  narrow := id
  e5Half := fun _ => 0
  e5Single := fun _ => 0
def targetAt (p : Parameters ℝ) (b : Buffers) (initial : Memory ℝ ℝ) (x : Cell s) :=
  realTarget p (rowInput idealStorage b initial x.1 x.2.1) x.2.2

theorem callee_correct (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    ff_llama realOps p x plans reduction n=realTarget p x n := value_real p x plans reduction n

/-- Every logical output and full frame for complete current-memory dispatch,
with decoded mathematical inputs and a target independent of the implementation. -/
theorem correct (p : Parameters ℝ) (b : Buffers) (decoded : b.useByte=Bool.false)
    (sep : b.Separate) (injective : Function.Injective (outputAddress b : Cell s → Address))
    (schedule : Schedule s) (initial : Memory ℝ ℝ) :
    (∀ x : Cell s,run idealStorage p b selectedPlans selectedReductions schedule.events initial (outputAddress b x)=
      .inr (.inr (targetAt p b initial x))) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      run idealStorage p b selectedPlans selectedReductions schedule.events initial q=initial q) := by
  have h:=whole_word_and_frame idealStorage p b sep injective selectedPlans selectedReductions schedule initial
  refine ⟨?_,h.2⟩
  intro x
  rw [h.1 x]
  unfold wordTarget
  rw [←value_reference,show idealStorage.toOps=realOps from rfl,value_real]
  rfl

/-- Local mathematical interpretation of a raw E5M2 byte. The chosen decoder
is explicit, including how exceptional byte patterns are represented in ℝ.
No native floating-point-to-Real equality is assumed. -/
def decodedStorage (decode : Byte → ℝ) : StorageOps ℝ ℝ :=
  { idealStorage with e5Half := decode, e5Single := decode }
def decodedTarget (decode : Byte → ℝ) (p : Parameters ℝ) (b : Buffers)
    (initial : Memory ℝ ℝ) (x : Cell s) :=
  realTarget p (rowInput (decodedStorage decode) b initial x.1 x.2.1) x.2.2

theorem checked_real (x w v r : Contract.TensorInfo) (ok : Contract.Admitted x w v r)
    (allocation : CheckedHost.Binding x w v r) (decode : Byte → ℝ)
    (schedule : Schedule (Contract.shape x w)) (initial : Memory ℝ ℝ) :
    let b := CheckedHost.buffers x w v r allocation.output.backing
    let p : Parameters ℝ := ⟨Contract.columns x,(1:ℝ)/1000000⟩
    (∀ cell : Cell (Contract.shape x w),run (decodedStorage decode) p b selectedPlans selectedReductions schedule.events initial
      (outputAddress b cell)=.inr (.inr (decodedTarget decode p b initial cell))) ∧
    (∀ q,(∀ cell : Cell (Contract.shape x w),q≠outputAddress b cell) →
      run (decodedStorage decode) p b selectedPlans selectedReductions schedule.events initial q=initial q) := by
  dsimp only
  let d:=CheckedHost.domain x w v r ok allocation
  have h:=whole_word_and_frame (decodedStorage decode)
    (⟨Contract.columns x,(1:ℝ)/1000000⟩ : Parameters ℝ) _ d.fresh d.nonoverlap
    selectedPlans selectedReductions schedule initial
  refine ⟨?_,h.2⟩
  intro cell
  rw [h.1 cell]
  unfold wordTarget
  rw [←value_reference,show (decodedStorage decode).toOps=realOps from rfl,value_real]
  rfl

#axiomsClean checked_real
#specNonCircular decodedTarget avoiding [value,referenceValue,accumulation,run,Code.native]
#axiomsClean callee_correct
#axiomsClean correct
#specNonCircular realTarget avoiding [value,referenceValue,accumulation,run,Code.native]
end
end VeriTile.Bench.Repairs.LlamaFF
