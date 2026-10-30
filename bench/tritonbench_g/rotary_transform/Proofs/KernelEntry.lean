import bench.tritonbench_g.rotary_transform.Proofs.KernelEndpoints
import bench.tritonbench_g.rotary_transform.Contract
import bench.tritonbench_g.rotary_transform.Proofs.RealWrapper

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelEntry
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section

def signed (m : Physical.Bytes) (r : RotaryTransform.Region) (wide : Bool) (i : Nat) : Int :=
  if wide then ((Physical.load m r (2*i+1) ++ Physical.load m r (2*i)) : BitVec 64).toInt
  else (Physical.load m r i).toInt

theorem signed_eq (m : Physical.Bytes) (r : RotaryTransform.Region) (wide : Bool) (i : Nat)
    (bound : Physical.integer m r wide i<2^31) :
    signed m r wide i=(Physical.integer m r wide i : Int) := by
  cases wide
  · exact Physical.signed32 _ bound
  · exact Physical.signed64 _ bound

/-- Kernel-entry state: the original cumulative Tensor keeps its signed
storage width; offset/coefficient views undergo the original preparation.
Only the existing nonrotary suffix is copied to the output. Real payloads
are arbitrary; integer metadata is independently decoded from caller bytes. -/
def lift {α : Type} (c : Checked.Call) (metadata : Physical.Bytes)
    (payload : Memory α) (other : State α) : State α :=
  { other with mem := fun r i =>
      if r="cu" then Cell.mk .int (signed metadata (.other 0) c.host.abi.cumulative64 i)
      else if r="offsets" then Cell.mk .int (signed (Physical.stageInputs c.host.shape c.views c.host.abi.offsets64 metadata)
          .offsets c.host.abi.offsets64 i)
      else if r="x" then Cell.mk .real (Host.prepare c.host.shape payload .input i)
      else if r="out" then Cell.mk .real (Host.prepare c.host.shape payload .output i)
      else if r="cos" then Cell.mk .real (payload .cosine i)
      else if r="sin" then Cell.mk .real (payload .sine i)
      else other.mem r i }

theorem cumulative_bound (c : Config) (valid : c.Valid) (abi : IndexABI c)
    (packed : c.shape.varlen=Bool.true) (i : Nat) (inside : i≤c.shape.batches) :
    c.metadata.cumulative i<2^31 := by
  have total : c.shape.total<2^31 := by
    have a : c.shape.total≤c.shape.total*c.shape.heads := Nat.le_mul_of_pos_right _ valid.heads
    have b : c.shape.total*c.shape.heads≤c.shape.total*c.shape.heads*c.shape.width :=
      Nat.le_mul_of_pos_right _ valid.width
    have limit := abi.data
    unfold Host.size at limit
    omega
  have ending := valid.total
  simp only [packed,if_true] at ending
  have monotone := valid.cumulativeMonotone packed i c.shape.batches inside (Nat.le_refl _)
  omega

theorem offset_bound (c : Config) (abi : IndexABI c)
    (tensor : c.shape.offsetTensor=Bool.true) (i : Nat) (inside : i<c.shape.batches) :
    c.metadata.offsets i<2^31 := by
  have bound := abi.offsets i inside
  simp only [Config.shift,tensor,if_true] at bound
  omega

theorem metadata {α : Type} [Inhabited α] (c : Checked.Call) (bytes : Physical.Bytes)
    (payload : Memory α) (other : State α) (accepted : check c bytes=Bool.true) :
    ProgramModel.MetadataBinding (Checked.configuration c bytes) (lift c bytes payload other) := by
  let cfg := Checked.configuration c bytes
  have accepted' := admitted c bytes accepted
  have valid : cfg.Valid := Checked.config_valid c _ accepted'
  have abi : IndexABI cfg := Checked.indices c _ accepted'
  have tensorsAccepted := (Checked.checks c _ accepted').1.1.2
  simp only [Checked.tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at tensorsAccepted
  have stride : c.host.abi.cumulativeStride=1 := by tauto
  constructor
  · intro packed i inside
    have bound := cumulative_bound cfg valid abi packed i inside
    have packed' : c.host.shape.varlen=Bool.true := packed
    have value : cfg.metadata.cumulative i=
        Physical.integer bytes (.other 0) c.host.abi.cumulative64 i := by
      simp [cfg,Checked.configuration,Physical.callerConfiguration,Physical.canonicalMetadata,packed',stride]
    rw [value] at bound
    change (Cell.mk .int (signed bytes (.other 0) c.host.abi.cumulative64 i)).read .int=_
    rw [Cell.read_same,signed_eq _ _ _ _ bound]
    exact congrArg Int.ofNat value.symm
  · intro tensor i inside
    have bound := offset_bound cfg abi tensor i inside
    have tensor' : c.host.shape.offsetTensor=Bool.true := tensor
    have value : cfg.metadata.offsets i=
        Physical.integer (Physical.stageInputs c.host.shape c.views c.host.abi.offsets64 bytes)
          .offsets c.host.abi.offsets64 i := by
      rw [Physical.staged_offset_integer]
      simp [cfg,Checked.configuration,Physical.callerConfiguration,Physical.canonicalMetadata,tensor']
    rw [value] at bound
    change (Cell.mk .int (signed _ .offsets c.host.abi.offsets64 i)).read .int=_
    rw [Cell.read_same,signed_eq _ _ _ _ bound]
    exact congrArg Int.ofNat value.symm

theorem payload_read {α : Type} [Inhabited α] (c : Checked.Call) (bytes : Physical.Bytes)
    (payload : Memory α) (other : State α) (r : RotaryTransform.Region) (i : Nat)
    (role : r=.input ∨ r=.output ∨ r=.cosine ∨ r=.sine) :
    KernelAlgebra.memory (lift c bytes payload other) r i=Host.prepare c.host.shape payload r i := by
  rcases role with rfl|rfl|rfl|rfl <;>
    simp [KernelAlgebra.memory,KernelAlgebra.regionName,lift,Host.prepare,Host.copied]

theorem suffix {α : Type} [Inhabited α] (c : Checked.Call) (bytes : Physical.Bytes)
    (payload : Memory α) (other : State α) :
    KernelEndpoints.TailReady (Checked.configuration c bytes) (lift c bytes payload other) := by
  intro token head col ht hh hd tail
  change KernelAlgebra.memory (lift c bytes payload other) c.host.shape.output
    (c.host.shape.address token head col)=KernelAlgebra.memory (lift c bytes payload other) .input _
  rw [payload_read _ _ _ _ _ _ (by cases h : c.host.shape.inplace <;> simp [Shape.output,h]),
    payload_read _ _ _ _ _ _ (Or.inl rfl),Host.prepare_tail c.host.shape payload token head col ht hh hd tail,
    Host.prepare_readonly _ _ _ _ (by decide)]
  rfl

theorem real_formula (c : Checked.Call) (bytes : Physical.Bytes) (payload : Memory ℝ) (other : State ℝ) :
    realFormula (Checked.configuration c bytes) (KernelAlgebra.memory (lift c bytes payload other))=
      realFormula (Checked.configuration c bytes) payload := by
  funext b r h d
  simp [realFormula,KernelAlgebra.memory,KernelAlgebra.regionName,lift,Host.prepare,Host.copied]

theorem native_formula (c : Checked.Call) (bytes : Physical.Bytes) (payload : Memory Native.Word)
    (other : State Native.Node) (p : Native.Instructions) :
    nativeFormula (Checked.configuration c bytes) p
      (Native.project (KernelAlgebra.memory (lift c bytes (fun r i => Native.atom (payload r i)) other)))=
      nativeFormula (Checked.configuration c bytes) p payload := by
  funext b r h d
  simp [nativeFormula,Native.project,KernelAlgebra.memory,KernelAlgebra.regionName,lift,
    Host.prepare,Host.copied,Native.atom]

#axiomsClean metadata
#axiomsClean payload_read
#axiomsClean suffix
#axiomsClean real_formula
#axiomsClean native_formula
#specNonCircular lift avoiding [TypedGrid.run,ProgramModel.result,source,realFormula,nativeFormula]
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelEntry
