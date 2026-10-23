import bench.repairs.tritonbench_g.log_softmax.Proofs.Address
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.Structural

/-! The kernel receives the original wrapper's contiguous() result. These
bindings use shared Tensor metadata/storage laws; they add no guard requiring
the caller's original input or upstream gradient to be contiguous. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.Binding
open VeriTile.Triton VeriTile.Torch.FP32
noncomputable section
local instance : Inhabited Word := ⟨⟨0⟩⟩

def dense (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true)
    (aligned : 0<t.metadata.numel → t.dataPtr.toNat%4=0) : Address.Dense t.metadata.numel where
  base := (dataView t laws contiguous).base
  finite := by simpa only [dataView,Nat.mul_comm] using (dataView t laws contiguous).spanFits
  aligned := dataView_aligned t laws contiguous aligned

theorem actual_pointer (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true)
    (aligned : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (nonempty : 0<t.metadata.numel) :
    BitVec.ofNat 64 ((dense t laws contiguous aligned).address 0)=t.dataPtr :=
  VeriTile.Torch.FP32.actual_pointer t laws contiguous nonempty

theorem direct_read (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true)
    (resolved : t.metadata.isNeg=Bool.false)
    (aligned : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (read : Nat → Word) (i : Nat) (hi : i<t.metadata.numel) :
    read ((dense t laws contiguous aligned).address i)=logicalValue read t laws i := by
  simpa only [dense,Address.Dense.address,DataView.address,Nat.mul_comm] using
    direct_read_eq_logical read t laws contiguous resolved i hi

/-- Local library result of the existing contiguous() call. Only observed
metadata and element preservation occur here, never a kernel or reduction
result. Tensor bounds and pointer meaning remain the shared core laws. -/
structure ContiguousResult (input : Tensor) (inputLaws : TensorLaws input)
    (before after : Nat → Word) where
  tensor : Tensor
  laws : TensorLaws tensor
  shape : tensor.metadata.shape=input.metadata.shape
  device : tensor.metadata.device=input.metadata.device
  contiguous : tensor.metadata.isContiguous=Bool.true
  resolved : tensor.metadata.isNeg=Bool.false
  aligned : 0<tensor.metadata.numel → tensor.dataPtr.toNat%4=0
  values : ∀ i,i < input.metadata.numel →
    logicalValue after tensor laws i=logicalValue before input inputLaws i

/-- The actual pointer passed to the kernel reads the original caller's logical
element, including an originally noncontiguous input or gradient. -/
theorem prepared_read (input : Tensor) (inputLaws : TensorLaws input)
    (before after : Nat → Word) (result : ContiguousResult input inputLaws before after)
    (i : Nat) (hi : i < input.metadata.numel) :
    after ((dense result.tensor result.laws result.contiguous result.aligned).address i)=
      logicalValue before input inputLaws i := by
  rw [direct_read result.tensor result.laws result.contiguous result.resolved result.aligned after i]
  · exact result.values i hi
  · simpa only [VeriTile.Torch.TensorMetadata.numel,result.shape] using hi

/-- A local embedding of the actual FP32 pointer memory into a source input
region. This is the ordinary memory binding, not a kernel-result premise. -/
def SourceInput (s : FP.Structural.State Word) (region : RegionName)
    (read : Nat → Word) (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true)
    (ha : 0<t.metadata.numel → t.dataPtr.toNat%4=0) : Prop :=
  ∀ i,i<t.metadata.numel → (s.mem region i).read .real=read ((dense t laws hc ha).address i)

/-- The initial cell read by the canonical kernel is the original caller's
logical Tensor value, after its already-existing contiguous() call. -/
theorem prepared_source_read (s : FP.Structural.State Word) (region : RegionName)
    (input : Tensor) (inputLaws : TensorLaws input) (before after : Nat → Word)
    (result : ContiguousResult input inputLaws before after)
    (binding : SourceInput s region after result.tensor result.laws result.contiguous result.aligned)
    (i : Nat) (hi : i < input.metadata.numel) :
    (s.mem region i).read .real=logicalValue before input inputLaws i := by
  rw [binding i (by simpa only [VeriTile.Torch.TensorMetadata.numel,result.shape] using hi)]
  exact prepared_read input inputLaws before after result i hi

/-- Ordinary source-input embedding at the pointer actually passed to Triton.
It constrains input reads only and assumes no output or execution result. -/
def PointerInput (s : FP.Structural.State Word) (region : RegionName)
    (read : Nat → Word) (t : Tensor) : Prop :=
  ∀ i,i<t.metadata.numel → (s.mem region i).read .real=read (t.dataPtr.toNat+4*i)

theorem dense_address (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true)
    (ha : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (i : Nat) (hi : i<t.metadata.numel) :
    (dense t laws hc ha).address i=t.dataPtr.toNat+4*i := by
  have pointer := congrArg BitVec.toNat (actual_pointer t laws hc ha (by omega))
  have capacity := (dense t laws hc ha).finite
  have small : (dense t laws hc ha).base<2^64 := by omega
  simp only [Address.Dense.address,Nat.zero_mul,Nat.add_zero,BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt small] at pointer
  simp only [Address.Dense.address,pointer,Nat.mul_comm]

theorem source_of_pointer (s : FP.Structural.State Word) (region : RegionName)
    (read : Nat → Word) (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true)
    (ha : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (binding : PointerInput s region read t) : SourceInput s region read t laws hc ha := by
  intro i hi
  rw [dense_address t laws hc ha i hi]
  exact binding i hi

/-- The mathematical model explicitly interprets source FP32 words as Real
inputs. This is neither an IEEE decoder claim nor a new finite-value guard. -/
def realState (interpret : Word → ℝ) (s : FP.Structural.State Word) :
    FP.Structural.State (WithBot ℝ) where
  mem := fun r i => .mk .real (some (interpret ((s.mem r i).read .real)))
  regs := fun _ _ _ => none
  pids := s.pids
  numPids := s.numPids
  undef := fun d _ _ => FP.Structural.defaultValue d

def logicalReal (interpret : Word → ℝ) (read : Nat → Word) (t : Tensor)
    (laws : TensorLaws t) (i : Nat) : ℝ := interpret (logicalValue read t laws i)

theorem prepared_real_read (interpret : Word → ℝ) (s : FP.Structural.State Word)
    (region : RegionName) (input : Tensor) (inputLaws : TensorLaws input)
    (before after : Nat → Word) (result : ContiguousResult input inputLaws before after)
    (binding : PointerInput s region after result.tensor) (i : Nat) (hi : i < input.metadata.numel) :
    ((realState interpret s).mem region i).read .real=
      some (logicalReal interpret before input inputLaws i) := by
  change some (interpret ((s.mem region i).read .real))=_
  rw [prepared_source_read s region input inputLaws before after result
    (source_of_pointer s region after result.tensor result.laws result.contiguous result.aligned binding) i hi]
  rfl

theorem direct_real_read (interpret : Word → ℝ) (s : FP.Structural.State Word)
    (region : RegionName) (read : Nat → Word) (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true) (hn : t.metadata.isNeg=Bool.false)
    (ha : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (binding : PointerInput s region read t) (i : Nat) (hi : i<t.metadata.numel) :
    ((realState interpret s).mem region i).read .real=some (logicalReal interpret read t laws i) := by
  change some (interpret ((s.mem region i).read .real))=_
  rw [source_of_pointer s region read t laws hc ha binding i hi,direct_read t laws hc hn ha read i hi]
  rfl

#axiomsClean actual_pointer
#axiomsClean direct_read
#axiomsClean prepared_read
#axiomsClean prepared_source_read
#axiomsClean dense_address
#axiomsClean prepared_real_read
#axiomsClean direct_real_read
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogSoftmax.Binding
