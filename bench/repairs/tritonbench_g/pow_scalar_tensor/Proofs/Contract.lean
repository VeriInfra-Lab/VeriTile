import bench.repairs.tritonbench_g.pow_scalar_tensor.Contract
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeAccess

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole (NativeBuffers)
open HostPlan SignedView SignedGrid
noncomputable section
set_option maxHeartbeats 4000000

theorem view_iff (v : ViewMetadata) : ViewContract v ↔
    v.recognized=Bool.true ∧ v.strided=Bool.true ∧ v.fp32=Bool.true ∧ v.resolved=Bool.true ∧
    v.cuda=Bool.true ∧ v.integerMetadata=Bool.true ∧ (v.shape.length=1 ∧ v.strides.length=1) ∧
    (-(2^63:Int)≤stride v ∧ stride v<2^63) ∧ (v.storageBase%4=0 ∧ v.delta%4=0) ∧
    v.storageBase+v.storageBytes≤2^64 ∧ (0<count v → 0≤lower v ∧ upper v+4≤v.storageBytes) := by
  constructor
  · intro h;exact ⟨h.recognized,h.strided,h.fp32,h.resolved,h.cuda,h.integerMetadata,h.rank,h.strideWord,h.aligned,h.allocation,h.bounds⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k⟩;exact ⟨a,b,c,d,e,f,g,h,i,j,k⟩

theorem checkView_iff (v : ViewMetadata) : checkView v=.ok () ↔ ViewContract v := by
  rw [view_iff]
  simp [checkView,ite_eq_iff]
  tauto

theorem contract_iff (f : Bool) (x y : ViewMetadata) : InputContract f x y ↔
    f=Bool.true ∧ ViewContract x ∧ ViewContract y ∧ count x=count y ∧ 0<count x ∧ count x<2^31 ∧
    x.device=y.device ∧ stride y≠0 ∧ (endByte x≤beginByte y ∨ endByte y≤beginByte x) := by
  constructor
  · intro h;exact ⟨h.scalar,h.input,h.output,h.shape,h.positive,h.extent,h.device,h.outputStride,h.separation⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩;exact ⟨a,b,c,d,e,f,g,h,i⟩

theorem checkInputs_iff (f : Bool) (x y : ViewMetadata) : checkInputs f x y=.ok () ↔ InputContract f x y := by
  rw [contract_iff]
  cases hx : checkView x <;> cases hy : checkView y <;> cases f <;>
    simp [checkInputs,hx,hy,←checkView_iff,ite_eq_iff,Except.bind,Except.pure] <;> omega

def buffer (v : ViewMetadata) : Buffer :=
  ⟨v.region,v.delta/4,v.shape,v.strides,some v.device⟩
@[simp] theorem buffer_region (v : ViewMetadata) : (buffer v).region = v.region := rfl
def capacity (v : ViewMetadata) : Nat := v.storageBytes/4

theorem shape_eq (v : ViewMetadata) (h : ViewContract v) : v.shape=[count v] := by
  have hr:=h.rank.1
  cases hs : v.shape with
  | nil => simp [hs] at hr
  | cons a rest =>
    have : rest=[] := by simpa [hs] using hr
    simp [this,count,hs]

theorem strides_eq (v : ViewMetadata) (h : ViewContract v) : v.strides=[stride v] := by
  have hr:=h.rank.2
  cases hs : v.strides with
  | nil => simp [hs] at hr
  | cons a rest =>
    have : rest=[] := by simpa [hs] using hr
    simp [this,stride,hs]

theorem coordinate_bounds (v : ViewMetadata) (i : Nat) (hi : i<count v) :
    lower v≤v.delta+4*(i:Int)*stride v ∧ v.delta+4*(i:Int)*stride v≤upper v := by
  have hn : (i:Int)≤((count v-1:Nat):Int) := by exact_mod_cast (by omega : i≤count v-1)
  have iz : (0:Int) ≤ i := by omega
  by_cases hs : 0≤stride v
  · have a:=mul_nonneg iz hs
    have b:=mul_le_mul_of_nonneg_right hn hs
    have c:=mul_nonneg (show (0:Int)≤((count v-1:Nat):Int) by omega) hs
    simp only [lower,upper,last,min_def,max_def]
    split_ifs <;> constructor <;> nlinarith
  · have neg : stride v≤0 := by omega
    have a:=mul_nonpos_of_nonneg_of_nonpos iz neg
    have b:=mul_le_mul_of_nonpos_right hn neg
    have c:=mul_nonpos_of_nonneg_of_nonpos (show (0:Int)≤((count v-1:Nat):Int) by omega) neg
    simp only [lower,upper,last,min_def,max_def]
    split_ifs <;> constructor <;> nlinarith

theorem view_valid (v : ViewMetadata) (h : ViewContract v) :
    ((buffer v).view (stride v)).Valid (count v) (capacity v) := by
  intro i hi
  obtain ⟨lo,hiBound⟩:=coordinate_bounds v i hi
  obtain ⟨lowerBound,upperBound⟩:=h.bounds (by omega)
  have aligned:=h.aligned.2
  have divided : 4*(v.delta/4)=v.delta := by omega
  change 0≤v.delta/4+(i:Int)*stride v ∧ v.delta/4+(i:Int)*stride v<(capacity v:Int)
  have capBound : (v.storageBytes:Int)<4*(capacity v:Int)+4 := by unfold capacity;omega
  constructor <;> nlinarith

/-- A logical cell in the proof is exactly the physical byte location queried
by the checked Python view, including its signed stride and storage offset. -/
theorem logical_cell_bytes (v : ViewMetadata) (h : ViewContract v)
    (i : Nat) (hi : i < count v) :
    (v.storageBase : Int) + 4 * (((buffer v).view (stride v)).address i : Int) =
      (v.storageBase : Int) + v.delta + 4 * (i : Int) * stride v := by
  rw [address_cast _ ((view_valid v h) i hi).1]
  change (v.storageBase : Int) + 4 * (v.delta / 4 + (i : Int) * stride v) = _
  have aligned := h.aligned.2
  have divided : 4 * (v.delta / 4) = v.delta := by omega
  nlinarith

/-- Optional historical regional interpretation with whole-backing coherence.
This is stronger than Python admission (DLPack storage objects can overlap
with unequal bases). Public kernel proofs use `Physical` instead and impose
no StorageBindings premise. Retained host corollaries state this condition. -/
structure StorageBindings (x y : ViewMetadata) : Prop where
  same : x.region=y.region → x.storageBase=y.storageBase
  separate : x.region≠y.region → x.storageBase+x.storageBytes≤y.storageBase ∨
    y.storageBase+y.storageBytes≤x.storageBase

def native_buffers (f : Bool) (x y : ViewMetadata) (h : InputContract f x y)
    (storage : StorageBindings x y) : NativeBuffers (buffer x) (buffer y) (capacity x) (capacity y) := by
  refine ⟨x.storageBase,y.storageBase,h.input.aligned.1,h.output.aligned.1,?_,?_,storage.same,?_⟩
  · change x.storageBase<2^64 ∧ x.storageBase+capacity x*4≤2^64
    have a:=h.input.allocation
    have valid:=view_valid x h.input 0 h.positive
    have positiveCapacity : 0<capacity x := by omega
    unfold capacity at *
    omega
  · change y.storageBase<2^64 ∧ y.storageBase+capacity y*4≤2^64
    have a:=h.output.allocation
    have valid:=view_valid y h.output 0 (by simpa [←h.shape] using h.positive)
    have positiveCapacity : 0<capacity y := by omega
    unfold capacity at *
    omega
  · intro ne
    have sep:=storage.separate ne
    unfold capacity
    omega

theorem footprint_separation (f : Bool) (x y : ViewMetadata) (h : InputContract f x y)
    (storage : StorageBindings x y) :
    FootprintsDisjoint ((buffer x).view (stride x)) ((buffer y).view (stride y)) (count x) := by
  intro i hi j hj
  by_cases different : x.region≠y.region
  · exact Or.inl different
  · right
    intro eq
    have sameBase:=storage.same (by simpa using different)
    have vx:=view_valid x h.input i hi
    have vy:=view_valid y h.output j (by simpa [←h.shape] using hj)
    have coords:=congrArg (fun n:Nat => (n:Int)) eq
    dsimp only at coords
    rw [address_cast _ vx.1,address_cast _ vy.1] at coords
    change x.delta/4+(i:Int)*stride x=y.delta/4+(j:Int)*stride y at coords
    obtain ⟨xl,xh⟩:=coordinate_bounds x i hi
    obtain ⟨yl,yh⟩:=coordinate_bounds y j (by simpa [←h.shape] using hj)
    have xa:=h.input.aligned.2
    have ya:=h.output.aligned.2
    have dividedX : 4*(x.delta/4)=x.delta := by omega
    have dividedY : 4*(y.delta/4)=y.delta := by omega
    have sep:=h.separation
    simp only [beginByte,endByte] at sep
    rw [sameBase] at sep
    rcases sep with sep|sep <;> nlinarith

#axiomsClean checkInputs_iff
#axiomsClean view_valid
#axiomsClean logical_cell_bytes
#axiomsClean native_buffers
#axiomsClean footprint_separation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked
