import bench.repairs.tritonbench_g.triton_mul2.Proofs.Native
import VeriTile.Torch.FP32

/-! Successful Tensor API boundary. Allocator facts mention metadata, freshness,
and memory frames only. The input is used directly without a materialization.
The memory model uses aligned four-byte slots, with byte observations below. -/
namespace VeriTile.Bench.Repairs.TritonMul2.Host
open VeriTile.Triton
open VeriTile.Torch.FP32 (Tensor TensorLaws DataView dataView)
noncomputable section
set_option maxHeartbeats 4000000

def logical (s : Memory α) (t : Tensor) (i : Nat) : α :=
  s (t.storage.address (t.physicalIndex i)/4)
def slot (v : DataView) : Nat := v.base/4

theorem slot_bound (v : DataView) : slot v+v.elements≤2^62 := by
  have span:=v.spanFits
  unfold slot
  omega

theorem logical_contiguous (s : Memory α) (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    logical s t i=s (slot (dataView t laws contiguous)+i) := by
  have nonempty : t.metadata.numel≠0 := by omega
  unfold logical slot
  rw [laws.contiguous contiguous i hi]
  simp only [dataView,DataView.base,if_neg nonempty,VeriTile.Torch.FP32.Storage.address]
  congr 1
  omega

/-- Successful zeros_like returns output; in-place passes x directly. Metadata
laws, allocator freshness and frame are ordinary API contracts. No field
mentions doubling or a kernel result. Zero initialization is unnecessary for
correctness because the proof establishes complete output coverage. -/
structure Prepared {α : Type} (x : Tensor) (lx : TensorLaws x)
    (B : Nat) (h : InputContract x.metadata B) (inplace : Bool) (before : Memory α) where
  output : Tensor
  laws : TensorLaws output
  current : Memory α
  metadata : output.metadata={x.metadata with isNeg:=Bool.false,dataPtr:=output.dataPtr}
  contiguous : output.metadata.isContiguous=Bool.true
  aligned : 0<output.metadata.numel → output.dataPtr.toNat%4=0
  inplaceObject : inplace=Bool.true → output=x
  inplaceMemory : inplace=Bool.true → current=before
  freshObject : inplace=Bool.false → output.object≠x.object
  fresh : inplace=Bool.false →
    slot (dataView output laws contiguous)+output.metadata.numel≤slot (dataView x lx h.contiguous) ∨
    slot (dataView x lx h.contiguous)+x.metadata.numel≤slot (dataView output laws contiguous)
  allocationFrame : inplace=Bool.false → ∀ cell,
    (¬∃ i,i<output.metadata.numel ∧ cell=slot (dataView output laws contiguous)+i) →
    current cell=before cell

def layout (p : Prepared x lx B h inplace before) : Layout :=
  ⟨slot (dataView x lx h.contiguous),slot (dataView p.output p.laws p.contiguous),x.metadata.numel,B⟩

theorem same_numel (p : Prepared x lx B h inplace before) :
    p.output.metadata.numel=x.metadata.numel := by
  simp [p.metadata,VeriTile.Torch.TensorMetadata.numel]

theorem allocation (p : Prepared x lx B h inplace before) : Allocated (layout p) := by
  refine ⟨slot_bound (dataView x lx h.contiguous),?_,?_⟩
  · have bound:=slot_bound (dataView p.output p.laws p.contiguous)
    change slot (dataView p.output p.laws p.contiguous)+x.metadata.numel≤2^62
    change slot (dataView p.output p.laws p.contiguous)+p.output.metadata.numel≤2^62 at bound
    rw [same_numel p] at bound
    exact bound
  · cases inplace with
    | «true» => left; simp only [layout,p.inplaceObject rfl,dataView,slot]
    | «false» =>
      right
      have sep:=p.fresh rfl
      rw [same_numel p] at sep
      intro i j hi hj eq
      change slot (dataView p.output p.laws p.contiguous)+i=slot (dataView x lx h.contiguous)+j at eq
      change i<x.metadata.numel at hi
      change j<x.metadata.numel at hj
      omega

theorem wired (p : Prepared x lx B h inplace before) : Native.Wired inplace (layout p) := by
  intro yes
  subst inplace
  simp only [layout,p.inplaceObject rfl,dataView,slot]

theorem input_preserved (p : Prepared x lx B h inplace before) (i : Nat) (hi : i<x.metadata.numel) :
    p.current ((layout p).input+i)=before ((layout p).input+i) := by
  cases inplace with
  | «true» => rw [p.inplaceMemory rfl]
  | «false» =>
    apply p.allocationFrame rfl
    rintro ⟨j,hj,eq⟩
    rw [same_numel p] at hj
    have sep:=p.fresh rfl
    rw [same_numel p] at sep
    change slot (dataView x lx h.contiguous)+i=slot (dataView p.output p.laws p.contiguous)+j at eq
    omega

def runWrapper (ops : Operations α) {before : Memory α} (p : Prepared x lx B h inplace before)
    (order : List Nat) : Except InputError (Option (Tensor × Memory α)) := do
  checkTensorInputs x.metadata B
  if x.metadata.numel=0 then return some (p.output,p.current)
  return (Native.sourceLaunch ops (layout p) inplace order p.current).map (p.output,·)

theorem wrapper_correct (ops : Operations α) (x : Tensor) (lx : TensorLaws x)
    (B : Nat) (h : InputContract x.metadata B) (inplace : Bool) (before : Memory α)
    (p : Prepared x lx B h inplace before) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,
      runWrapper ops p order=.ok (some (p.output,final)) ∧
      (∀ i,i<x.metadata.numel → logical final p.output i=ops.twice (logical before x i)) ∧
      (∀ cell,(¬∃ i,i<x.metadata.numel ∧ cell=(layout p).output+i) → final cell=before cell) ∧
      Native.Wired inplace (layout p) := by
  have accepted := (checkTensorInputs_iff x.metadata B).mpr h
  by_cases empty : x.metadata.numel=0
  · refine ⟨p.current,by simp [runWrapper,accepted,empty],by simp [empty],?_,wired p⟩
    intro cell outside
    cases inplace with
    | «true» => rw [p.inplaceMemory rfl]
    | «false» => exact p.allocationFrame rfl cell (by simpa [same_numel p,layout] using outside)
  · obtain ⟨exec,values,frame⟩ := Native.source_launch_correct ops (layout p) inplace
      h.extent h.block (allocation p) order schedule p.current
    refine ⟨launch ops (layout p) p.current,?_,?_,?_,wired p⟩
    · simp [runWrapper,accepted,empty,exec]
    · intro i hi
      rw [logical_contiguous _ p.output p.laws p.contiguous i (by rw [same_numel p];exact hi)]
      change launch ops (layout p) p.current ((layout p).output+i)=_
      have value := values i hi
      dsimp at value
      rw [value,input_preserved p i hi,logical_contiguous before x lx h.contiguous i hi]
      rfl
    · intro cell outside
      rw [frame cell outside]
      cases inplace with
      | «true» => rw [p.inplaceMemory rfl]
      | «false» => exact p.allocationFrame rfl cell (by simpa [same_numel p,layout] using outside)

/-- The finite source starts from the checked actual data pointer, including
nonzero contiguous storage offsets. Empty tensors do not query a pointer. -/
theorem input_pointer (x : Tensor) (lx : TensorLaws x) (B : Nat)
    (h : InputContract x.metadata B) (nonempty : 0<x.metadata.numel) :
    BitVec.ofNat 64 (4*slot (dataView x lx h.contiguous))=x.dataPtr := by
  have aligned:=VeriTile.Torch.FP32.dataView_aligned x lx h.contiguous h.alignment
  have eq : 4*slot (dataView x lx h.contiguous)=(dataView x lx h.contiguous).base := by
    unfold slot;omega
  rw [eq]
  exact VeriTile.Torch.FP32.actual_pointer x lx h.contiguous nonempty

/-- Active FP32 accesses retain the actual contiguous Tensor storage offset,
are naturally aligned and keep every byte within the allocation and u64 range. -/
theorem active_tensor_bytes (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true)
    (checked : 0<t.metadata.numel → t.dataPtr.toNat%4=0)
    (i byte : Nat) (hi : i<t.metadata.numel) (hb : byte<4) :
    let address := 4*slot (dataView t laws contiguous)+4*i
    address%4=0 ∧
    address+byte=t.storage.address (t.physicalIndex i)+byte ∧
    t.storage.base≤address+byte ∧
    address+byte<t.storage.base+4*t.storage.elements ∧ address+byte<2^64 := by
  have aligned:=VeriTile.Torch.FP32.dataView_aligned t laws contiguous checked
  have inside:=laws.inBounds i hi
  have span:=t.storage.spanFits
  have nonempty : t.metadata.numel≠0 := by omega
  have eq : 4*slot (dataView t laws contiguous)+4*i=t.storage.address (t.physicalIndex i) := by
    rw [laws.contiguous contiguous i hi]
    simp only [slot,dataView,DataView.base,if_neg nonempty,
      VeriTile.Torch.FP32.Storage.address] at aligned ⊢
    omega
  dsimp only
  rw [eq]
  simp only [VeriTile.Torch.FP32.Storage.address] at *
  have modZero : (t.storage.base+4*t.physicalIndex i)%4=0 := by
    rw [←eq]
    omega
  exact ⟨modZero,True.intro,by omega,by omega,by omega⟩

theorem output_pointer (p : Prepared x lx B h inplace before) (nonempty : 0<x.metadata.numel) :
    BitVec.ofNat 64 (4*(layout p).output)=p.output.dataPtr := by
  have aligned:=VeriTile.Torch.FP32.dataView_aligned p.output p.laws p.contiguous p.aligned
  have eq : 4*(layout p).output=(dataView p.output p.laws p.contiguous).base := by
    change 4*slot (dataView p.output p.laws p.contiguous)=_
    unfold slot;omega
  rw [eq]
  exact VeriTile.Torch.FP32.actual_pointer p.output p.laws p.contiguous
    (by rw [same_numel p];exact nonempty)

/-- A byte observation is a projection of the FP32 word containing that byte;
this includes byte aliases and words outside the Tensor's logical view. -/
def observeByte (s : Memory (FP.Value .fp32)) (address : Nat) : BitVec 8 :=
  BitVec.ofNat 8 ((s (address/4)).bits.toNat / 2^(8*(address%4)))

theorem byte_frame (before final : Memory (FP.Value .fp32)) (base count : Nat)
    (frame : ∀ cell,(¬∃ i,i<count ∧ cell=base+i) → final cell=before cell)
    (address : Nat) (outside : address<4*base ∨ 4*(base+count)≤address) :
    observeByte final address=observeByte before address := by
  unfold observeByte
  rw [frame (address/4) (by rintro ⟨i,hi,eq⟩;omega)]

theorem logical_fp32 (s : Memory (FP.Value .fp32)) (t : Tensor) (laws : TensorLaws t)
    (noNeg : t.metadata.isNeg=Bool.false) (i : Nat) :
    VeriTile.Torch.FP32.logicalValue (fun a => s (a/4)) t laws i=logical s t i := by
  simp [VeriTile.Torch.FP32.logicalValue,noNeg,logical]

#axiomsClean wrapper_correct
#axiomsClean input_pointer
#axiomsClean active_tensor_bytes
#axiomsClean output_pointer
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonMul2.Host
