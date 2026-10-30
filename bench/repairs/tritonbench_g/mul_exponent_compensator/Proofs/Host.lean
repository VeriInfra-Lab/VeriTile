import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Native

/-! Successful torch.empty(src.shape, FP32, device='cuda') on the current
CUDA device. Allocation premises contain only ordinary Tensor/storage facts
and a fresh-allocation frame. Source/result values are derived below. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Host
open VeriTile.Triton
open VeriTile.Torch.FP32 (Tensor TensorLaws DataView dataView)
open VeriTile.Bench.Repairs.TritonMul2 (Operations Layout Allocated Memory launch)
open VeriTile.Bench.Repairs.TritonMul2.Host (slot logical slot_bound logical_contiguous active_tensor_bytes observeByte byte_frame logical_fp32)
noncomputable section
set_option maxHeartbeats 4000000

structure Prepared {α : Type} (x : Tensor) (lx : TensorLaws x)
    (current B : Nat) (h : InputContract x.metadata current B) (before : Memory α) where
  output : Tensor
  laws : TensorLaws output
  memory : Memory α
  shape : output.metadata.shape=x.metadata.shape
  dtype : output.metadata.dtype=.floating .fp32
  device : output.metadata.device=⟨"cuda",current⟩
  contiguous : output.metadata.isContiguous=Bool.true
  noNeg : output.metadata.isNeg=Bool.false
  aligned : 0<output.metadata.numel → output.dataPtr.toNat%4=0
  freshObject : output.object≠x.object
  fresh : slot (dataView output laws contiguous)+output.metadata.numel≤slot (dataView x lx h.contiguous) ∨
    slot (dataView x lx h.contiguous)+x.metadata.numel≤slot (dataView output laws contiguous)
  allocationFrame : ∀ cell,
    (¬∃ i,i<output.metadata.numel ∧ cell=slot (dataView output laws contiguous)+i) → memory cell=before cell

def layout (p : Prepared x lx current B h before) : Layout :=
  ⟨slot (dataView x lx h.contiguous),slot (dataView p.output p.laws p.contiguous),x.metadata.numel,B⟩

theorem same_numel (p : Prepared x lx current B h before) : p.output.metadata.numel=x.metadata.numel := by
  simp [VeriTile.Torch.TensorMetadata.numel,p.shape]

theorem allocation (p : Prepared x lx current B h before) : Allocated (layout p) := by
  refine ⟨slot_bound (dataView x lx h.contiguous),?_,Or.inr ?_⟩
  · have bound:=slot_bound (dataView p.output p.laws p.contiguous)
    change slot (dataView p.output p.laws p.contiguous)+x.metadata.numel≤2^62
    change slot (dataView p.output p.laws p.contiguous)+p.output.metadata.numel≤2^62 at bound
    rw [same_numel p] at bound
    exact bound
  · have sep:=p.fresh
    rw [same_numel p] at sep
    intro i j hi hj eq
    change slot (dataView p.output p.laws p.contiguous)+i=slot (dataView x lx h.contiguous)+j at eq
    change i<x.metadata.numel at hi
    change j<x.metadata.numel at hj
    omega

theorem input_preserved (p : Prepared x lx current B h before) (i : Nat) (hi : i<x.metadata.numel) :
    p.memory ((layout p).input+i)=before ((layout p).input+i) := by
  apply p.allocationFrame
  rintro ⟨j,hj,eq⟩
  rw [same_numel p] at hj
  have sep:=p.fresh
  rw [same_numel p] at sep
  change slot (dataView x lx h.contiguous)+i=slot (dataView p.output p.laws p.contiguous)+j at eq
  omega

/-- Records the allocation state before the original src.shape[0] inspection.
Rank-zero calls therefore fail after allocation, even with an invalid block. -/
def afterAllocation (allocated : Tensor × Memory α) (x : TensorMetadata) (current : Nat) (b : BlockArgument) :
    (Tensor × Memory α) × Except InputError Unit :=
  (allocated,checkInputs (.tensor x) current b)

theorem scalar_failure_after_allocation (allocated : Tensor × Memory α) (x : TensorMetadata)
    (rank : x.shape=[]) (current : Nat) (b : BlockArgument) :
    afterAllocation allocated x current b=(allocated,.error .rank) := by
  simp [afterAllocation,checkInputs,rank]

def runWrapper (ops : Operations α) {before : Memory α} (p : Prepared x lx current B h before)
    (order : List Nat) : Except InputError (Option (Tensor × Memory α)) := do
  checkTensorInputs x.metadata current B
  if x.metadata.numel=0 then return some (p.output,p.memory)
  return (Native.sourceLaunch ops (layout p) order p.memory).map (p.output,·)

theorem wrapper_correct (ops : Operations α) (x : Tensor) (lx : TensorLaws x)
    (current B : Nat) (h : InputContract x.metadata current B) (before : Memory α)
    (p : Prepared x lx current B h before) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,
      runWrapper ops p order=.ok (some (p.output,final)) ∧
      (∀ i,i<x.metadata.numel → logical final p.output i=ops.twice (logical before x i)) ∧
      (∀ cell,(¬∃ i,i<x.metadata.numel ∧ cell=(layout p).output+i) → final cell=before cell) := by
  have accepted := (checkTensorInputs_iff x.metadata current B).mpr h
  by_cases empty : x.metadata.numel=0
  · refine ⟨p.memory,by simp [runWrapper,accepted,empty],by simp [empty],?_⟩
    intro cell outside
    exact p.allocationFrame cell (by simpa [same_numel p,layout] using outside)
  · obtain ⟨exec,values,frame⟩ := Native.source_launch_correct ops (layout p)
      h.block h.grid (allocation p) order schedule p.memory
    refine ⟨launch ops (layout p) p.memory,?_,?_,?_⟩
    · simp [runWrapper,accepted,empty,exec]
    · intro i hi
      rw [logical_contiguous _ p.output p.laws p.contiguous i (by rw [same_numel p];exact hi)]
      change launch ops (layout p) p.memory ((layout p).output+i)=_
      rw [values i hi,input_preserved p i hi,logical_contiguous before x lx h.contiguous i hi]
      rfl
    · intro cell outside
      rw [frame cell outside]
      exact p.allocationFrame cell (by simpa [same_numel p,layout] using outside)

theorem input_pointer (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (h : InputContract x.metadata current B) (nonempty : 0<x.metadata.numel) :
    BitVec.ofNat 64 (4*slot (dataView x lx h.contiguous))=x.dataPtr := by
  have aligned:=VeriTile.Torch.FP32.dataView_aligned x lx h.contiguous h.alignment
  have eq : 4*slot (dataView x lx h.contiguous)=(dataView x lx h.contiguous).base := by unfold slot;omega
  rw [eq]
  exact VeriTile.Torch.FP32.actual_pointer x lx h.contiguous nonempty

theorem output_pointer (p : Prepared x lx current B h before) (nonempty : 0<x.metadata.numel) :
    BitVec.ofNat 64 (4*(layout p).output)=p.output.dataPtr := by
  have aligned:=VeriTile.Torch.FP32.dataView_aligned p.output p.laws p.contiguous p.aligned
  have eq : 4*(layout p).output=(dataView p.output p.laws p.contiguous).base := by
    change 4*slot (dataView p.output p.laws p.contiguous)=_
    unfold slot;omega
  rw [eq]
  exact VeriTile.Torch.FP32.actual_pointer p.output p.laws p.contiguous (by rw [same_numel p];exact nonempty)

theorem input_frame {before : Memory α} (p : Prepared x lx current B h before) (final : Memory α)
    (frame : ∀ cell,(¬∃ i,i<x.metadata.numel ∧ cell=(layout p).output+i) → final cell=before cell)
    (i : Nat) (hi : i<x.metadata.numel) : logical final x i=logical before x i := by
  rw [logical_contiguous _ x lx h.contiguous i hi,logical_contiguous _ x lx h.contiguous i hi]
  apply frame
  rintro ⟨j,hj,eq⟩
  have sep:=p.fresh
  rw [same_numel p] at sep
  change slot (dataView x lx h.contiguous)+i=slot (dataView p.output p.laws p.contiguous)+j at eq
  omega

#axiomsClean scalar_failure_after_allocation
#axiomsClean wrapper_correct
#axiomsClean input_pointer
#axiomsClean output_pointer
#axiomsClean input_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Host
