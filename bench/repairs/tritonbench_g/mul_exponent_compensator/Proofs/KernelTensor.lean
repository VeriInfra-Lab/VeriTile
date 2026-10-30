import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Native

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.KernelTensor
open VeriTile.Triton
open VeriTile.Torch.FP32 (Tensor TensorLaws DataView dataView)
open VeriTile.Bench.Repairs.TritonMul2 (Memory Operations Layout Allocated launch)
open TritonMul2.Host (slot logical slot_bound logical_contiguous active_tensor_bytes)
noncomputable section

/-- Ordinary facts about the actual buffers supplied to this kernel. No field
mentions execution, multiplication, output values, allocation or return identity.
Input offset views and empty out-of-storage offsets remain admitted. -/
structure Buffers (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (admit : InputContract x.metadata current B) where
  output : Tensor
  laws : TensorLaws output
  count : output.metadata.numel=x.metadata.numel
  contiguous : output.metadata.isContiguous=Bool.true
  noNeg : output.metadata.isNeg=Bool.false
  aligned : 0<output.metadata.numel → output.dataPtr.toNat%4=0
  separate : slot (dataView output laws contiguous)+output.metadata.numel≤slot (dataView x lx admit.contiguous) ∨
    slot (dataView x lx admit.contiguous)+x.metadata.numel≤slot (dataView output laws contiguous)

def layout (p : Buffers x lx current B admit) : Layout :=
  ⟨slot (dataView x lx admit.contiguous),slot (dataView p.output p.laws p.contiguous),x.metadata.numel,B⟩

theorem allocation (p : Buffers x lx current B admit) : Allocated (layout p) := by
  refine ⟨slot_bound (dataView x lx admit.contiguous),?_,Or.inr ?_⟩
  · have bound:=slot_bound (dataView p.output p.laws p.contiguous)
    change slot (dataView p.output p.laws p.contiguous)+x.metadata.numel≤2^62
    change slot (dataView p.output p.laws p.contiguous)+p.output.metadata.numel≤2^62 at bound
    rw [p.count] at bound
    exact bound
  · have separate:=p.separate
    rw [p.count] at separate
    intro i j hi hj eq
    change slot (dataView p.output p.laws p.contiguous)+i=slot (dataView x lx admit.contiguous)+j at eq
    change i<x.metadata.numel at hi
    change j<x.metadata.numel at hj
    omega

def Frame (p : Buffers x lx current B admit) (before final : Memory α) : Prop :=
  ∀ cell,(¬∃ i,i<x.metadata.numel ∧ cell=(layout p).output+i) → final cell=before cell

theorem input_frame (p : Buffers x lx current B admit) (before final : Memory α)
    (frame : Frame p before final) (i : Nat) (hi : i<x.metadata.numel) :
    logical final x i=logical before x i := by
  rw [logical_contiguous final x lx admit.contiguous i hi,
    logical_contiguous before x lx admit.contiguous i hi]
  apply frame
  rintro ⟨j,hj,eq⟩
  have separate:=p.separate
  rw [p.count] at separate
  change slot (dataView x lx admit.contiguous)+i=slot (dataView p.output p.laws p.contiguous)+j at eq
  omega

/-- Executes the lowered complete readable DSL with the native widened i64
index/mask/address semantics. All complete program orders are covered. -/
theorem execute (ops : Operations α) (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (admit : InputContract x.metadata current B) (p : Buffers x lx current B admit)
    (before : Memory α) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,Native.sourceLaunch ops (layout p) order before=some final ∧
      (∀ i,i<x.metadata.numel → logical final p.output i=ops.twice (logical before x i)) ∧
      Frame p before final ∧
      (∀ i,i<x.metadata.numel → logical final x i=logical before x i) := by
  obtain ⟨execution,values,frame⟩ := Native.source_launch_correct ops (layout p)
    admit.block admit.grid (allocation p) order schedule before
  refine ⟨launch ops (layout p) before,execution,?_,frame,input_frame p before _ frame⟩
  intro i hi
  rw [logical_contiguous _ p.output p.laws p.contiguous i (by rw [p.count];exact hi),
    logical_contiguous before x lx admit.contiguous i hi]
  exact values i hi

theorem pointer (t : Tensor) (laws : TensorLaws t) (contiguous : t.metadata.isContiguous=Bool.true)
    (aligned : 0<t.metadata.numel → t.dataPtr.toNat%4=0) (nonempty : 0<t.metadata.numel) :
    BitVec.ofNat 64 (4*slot (dataView t laws contiguous))=t.dataPtr := by
  have h:=VeriTile.Torch.FP32.dataView_aligned t laws contiguous aligned
  have eq : 4*slot (dataView t laws contiguous)=(dataView t laws contiguous).base := by unfold slot;omega
  rw [eq]
  exact VeriTile.Torch.FP32.actual_pointer t laws contiguous nonempty

theorem actual_pointers (p : Buffers x lx current B admit) (nonempty : 0<x.metadata.numel) :
    BitVec.ofNat 64 (4*(layout p).input)=x.dataPtr ∧
    BitVec.ofNat 64 (4*(layout p).output)=p.output.dataPtr :=
  ⟨pointer x lx admit.contiguous admit.alignment nonempty,
    pointer p.output p.laws p.contiguous p.aligned (by rw [p.count];exact nonempty)⟩

/-- Every active source/destination byte is the corresponding Tensor byte and
lies inside its live backing. Alignment is derived from the observed pointers. -/
theorem actual_bytes (p : Buffers x lx current B admit) (i byte : Nat)
    (hi : i<x.metadata.numel) (hb : byte<4) :
    (let address:=4*(layout p).input+4*i;
      address%4=0 ∧ address+byte=x.storage.address (x.physicalIndex i)+byte ∧
      x.storage.base≤address+byte ∧ address+byte<x.storage.base+4*x.storage.elements ∧ address+byte<2^64) ∧
    (let address:=4*(layout p).output+4*i;
      address%4=0 ∧ address+byte=p.output.storage.address (p.output.physicalIndex i)+byte ∧
      p.output.storage.base≤address+byte ∧
      address+byte<p.output.storage.base+4*p.output.storage.elements ∧ address+byte<2^64) :=
  ⟨active_tensor_bytes x lx admit.contiguous admit.alignment i byte hi hb,
    active_tensor_bytes p.output p.laws p.contiguous p.aligned i byte (by rw [p.count];exact hi) hb⟩

/-- All 21 admitted blocks use the same signed i64 source semantics, including
masked lanes. The statement does not impose a new numel<2^31 guard. -/
theorem formed_indices (p : Buffers x lx current B admit) (pid lane : Nat)
    (hp : pid<grid x.metadata.numel B) (hl : lane<B) :
    (Native.offset B pid lane).toInt=(pid*B+lane:Nat) ∧
      Native.active x.metadata.numel B pid lane=decide (pid*B+lane<x.metadata.numel) := by
  have allocated:=allocation p
  have extent : x.metadata.numel≤2^62 := by have := allocated.input;change _+x.metadata.numel≤2^62 at this;omega
  exact Native.indices _ _ _ _ extent admit.block admit.grid hp hl

theorem empty (ops : Operations α) (p : Buffers x lx current B admit)
    (before : Memory α) (zero : x.metadata.numel=0) :
    Native.sourceLaunch ops (layout p) (List.range (grid x.metadata.numel B)) before=some before := by
  have positive:=block_positive B admit.block
  have hg : grid x.metadata.numel B=0 := by simp [grid,zero,Nat.div_eq_of_lt (by omega : B-1<B)]
  simp [hg,Native.sourceLaunch]

#axiomsClean execute
#axiomsClean actual_pointers
#axiomsClean actual_bytes
#axiomsClean formed_indices
#axiomsClean empty
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.KernelTensor
