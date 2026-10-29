import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.CheckedGeometry
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.Bytes
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedNative

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
open VeriTile.Bench.Repairs.RmsMatmulRbe (Shape Parameters RowInput DotPlan Reduction)
open VeriTile.Bench.Repairs.RmsMatmulRbe.Contract (TensorAdmitted ScalarAdmitted fp32 fp16 int8 width)
open VeriTile.Bench.Repairs.RmsMatmulRbe.Checked (flat3 flat3_lt pointer64 pointer_exact)
open VeriTile.Bench.Repairs.RmsRbeMatmul.Contract (Inputs InputContract)
noncomputable section
set_option maxHeartbeats 6000000

def parameters (i : Inputs) (scalar : ℝ → F) : Parameters F :=
  ⟨(shape i).K,(shape i).headDim,i.scalars.start,scalar (1/1000000),scalar 10000⟩
def sourceArguments (i : Inputs) : RmsMatmulRbe.TypedSource.Arguments :=
  let s := shape i
  ⟨"X","W","RMS","OUT",s.M,s.N,s.K,s.M*s.K,s.K,1,1,s.K,1,s.M*s.N,s.N,1,
    16,64,64,s.headDim,i.scalars.start,i.rotary,10000,1/1000000⟩
def typedKernel (i : Inputs) : VeriTile.Triton.ComputeKernel :=
  TypedSource.kernel (decide (i.weight.metadata.dtype=int8)) (sourceArguments i)
structure Compiler (F H : Type) (i : Inputs) (a : Allocation i) (scalar : ℝ → F) where
  ast : String
  sourceIdentity : ast=productionAST
  source : VeriTile.Triton.ComputeKernel
  typedIdentity : source=typedKernel i
  program : Code F H Unit
  lowered : program=gridCode (parameters i scalar) (buffers i a)
    selectedPlans selectedReductions (schedule (shape i)).events
def Compiler.legacy (c : Compiler F H i a scalar) : CompilerBinding F H (shape i) (parameters i scalar) (buffers i a) where
  ast := c.ast
  sourceIdentity := c.sourceIdentity
  schedule := schedule (shape i)
  physicalLanes := lanes (shape i)
  dispatched := lanes_cells (shape i)
  program := c.program
  lowered := c.lowered

def dtype (d : VeriTile.Torch.DType) : DType := if d=fp32 then .fp32 else if d=fp16 then .fp16 else .int8
def sourceDType (i : Inputs) := dtype i.rms.metadata.dtype
theorem dtype_weight (i : Inputs) (h : InputContract i) (a : Allocation i) :
    dtype i.weight.metadata.dtype=weightDType (buffers i a) := by
  have hmem := (facts i h).weightAdmitted.2.2.1
  simp only [RmsMatmulRbe.Contract.weightTypes,List.mem_cons,List.not_mem_nil,or_false] at hmem
  rcases hmem with ht|ht <;> simp [dtype,ht,buffers,weightDType,fp32,fp16,int8]
theorem dtype_same_iff (i : Inputs) (h : InputContract i) (a : Allocation i) :
    sourceDType i=weightDType (buffers i a) ↔ i.rms.metadata.dtype=i.weight.metadata.dtype := by
  rw [←dtype_weight i h a]
  have hr := (facts i h).rmsAdmitted.2.2.1
  have hw := (facts i h).weightAdmitted.2.2.1
  simp only [Contract.rmsTypes,RmsMatmulRbe.Contract.weightTypes,List.mem_cons,List.not_mem_nil,or_false] at hr hw
  rcases hr with hr|hr|hr <;> rcases hw with hw|hw <;>
    simp [sourceDType,dtype,hr,hw,fp32,fp16,int8]

/-- Ordinary preparation: preserve all original storage, evaluate precisely
the original Tensor.to scalar conversion, allocate only its result and OUT. -/
structure HostPreparation (i : Inputs) (a : Allocation i) (cast : HostCast F H)
    (before prepared : Memory F H) : Prop where
  original : ∀ t,t∈i.reads → ∀ offset,prepared (t.base,offset)=before (t.base,offset)
  castOne : ∀ k,k<(shape i).K → prepared (a.rms.base,a.rms.offset+k)=
    castWord cast (sourceDType i) (weightDType (buffers i a)) (before (i.rms.base,i.rms.offset+k))
  frame : ∀ q,q.1≠a.out.base → (sourceDType i=weightDType (buffers i a) ∨ q.1≠a.rms.base) → prepared q=before q
theorem HostPreparation.legacy (i : Inputs) (h : InputContract i) (a : Allocation i)
    (cast : HostCast F H) (before prepared : Memory F H) (prep : HostPreparation i a cast before prepared) :
    Preparation (shape i).K cast (sourceDType i) (dense1 i.rms) (buffers i a) before prepared := by
  refine ⟨?_,?_,?_,?_,?_,prep.frame⟩
  · intro batch row k
    exact prep.original i.x (by simp [Contract.Inputs.reads]) _
  · intro n k
    exact prep.original i.weight (by simp [Contract.Inputs.reads]) _
  · intro k hk
    simpa [buffers,dense1,View1.address] using prep.castOne k hk
  · intro hs
    have same := a.same ((dtype_same_iff i h a).mp hs)
    simp [buffers,same]
  · intro hs
    have diff := fun he => hs ((dtype_same_iff i h a).mpr he)
    have hr := (a.rmsOrdinary.positiveBytes (by rw [a.rmsCount]; exact (facts i h).valid.1))
    have ho := (a.ordinary i.rms (by simp [Contract.Inputs.reads])).positiveBytes
      (by rw [(facts i h).rCount]; exact (facts i h).valid.1)
    exact RmsMatmulRbe.Checked.disjoint_base_ne _ _ (a.castFresh diff i.rms (by simp [Contract.Inputs.reads])) hr ho

structure Returns (i : Inputs) (a : Allocation i) where
  tensor : RmsMatmulRbe.Contract.Tensor
  view : RmsMatmulRbe.Checked.View a.out tensor
    [(shape i).B,(shape i).M,(shape i).heads,(shape i).headDim]
def entry (abi : LocalEventABI F H σ ops) (c : Compiler F H i a scalar)
    (returns : Returns i a) (state : σ) : RmsMatmulRbe.Contract.Tensor × σ :=
  (returns.tensor,((hostCode (shape i) c.legacy).native abi state).2)
def tensorWord (memory : Memory F H) (t : RmsMatmulRbe.Contract.Tensor) (index : Nat) : Stored F H :=
  memory (t.base,t.offset+index)
theorem return_address (returns : Returns i a) (x : Cell (shape i)) :
    (returns.tensor.base,returns.tensor.offset+flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2)=
      outputAddress (buffers i a) x := by
  rw [returns.view.base,returns.view.offset]
  exact (dense3_address _ _ _ _ _ _).symm

theorem native_words (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (castA castR : HostCast (BitVec 32) (BitVec 16)) (castLeaf : HostCastLeaf castA castR)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (before prepared : σ) (prep : HostPreparation i a castA (abi.observe before) (abi.observe prepared)) :
    (∀ x : Cell (shape i),tensorWord (abi.observe (entry abi c returns prepared).2)
      (entry abi c returns prepared).1 (flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2)=
      hostTarget reference castR (sourceDType i) (dense1 i.rms) (parameters i scalar) (buffers i a)
        (abi.observe before) x) ∧
    (∀ q,q.1≠a.out.base → (sourceDType i=weightDType (buffers i a) ∨ q.1≠a.rms.base) →
      abi.observe (entry abi c returns prepared).2 q=abi.observe before q) := by
  have frame (q) (outside : ∀ x : Cell (shape i),q≠outputAddress (buffers i a) x) :
      abi.observe (entry abi c returns prepared).2 q=abi.observe prepared q := by
    change abi.observe ((hostCode (shape i) c.legacy).native abi prepared).2 q=_
    rw [hostCode_eq,c.legacy.lowered,native_grid]
    exact run_frame actual _ _ _ _ _ _ q outside
  constructor
  · intro x
    have active : 0<(shape i).B ∧ 0<(shape i).M := ⟨by have := x.1.isLt; omega,by have := x.2.1.isLt; omega⟩
    have result := (bound_native_complete actual reference leaves abi (parameters i scalar) (buffers i a)
      (buffers_separate i h a active) (output_injective i a) c.legacy prepared).1 x
    change abi.observe ((hostCode (shape i) c.legacy).native abi prepared).2
      (returns.tensor.base,returns.tensor.offset+_) = _
    rw [return_address returns x,hostCode_eq]
    exact result.trans (prepared_target actual reference castA castR castLeaf (sourceDType i) (dense1 i.rms)
      (parameters i scalar) (buffers i a) _ _ (HostPreparation.legacy i h a castA _ _ prep) x)
  · intro q fresh old
    rw [frame q (fun x he => fresh (congrArg Prod.fst he))]
    exact prep.frame q fresh old

theorem scalar_bytes (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (state : σ) (q : Address) (lane : Nat) (hlane : lane<raw.width q.1) :
    raw.bytes state (q.1,raw.width q.1*q.2+lane)=cellByte (abi.observe state q) lane := by
  rw [raw.decode]
  simp only [Nat.mul_add_div (raw.positive _),Nat.div_eq_of_lt hlane,Nat.add_zero,
    Nat.mul_add_mod,Nat.mod_eq_of_lt hlane,Prod.mk.eta]
structure PhysicalABI (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (a : Allocation i) extends RawByteABI abi where
  physicalByte : σ → Nat → Byte
  load32 : σ → Nat → BitVec 32
  load16 : σ → Nat → BitVec 16
  load8 : σ → Nat → Byte
  load32Bytes : ∀ state pointer lane,lane<4 →
    cellByte (.inr (.inl (load32 state pointer))) lane=physicalByte state (pointer+lane)
  load16Bytes : ∀ state pointer lane,lane<2 →
    cellByte (.inr (.inr (load16 state pointer))) lane=physicalByte state (pointer+lane)
  load8Byte : ∀ state pointer,load8 state pointer=physicalByte state pointer
  bytesAt : ∀ state t,t∈i.reads++[a.rms,a.out] → ∀ byte,byte<t.bytes →
    toRawByteABI.bytes state (t.base,byte)=physicalByte state (t.base+byte)
  widthAt : ∀ t,t∈i.reads++[a.rms,a.out] → 0<t.bytes → toRawByteABI.width t.base=RmsMatmulRbe.Contract.width t.metadata.dtype
  input32 : ∀ state t,t∈i.reads++[a.rms] → t.metadata.dtype=fp32 → ∀ j,j<t.metadata.numel →
    abi.observe state (t.base,t.offset+j)=.inr (.inl (load32 state (pointer64 t j).toNat))
  input16 : ∀ state t,t∈i.reads++[a.rms] → t.metadata.dtype=fp16 → ∀ j,j<t.metadata.numel →
    abi.observe state (t.base,t.offset+j)=.inr (.inr (load16 state (pointer64 t j).toNat))
  input8 : ∀ state t,t∈i.reads++[a.rms] → t.metadata.dtype=int8 → ∀ j,j<t.metadata.numel →
    abi.observe state (t.base,t.offset+j)=.inl (load8 state (pointer64 t j).toNat)

theorem native_return_bytes (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (castA castR : HostCast (BitVec 32) (BitVec 16)) (castLeaf : HostCastLeaf castA castR)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (physical : PhysicalABI abi i a) (before prepared : σ)
    (prep : HostPreparation i a castA (abi.observe before) (abi.observe prepared))
    (x : Cell (shape i)) (lane : Nat) (hlane : lane<width a.out.metadata.dtype) :
    physical.physicalByte (entry abi c returns prepared).2
      ((pointer64 returns.tensor (flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2)).toNat+lane)=
      cellByte (hostTarget reference castR (sourceDType i) (dense1 i.rms) (parameters i scalar)
        (buffers i a) (abi.observe before) x) lane := by
  let index := flat3 (shape i).M (shape i).N x.1 x.2.1 x.2.2
  have ht : a.out∈i.reads++[a.rms,a.out] := by simp
  have hj : index<a.out.metadata.numel := by
    rw [a.outCount]; exact flat3_lt _ _ _ _ _ _ x.1.isLt x.2.1.isLt x.2.2.isLt
  have hexact := (pointer_exact a.out a.outOrdinary index hj).1
  have hbyte : width a.out.metadata.dtype*(a.out.offset+index)+lane<a.out.bytes := by
    have := a.outOrdinary.span
    have := RmsMatmulRbe.Checked.width_positive a.out.metadata.dtype
    nlinarith
  have hw := physical.widthAt a.out ht (by omega)
  have hwords := (native_words actual reference leaves castA castR castLeaf abi i h a scalar c returns before prepared prep).1 x
  have hp : pointer64 returns.tensor index=pointer64 a.out index := by
    simp only [pointer64,returns.view.pointer,returns.view.dtype]
  change physical.physicalByte _ ((pointer64 returns.tensor index).toNat+lane)=_
  rw [hp,hexact,Nat.add_assoc,←physical.bytesAt _ a.out ht _ hbyte,←hw]
  rw [scalar_bytes abi physical.toRawByteABI _ (a.out.base,a.out.offset+index) lane (by simpa [hw] using hlane)]
  congr 1
  simpa only [tensorWord,entry,Prod.fst,returns.view.base,returns.view.offset] using hwords

theorem native_backing_frame (actual : StorageOps (BitVec 32) (BitVec 16))
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (physical : PhysicalABI abi i a) (before prepared : σ)
    (allocationFrame : ∀ t,t∈i.reads → ∀ byte,byte<t.bytes →
      physical.toRawByteABI.bytes prepared (t.base,byte)=physical.toRawByteABI.bytes before (t.base,byte))
    (t : RmsMatmulRbe.Contract.Tensor) (ht : t∈i.reads) (byte : Nat) (hbyte : byte<t.bytes)
    (outside : ∀ x : Cell (shape i),(t.base,byte/physical.toRawByteABI.width t.base)≠outputAddress (buffers i a) x) :
    physical.physicalByte (entry abi c returns prepared).2 (t.base+byte)=physical.physicalByte before (t.base+byte) := by
  have ht' : t∈i.reads++[a.rms,a.out] := by simp [ht]
  rw [←physical.bytesAt _ t ht' byte hbyte,←physical.bytesAt before t ht' byte hbyte]
  change physical.toRawByteABI.bytes ((hostCode (shape i) c.legacy).native abi prepared).2 _=_
  rw [native_full_byte_frame actual abi physical.toRawByteABI c.legacy prepared t.base byte outside]
  exact allocationFrame t ht byte hbyte

theorem native_metadata (abi : LocalEventABI F H σ ops) (i : Inputs)
    (a : Allocation i) (scalar : ℝ → F) (c : Compiler F H i a scalar) (returns : Returns i a) (state : σ) :
    (entry abi c returns state).1.metadata.shape=[(shape i).B,(shape i).M,(shape i).heads,(shape i).headDim] ∧
    (entry abi c returns state).1.metadata.dtype=i.weight.metadata.dtype ∧
    (entry abi c returns state).1.metadata.device=i.x.metadata.device ∧
    (entry abi c returns state).1.metadata.dataPtr=a.out.metadata.dataPtr :=
  ⟨returns.view.shape,returns.view.dtype.trans a.outDType,returns.view.device.trans a.outDevice,returns.view.pointer⟩

theorem input_outside (i : Inputs) (a : Allocation i)
    (t : RmsMatmulRbe.Contract.Tensor) (ht : t∈i.reads) (positive : 0<t.bytes) (offset : Nat) :
    ∀ x : Cell (shape i),(t.base,offset)≠outputAddress (buffers i a) x := by
  intro x he
  have count : 0<a.out.metadata.numel := by
    rw [a.outCount]
    exact Nat.mul_pos (Nat.mul_pos (by have := x.1.isLt; omega) (by have := x.2.1.isLt; omega))
      (by have := x.2.2.isLt; omega)
  have disj := RmsMatmulRbe.Checked.disjoint_symm _ _ (a.fresh t (by simp [ht]))
  exact RmsMatmulRbe.Checked.disjoint_base_ne _ _ disj positive (a.outOrdinary.positiveBytes count) (congrArg Prod.fst he)

theorem native_inputs_frame (actual : StorageOps (BitVec 32) (BitVec 16))
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (physical : PhysicalABI abi i a) (before prepared : σ)
    (allocationFrame : ∀ t,t∈i.reads → ∀ byte,byte<t.bytes →
      physical.toRawByteABI.bytes prepared (t.base,byte)=physical.toRawByteABI.bytes before (t.base,byte)) :
    ∀ t,t∈i.reads → ∀ byte,byte<t.bytes →
      physical.physicalByte (entry abi c returns prepared).2 (t.base+byte)=physical.physicalByte before (t.base+byte) := by
  intro t ht byte hb
  exact native_backing_frame actual abi i a scalar c returns physical before prepared allocationFrame t ht byte hb
    (input_outside i a t ht (by omega) _)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
