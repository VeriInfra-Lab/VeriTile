import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedIndex
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Bytes

namespace VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
noncomputable section
set_option maxHeartbeats 6000000
open Contract

def parameters (i : Inputs) (scalar : ℝ → F) : Parameters F :=
  ⟨(shape i).K,(shape i).headDim,i.scalars.start,scalar i.scalars.epsilon,scalar i.scalars.theta⟩
def sourceArguments (i : Inputs) (phase : Fin 3) : TypedSource.Arguments :=
  let s := shape i
  ⟨"X","W"++toString phase.val,"RMS","OUT"++toString phase.val,
    s.M,s.N,s.K,s.M*s.K,s.K,1,1,s.K,1,s.M*s.N,s.N,1,
    16,64,64,s.headDim,i.scalars.start,phase.val<2,i.scalars.theta,i.scalars.epsilon⟩
def typedKernel (i : Inputs) : VeriTile.Triton.ComputeKernel :=
  TypedSource.qkv (decide (i.qw.metadata.dtype=int8))
    (sourceArguments i 0) (sourceArguments i 1) (sourceArguments i 2)

/-- Structural compiler boundary, pinned to the complete Python AST and typed
source. The physical masks/coverage/schedule are derived in CheckedGeometry;
no input-domain, output-value, numerical or whole-kernel law is assumed here. -/
structure Compiler (F H : Type) (i : Inputs) (a : Allocation i) (scalar : ℝ → F) where
  ast : String
  sourceIdentity : ast=productionAST
  source : VeriTile.Triton.ComputeKernel
  typedIdentity : source=typedKernel i
  program : Code F H Unit
  lowered : program=gridCode (parameters i scalar) (buffers i a.q)
    selectedPlans selectedReductions (schedule (shape i)).events
def Compiler.legacy (c : Compiler F H i a scalar) : CompilerBinding F H (shape i) (parameters i scalar) (buffers i a.q) where
  ast := c.ast
  sourceIdentity := c.sourceIdentity
  schedule := schedule (shape i)
  physicalLanes := lanes (shape i)
  dispatched := lanes_cells (shape i)
  program := c.program
  lowered := c.lowered

/-- Ordinary no-copy Tensor.view observations. These are metadata equations
only; the task proves which source stores populate this storage. -/
structure View (before after : Contract.Tensor) (newShape : List Nat) : Prop where
  shape : after.metadata.shape=newShape
  base : after.base=before.base
  bytes : after.bytes=before.bytes
  offset : after.offset=before.offset
  pointer : after.metadata.dataPtr=before.metadata.dataPtr
  dtype : after.metadata.dtype=before.metadata.dtype
  device : after.metadata.device=before.metadata.device
  contiguous : after.metadata.isContiguous=Bool.true
  resolved : after.metadata.isNeg=Bool.false
  layout : after.metadata.layout=.strided
  tensor : after.tensor=Bool.true
structure Returns (i : Inputs) (a : Allocation i) where
  tensor : Fin 3 → Contract.Tensor
  views : ∀ phase,View (output i a.q phase) (tensor phase)
    [(shape i).B,(shape i).M,(shape i).heads,(shape i).headDim]
def entry (abi : LocalEventABI F H σ ops) (c : Compiler F H i a scalar)
    (returns : Returns i a) (state : σ) : (Fin 3 → Contract.Tensor) × σ :=
  (returns.tensor,((hostCode (shape i) c.legacy).native abi state).2)
def tensorWord (memory : Memory F H) (t : Contract.Tensor) (index : Nat) : Stored F H :=
  memory (t.base,t.offset+index)

theorem return_address (returns : Returns i a) (x : Cell (shape i)) :
    ((returns.tensor x.1).base,(returns.tensor x.1).offset+
      flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2)=outputAddress (buffers i a.q) x := by
  rw [(returns.views x.1).base,(returns.views x.1).offset]
  exact (dense3_address _ _ _ _ _ _).symm

theorem original_target (reference : StorageOps F H) (i : Inputs) (h : InputContract i)
    (a : Allocation i) (scalar : ℝ → F) (before prepared : Memory F H)
    (allocationFrame : ∀ q,q.1≠a.q.base → prepared q=before q) (x : Cell (shape i)) :
    wordTarget reference (parameters i scalar) (buffers i a.q) selectedPlans selectedReductions prepared x=
    wordTarget reference (parameters i scalar) (buffers i a.q) selectedPlans selectedReductions before x := by
  have active : 0<(shape i).B ∧ 0<(shape i).M := ⟨by have := x.2.1.isLt; omega,by have := x.2.2.1.isLt; omega⟩
  have same : InputsAgree (buffers i a.q) prepared before := by
    intro q outside
    apply allocationFrame q
    exact outside 0
  unfold wordTarget
  rw [rowInput_current reference _ (buffers_separate i h a active) prepared before same]

theorem native_words (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (before prepared : σ)
    (allocationFrame : ∀ q,q.1≠a.q.base → abi.observe prepared q=abi.observe before q) :
    (∀ x : Cell (shape i),tensorWord (abi.observe (entry abi c returns prepared).2)
      ((entry abi c returns prepared).1 x.1) (flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2)=
      wordTarget reference (parameters i scalar) (buffers i a.q) selectedPlans selectedReductions (abi.observe before) x) ∧
    (∀ q,q.1≠a.q.base → (∀ x : Cell (shape i),q≠outputAddress (buffers i a.q) x) →
      abi.observe (entry abi c returns prepared).2 q=abi.observe before q) := by
  have frame (q) (outside : ∀ x : Cell (shape i),q≠outputAddress (buffers i a.q) x) :
      abi.observe (entry abi c returns prepared).2 q=abi.observe prepared q := by
    change abi.observe ((hostCode (shape i) c.legacy).native abi prepared).2 q=_
    rw [hostCode_eq,c.legacy.lowered,native_grid]
    exact run_frame actual _ _ _ _ _ _ q outside
  constructor
  · intro x
    have active : 0<(shape i).B ∧ 0<(shape i).M := ⟨by have := x.2.1.isLt; omega,by have := x.2.2.1.isLt; omega⟩
    have result := (bound_native_complete actual reference leaves abi (parameters i scalar) (buffers i a.q)
      (buffers_separate i h a active) (output_injective i h a) c.legacy prepared).1 x
    change abi.observe ((hostCode (shape i) c.legacy).native abi prepared).2
      ((returns.tensor x.1).base,(returns.tensor x.1).offset+_) = _
    rw [return_address returns x,hostCode_eq]
    exact result.trans (original_target reference i h a scalar _ _ allocationFrame x)
  · intro q fresh outside
    rw [frame q outside,allocationFrame q fresh]

theorem scalar_bytes (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (state : σ) (q : Address) (lane : Nat) (hlane : lane<raw.width q.1) :
    raw.bytes state (q.1,raw.width q.1*q.2+lane)=cellByte (abi.observe state q) lane := by
  rw [raw.decode]
  simp only [Nat.mul_add_div (raw.positive _),Nat.div_eq_of_lt hlane,Nat.add_zero,
    Nat.mul_add_mod,Nat.mod_eq_of_lt hlane,Prod.mk.eta]

/-- Scalar physical storage ABI. `bytesAt` is a byte-address observation,
not an output formula. The width is fixed by each Tensor's actual dtype. -/
structure PhysicalABI (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (a : Allocation i) (returns : Returns i a) extends RawByteABI abi where
  physicalByte : σ → Nat → Byte
  load32 : σ → Nat → BitVec 32
  load16 : σ → Nat → BitVec 16
  load8 : σ → Nat → Byte
  load32Bytes : ∀ state pointer lane,lane<4 →
    cellByte (.inr (.inl (load32 state pointer))) lane=physicalByte state (pointer+lane)
  load16Bytes : ∀ state pointer lane,lane<2 →
    cellByte (.inr (.inr (load16 state pointer))) lane=physicalByte state (pointer+lane)
  load8Byte : ∀ state pointer,load8 state pointer=physicalByte state pointer
  bytesAt : ∀ state t,t∈i.reads++[i.k,i.v,a.q] → ∀ byte,byte<t.bytes →
    toRawByteABI.bytes state (t.base,byte)=physicalByte state (t.base+byte)
  widthAt : ∀ t,t∈i.reads++[i.k,i.v,a.q] → 0<t.bytes → toRawByteABI.width t.base=Contract.width t.metadata.dtype
  input32 : ∀ state t,t∈i.reads → t.metadata.dtype=fp32 → ∀ j,j<t.metadata.numel →
    abi.observe state (t.base,t.offset+j)=.inr (.inl (load32 state (pointer64 t j).toNat))
  input16 : ∀ state t,t∈i.reads → t.metadata.dtype=fp16 → ∀ j,j<t.metadata.numel →
    abi.observe state (t.base,t.offset+j)=.inr (.inr (load16 state (pointer64 t j).toNat))
  input8 : ∀ state t,t∈i.reads → t.metadata.dtype=int8 → ∀ j,j<t.metadata.numel →
    abi.observe state (t.base,t.offset+j)=.inl (load8 state (pointer64 t j).toNat)

theorem native_return_bytes (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (physical : PhysicalABI abi i a returns) (before prepared : σ)
    (allocationFrame : ∀ q,q.1≠a.q.base → abi.observe prepared q=abi.observe before q)
    (x : Cell (shape i)) (lane : Nat) (hlane : lane<width (output i a.q x.1).metadata.dtype) :
    physical.physicalByte (entry abi c returns prepared).2
      ((pointer64 (returns.tensor x.1) (flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2)).toNat+lane)=
      cellByte (wordTarget reference (parameters i scalar) (buffers i a.q)
        selectedPlans selectedReductions (abi.observe before) x) lane := by
  let t := output i a.q x.1
  let index := flat3 (shape i).M (shape i).N x.2.1 x.2.2.1 x.2.2.2
  have ht : t∈i.reads++[i.k,i.v,a.q] := by
    dsimp [t]; generalize x.1=phase; fin_cases phase <;> simp [output]
  have hj : index<t.metadata.numel := by
    dsimp [t]; rw [output_count i h a]; exact flat3_lt _ _ _ _ _ _ x.2.1.isLt x.2.2.1.isLt x.2.2.2.isLt
  have hexact := (pointer_exact t (output_ordinary a x.1) index hj).1
  have hbyte : width t.metadata.dtype*(t.offset+index)+lane<t.bytes := by
    have := (output_ordinary a x.1).span
    have := width_positive t.metadata.dtype
    change lane<width t.metadata.dtype at hlane
    nlinarith
  have hw := physical.widthAt t ht (by omega)
  have hwords := (native_words actual reference leaves abi i h a scalar c returns before prepared allocationFrame).1 x
  change _=cellByte _ lane
  have hp : pointer64 (returns.tensor x.1) index=pointer64 t index := by
    simp only [pointer64,(returns.views x.1).pointer,(returns.views x.1).dtype,t]
  change physical.physicalByte _ ((pointer64 (returns.tensor x.1) index).toNat+lane)=_
  rw [hp,hexact,Nat.add_assoc,←physical.bytesAt _ t ht _ hbyte,←hw]
  rw [scalar_bytes abi physical.toRawByteABI _ (t.base,t.offset+index) lane (by simpa [hw] using hlane)]
  congr 1
  change abi.observe _ (t.base,t.offset+index)=_
  simpa only [tensorWord,entry,Prod.fst,(returns.views x.1).base,(returns.views x.1).offset] using hwords

theorem native_backing_frame (actual : StorageOps (BitVec 32) (BitVec 16))
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (physical : PhysicalABI abi i a returns) (before prepared : σ)
    (allocationFrame : ∀ t,t∈i.reads++[i.k,i.v] → ∀ byte,byte<t.bytes →
      physical.toRawByteABI.bytes prepared (t.base,byte)=physical.toRawByteABI.bytes before (t.base,byte))
    (t : Contract.Tensor) (ht : t∈i.reads++[i.k,i.v]) (byte : Nat) (hbyte : byte<t.bytes)
    (outside : ∀ x : Cell (shape i),(t.base,byte/physical.toRawByteABI.width t.base)≠outputAddress (buffers i a.q) x) :
    physical.physicalByte (entry abi c returns prepared).2 (t.base+byte)=physical.physicalByte before (t.base+byte) := by
  have ht' : t∈i.reads++[i.k,i.v,a.q] := by simp only [List.mem_append,List.mem_cons,List.mem_singleton] at ht ⊢; tauto
  rw [←physical.bytesAt _ t ht' byte hbyte,←physical.bytesAt before t ht' byte hbyte]
  change physical.toRawByteABI.bytes ((hostCode (shape i) c.legacy).native abi prepared).2 _=_
  rw [native_full_byte_frame actual abi physical.toRawByteABI c.legacy prepared t.base byte outside]
  exact allocationFrame t ht byte hbyte

theorem native_metadata (abi : LocalEventABI F H σ ops) (i : Inputs) (h : InputContract i)
    (a : Allocation i) (scalar : ℝ → F) (c : Compiler F H i a scalar) (returns : Returns i a) (state : σ) :
    (∀ phase,((entry abi c returns state).1 phase).metadata.shape=
      [(shape i).B,(shape i).M,(shape i).heads,(shape i).headDim]) ∧
    ((entry abi c returns state).1 0).metadata.dtype=fp16 ∧
    ((entry abi c returns state).1 1).metadata.dataPtr=i.k.metadata.dataPtr ∧
    ((entry abi c returns state).1 2).metadata.dataPtr=i.v.metadata.dataPtr ∧
    (∀ phase,((entry abi c returns state).1 phase).metadata.device=i.x.metadata.device) := by
  have f := facts i h
  refine ⟨fun phase => (returns.views phase).shape,?_,(returns.views 1).pointer,(returns.views 2).pointer,?_⟩
  · exact (returns.views 0).dtype.trans a.qDType
  · intro phase
    change (returns.tensor phase).metadata.device=_
    rw [(returns.views phase).device]
    fin_cases phase
    · exact a.qDevice
    · exact f.device i.k (by simp [Inputs.others])
    · exact f.device i.v (by simp [Inputs.others])

theorem returned_dtypes (i : Inputs) (h : InputContract i) (a : Allocation i) (returns : Returns i a) :
    ∀ phase,(returns.tensor phase).metadata.dtype=if phase.val=0 then fp16 else i.qw.metadata.dtype := by
  have f := facts i h
  intro phase
  rw [(returns.views phase).dtype]
  fin_cases phase
  · exact a.qDType
  · exact f.format i.k (by simp [Inputs.sameFormat])
  · exact f.format i.v (by simp [Inputs.sameFormat])

theorem input_outside (i : Inputs) (h : InputContract i) (a : Allocation i)
    (t : Contract.Tensor) (ht : t∈i.reads) (positive : 0<t.bytes) (offset : Nat) :
    ∀ x : Cell (shape i),(t.base,offset)≠outputAddress (buffers i a.q) x := by
  intro x he
  have f := facts i h
  have count : 0<(output i a.q x.1).metadata.numel := by
    rw [output_count i h a]
    exact Nat.mul_pos (Nat.mul_pos (by have := x.2.1.isLt; omega) (by have := x.2.2.1.isLt; omega))
      (by have := x.2.2.2.isLt; omega)
  have disj : disjoint t (output i a.q x.1) := by
    generalize hp : x.1=phase
    fin_cases phase
    · exact disjoint_symm _ _ (a.fresh t (by simp [ht]))
    · exact disjoint_symm _ _ (f.outputsDisjoint t ht).1
    · exact disjoint_symm _ _ (f.outputsDisjoint t ht).2
  exact disjoint_base_ne _ _ disj positive ((output_ordinary a x.1).positiveBytes count) (congrArg Prod.fst he)

theorem native_inputs_frame (actual : StorageOps (BitVec 32) (BitVec 16))
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (i : Inputs) (h : InputContract i) (a : Allocation i) (scalar : ℝ → BitVec 32)
    (c : Compiler (BitVec 32) (BitVec 16) i a scalar) (returns : Returns i a)
    (physical : PhysicalABI abi i a returns) (before prepared : σ)
    (allocationFrame : ∀ t,t∈i.reads++[i.k,i.v] → ∀ byte,byte<t.bytes →
      physical.toRawByteABI.bytes prepared (t.base,byte)=physical.toRawByteABI.bytes before (t.base,byte)) :
    ∀ t,t∈i.reads → ∀ byte,byte<t.bytes →
      physical.physicalByte (entry abi c returns prepared).2 (t.base+byte)=physical.physicalByte before (t.base+byte) := by
  intro t ht byte hb
  exact native_backing_frame actual abi i a scalar c returns physical before prepared allocationFrame t
    (by simp [ht]) byte hb (input_outside i h a t ht (by omega) _)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
