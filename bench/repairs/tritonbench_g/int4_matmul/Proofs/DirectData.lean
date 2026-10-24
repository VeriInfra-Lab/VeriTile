import bench.repairs.tritonbench_g.int4_matmul.Proofs.Admission
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Real

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host HelperHost Physical
noncomputable section
set_option maxHeartbeats 2000000

abbrev Heap (α : Type) := Nat → Nat → α
structure HeapState (α : Type) where
  values : Heap α
  packed : Heap Word

def logicalIndex (t : TensorInfo) (r c : Nat) := t.descriptor.offset+t.descriptor.elementOffset r c
/-- Logical read through the original view on valid coordinates. Values outside
its domain only totalize the reader and are never reached by an active load. -/
def logicalBuffer (t : TensorInfo) (m : Heap α) (i : Nat) : α :=
  if i<t.descriptor.rows*t.descriptor.columns then
    m t.descriptor.allocation (logicalIndex t (i/t.descriptor.columns) (i%t.descriptor.columns))
  else m t.descriptor.allocation i

theorem logicalBuffer_eq (t : TensorInfo) (d : DType) (layout : t.descriptor.PackedLayout d t.descriptor.rows t.descriptor.columns t.descriptor.device)
    (m : Heap α) (i : Nat) : logicalBuffer t m i=m t.descriptor.allocation i := by
  by_cases hi : i<t.descriptor.rows*t.descriptor.columns
  · have cp : 0<t.descriptor.columns:=by nlinarith
    have hr : i/t.descriptor.columns<t.descriptor.rows:=(Nat.div_lt_iff_lt_mul cp).mpr hi
    have hc : i%t.descriptor.columns<t.descriptor.columns:=Nat.mod_lt _ cp
    have stride:=t.descriptor.packed_active_offset d _ _ _ _ _ layout hr hc
    have offset : t.descriptor.offset=0:=by rcases layout with h|h <;> exact h.2.2.2.2.2.2.1
    simp only [logicalBuffer,hi,ite_true,logicalIndex,offset,Nat.zero_add,stride]
    congr 1
    nlinarith [Nat.mod_add_div i t.descriptor.columns]
  · simp [logicalBuffer,hi]

structure TensorLaws (base : Nat → Nat) (t : TensorInfo) : Prop where
  pointer : t.dataPtr=BitVec.ofNat 64 (base t.descriptor.allocation+elementBytes t.descriptor.dtype*t.descriptor.offset)
  capacity : base t.descriptor.allocation+elementBytes t.descriptor.dtype*
    (t.descriptor.offset+t.descriptor.rows*t.descriptor.columns)≤2^64

def PointerBinding (base : Nat → Nat) (t : TensorInfo) : Prop :=
  ∀ i,i<t.descriptor.rows*t.descriptor.columns →
    (t.dataPtr+BitVec.ofNat 64 (elementBytes t.descriptor.dtype*i)).toNat=
      base t.descriptor.allocation+elementBytes t.descriptor.dtype*(t.descriptor.offset+i)
theorem pointer_binding (base : Nat → Nat) (t : TensorInfo) (laws : TensorLaws base t) : PointerBinding base t := by
  intro i hi
  have positive : 0<elementBytes t.descriptor.dtype:=by unfold elementBytes;split_ifs <;> decide
  have lt:=Nat.mul_lt_mul_of_pos_left (show t.descriptor.offset+i<t.descriptor.offset+t.descriptor.rows*t.descriptor.columns by omega) positive
  have cap:=laws.capacity
  simp only [laws.pointer,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  rw [show base t.descriptor.allocation+elementBytes t.descriptor.dtype*t.descriptor.offset+elementBytes t.descriptor.dtype*i=
    base t.descriptor.allocation+elementBytes t.descriptor.dtype*(t.descriptor.offset+i) by ring]
  apply Nat.mod_eq_of_lt
  omega

structure MatmulRequest where
  input : TensorInfo
  weight : TensorInfo
  scales : TensorInfo
  zeros : TensorInfo
  group : Int
  output : Option TensorInfo
  contract : MatmulContract input weight scales zeros group output

def MatmulRequest.shape (r : MatmulRequest) := matrixShape r.input r.weight r.scales
def MatmulRequest.arguments (r : MatmulRequest) := Direct.arguments r.input r.weight r.scales r.zeros r.output
def MatmulRequest.views (r : MatmulRequest) := [r.input,r.weight,r.scales,r.zeros]
def MatmulRequest.admitted (r : MatmulRequest) : Host.Contract r.arguments := matmul_admitted _ _ _ _ _ _ r.contract

def matrixState (r : MatmulRequest) (out : TensorInfo) (m : HeapState α) : State α :=
  ⟨(fun region=>match region with
    | .a=>m.values r.input.descriptor.allocation
    | .scale=>m.values r.scales.descriptor.allocation
    | .output=>m.values out.descriptor.allocation
    | .weight=>m.values r.weight.descriptor.allocation
    | .unpacked=>m.values r.input.descriptor.allocation
    | .other _=>m.values r.input.descriptor.allocation),
   (fun region=>match region with
    | .weight=>m.packed r.weight.descriptor.allocation
    | .zero=>m.packed r.zeros.descriptor.allocation)⟩
def originalMatrixState (r : MatmulRequest) (out : TensorInfo) (m : HeapState α) : State α :=
  ⟨(fun region=>match region with
    | .a=>logicalBuffer r.input m.values
    | .scale=>logicalBuffer r.scales m.values
    | .output=>m.values out.descriptor.allocation
    | .weight=>m.values r.weight.descriptor.allocation
    | .unpacked=>m.values r.input.descriptor.allocation
    | .other _=>m.values r.input.descriptor.allocation),
   (fun region=>match region with
    | .weight=>logicalBuffer r.weight m.packed
    | .zero=>logicalBuffer r.zeros m.packed)⟩
theorem matrixState_original (r : MatmulRequest) (out : TensorInfo) (m : HeapState α) : matrixState r out m=originalMatrixState r out m := by
  have a:=canonical_tensor r.input .fp32 r.contract.input
  have s:=packed_tensor r.scales .fp32 r.contract.packed.scales
  have w:=packed_tensor r.weight .int32 r.contract.packed.weight
  have z:=packed_tensor r.zeros .int32 r.contract.packed.zeros
  change State.mk _ _=State.mk _ _
  rw [State.mk.injEq]
  constructor
  · funext region i
    cases region <;> simp only [matrixState,originalMatrixState,logicalBuffer_eq _ _ (Or.inl a),logicalBuffer_eq _ _ s]
  · funext region i
    cases region <;> simp only [matrixState,originalMatrixState,logicalBuffer_eq _ _ w,logicalBuffer_eq _ _ z]

def realTarget (r : MatmulRequest) (out : TensorInfo) (caller : HeapState ℝ) (row col : Nat) : ℝ :=
  mathematical r.shape (originalMatrixState r out caller) row col

def returned (r : MatmulRequest) (fresh : TensorInfo) : TensorInfo := r.output.getD fresh
structure OutputAllocation (r : MatmulRequest) (caller : HeapState α) (live : Nat → Prop) where
  tensor : TensorInfo
  memory : HeapState α
  identity : ∀ supplied,r.output=some supplied → tensor=supplied
  descriptor : TensorContract tensor .fp32 Bool.false
  shape : tensor.descriptor.rows=r.shape.m ∧ tensor.descriptor.columns=r.shape.n
  device : tensor.descriptor.device=r.input.descriptor.device
  fresh : r.output=none → ¬live tensor.descriptor.allocation
  frameValues : ∀ storage i,storage≠tensor.descriptor.allocation ∨ r.output.isSome=Bool.true → memory.values storage i=caller.values storage i
  framePacked : memory.packed=caller.packed

theorem OutputAllocation.separate (r : MatmulRequest) (caller : HeapState α) (live : Nat → Prop)
    (inputs : ∀ t∈r.views,live t.descriptor.allocation) (alloc : OutputAllocation r caller live) :
    ∀ t∈r.views,t.descriptor.allocation≠alloc.tensor.descriptor.allocation := by
  intro t ht
  cases eq : r.output with
  | none=>intro he;have hl:=inputs t ht;rw [he] at hl;exact alloc.fresh eq hl
  | some supplied=>
    rw [alloc.identity supplied eq]
    have hn:=r.contract.independent
    simp only [Independent,matrixInputs,eq,Option.toList_some,List.map_append,List.nodup_append] at hn
    have disjoint:=hn.2.2
    have mem : t.descriptor.allocation∈[r.input,r.weight,r.scales,r.zeros].map (fun x=>x.descriptor.allocation):=
      List.mem_map.mpr ⟨t,ht,rfl⟩
    exact disjoint _ mem _ (by simp)

theorem allocated_inputs (r : MatmulRequest) (caller : HeapState α) (live : Nat → Prop)
    (inputs : ∀ t∈r.views,live t.descriptor.allocation) (alloc : OutputAllocation r caller live) :
    Readonly (matrixState r alloc.tensor caller) (matrixState r alloc.tensor alloc.memory) := by
  constructor
  · intro region i ne
    cases region with
    | a=>exact alloc.frameValues _ _ (Or.inl (alloc.separate r caller live inputs _ (by simp [MatmulRequest.views])))
    | scale=>exact alloc.frameValues _ _ (Or.inl (alloc.separate r caller live inputs _ (by simp [MatmulRequest.views])))
    | output=>exact False.elim (ne rfl)
    | weight=>exact alloc.frameValues _ _ (Or.inl (alloc.separate r caller live inputs _ (by simp [MatmulRequest.views])))
    | unpacked=>exact alloc.frameValues _ _ (Or.inl (alloc.separate r caller live inputs _ (by simp [MatmulRequest.views])))
    | other n=>exact alloc.frameValues _ _ (Or.inl (alloc.separate r caller live inputs _ (by simp [MatmulRequest.views])))
  · simp only [matrixState,alloc.framePacked]

#axiomsClean logicalBuffer_eq
#axiomsClean pointer_binding
#axiomsClean matrixState_original
#specNonCircular realTarget avoiding [call,execute,partialValue,referencePartial]
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
