import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectMatmulReal

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host Physical
noncomputable section
set_option maxHeartbeats 2500000

abbrev ByteHeap := Heap Byte
def wordHeap (m : ByteHeap) : Heap Word := fun storage index=>decode (fun i=>m storage (4*index+i.val))
def heapWords (m : ByteHeap) : HeapState Word := ⟨wordHeap m,wordHeap m⟩
def matrixBytes (r : MatmulRequest) (out : TensorInfo) (m : ByteHeap) : Bytes :=
  fun region=>match region with
  | .value .a=>m r.input.descriptor.allocation
  | .value .scale=>m r.scales.descriptor.allocation
  | .value .output=>m out.descriptor.allocation
  | .value .weight=>m r.weight.descriptor.allocation
  | .value .unpacked=>m r.input.descriptor.allocation
  | .value (.other _)=>m r.input.descriptor.allocation
  | .packed .weight=>m r.weight.descriptor.allocation
  | .packed .zero=>m r.zeros.descriptor.allocation
theorem matrixBytes_load (r : MatmulRequest) (out : TensorInfo) (m : ByteHeap) :
    load (matrixBytes r out m)=matrixState r out (heapWords m) := by
  change State.mk _ _=State.mk _ _
  rw [State.mk.injEq]
  constructor
  · funext region i;cases region <;> rfl
  · funext region i;cases region <;> rfl

structure NativeOutputAllocation (r : MatmulRequest) (caller : ByteHeap) (live : Nat → Prop) where
  tensor : TensorInfo
  memory : ByteHeap
  descriptor : OutputDescriptor r tensor
  fresh : r.output=none → ¬live tensor.descriptor.allocation
  frame : ∀ storage i,storage≠tensor.descriptor.allocation ∨ r.output.isSome=Bool.true → memory storage i=caller storage i

theorem NativeOutputAllocation.separate (r : MatmulRequest) (caller : ByteHeap) (live : Nat → Prop)
    (inputs : ∀ t∈r.views,live t.descriptor.allocation) (alloc : NativeOutputAllocation r caller live) :
    ∀ t∈r.views,t.descriptor.allocation≠alloc.tensor.descriptor.allocation := by
  intro t ht
  cases eq : r.output with
  | none=>intro he;have hl:=inputs t ht;rw [he] at hl;exact alloc.fresh eq hl
  | some supplied=>
    rw [alloc.descriptor.2.2.2.2 supplied eq]
    have hn:=r.contract.independent
    simp only [Independent,matrixInputs,eq,Option.toList_some,List.map_append,List.nodup_append] at hn
    exact hn.2.2 _ (List.mem_map.mpr ⟨t,ht,rfl⟩) _ (by simp)
theorem native_allocated_inputs (r : MatmulRequest) (caller : ByteHeap) (live : Nat → Prop)
    (inputs : ∀ t∈r.views,live t.descriptor.allocation) (alloc : NativeOutputAllocation r caller live) :
    Readonly (load (matrixBytes r alloc.tensor caller)) (load (matrixBytes r alloc.tensor alloc.memory)) := by
  have same (t : TensorInfo) (ht : t∈r.views) : alloc.memory t.descriptor.allocation=caller t.descriptor.allocation :=
    funext (fun i=>alloc.frame _ i (Or.inl (alloc.separate r caller live inputs t ht)))
  constructor
  · intro region i ne
    cases region with
    | output=>exact False.elim (ne rfl)
    | a|unpacked|other _=>simp only [load,matrixBytes,same r.input (by simp [MatmulRequest.views])]
    | scale=>simp only [load,matrixBytes,same r.scales (by simp [MatmulRequest.views])]
    | weight=>simp only [load,matrixBytes,same r.weight (by simp [MatmulRequest.views])]
  · funext region i
    cases region <;> simp only [load,matrixBytes,same r.weight (by simp [MatmulRequest.views]),same r.zeros (by simp [MatmulRequest.views])]

structure NativeResult where
  tensor : TensorInfo
  memory : ByteHeap
  currentDevice : Nat

def mergeBytes (out : TensorInfo) (initial : ByteHeap) (result : Bytes) : ByteHeap :=
  fun storage i=>if storage=out.descriptor.allocation then result (.value .output) i else initial storage i
def runMatmulNative (r : MatmulRequest) (p : Native.Instructions) (callerDevice : Nat)
    (trials : List (NativeInvocation r.shape)) (chosen : NativeInvocation r.shape)
    (caller : ByteHeap) (live : Nat → Prop) (alloc : NativeOutputAllocation r caller live) : Except InputError NativeResult := do
  checkMatmul r.input r.weight r.scales r.zeros r.group r.output
  let result:=nativeWrapper r.shape p trials chosen (matrixBytes r alloc.tensor alloc.memory)
  return ⟨alloc.tensor,mergeBytes alloc.tensor alloc.memory result,callerDevice⟩

def wordTarget (r : MatmulRequest) (out : TensorInfo) (p : Native.Instructions) (chosen : NativeInvocation r.shape)
    (caller : ByteHeap) (row col : Nat) : Word :=
  referenceOutput r.shape chosen.tile (Native.ops p chosen.isa) (originalMatrixState r out (heapWords caller)) row col chosen.schedule 0

theorem matmul_native_if_local (base : Nat → Nat) (r : MatmulRequest)
    (inputLaws : ∀ t∈r.views,TensorLaws base t) (p reference : Native.Instructions) (numeric : Native.Correspondence p reference)
    (callerDevice : Nat) (trials : List (NativeInvocation r.shape)) (chosen : NativeInvocation r.shape)
    (caller : ByteHeap) (live : Nat → Prop) (inputs : ∀ t∈r.views,live t.descriptor.allocation)
    (alloc : NativeOutputAllocation r caller live) (outputLaws : TensorLaws base alloc.tensor) :
    ∃ result,runMatmulNative r p callerDevice trials chosen caller live alloc=.ok result ∧
      result.tensor=alloc.tensor ∧ OutputDescriptor r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ t∈r.views,PointerBinding base t) ∧ PointerBinding base result.tensor ∧
      chosen.parsedProduction=SourceBinding.productionAST ∧
      (∀ row col,row<r.shape.m → col<r.shape.n → ∀ i : Fin 4,
        result.memory result.tensor.descriptor.allocation (4*logicalIndex result.tensor row col+i.val)=
          encode (wordTarget r alloc.tensor reference chosen caller row col) i) ∧
      (∀ storage i,live storage → ¬OutputRegion r result.tensor storage (i/4) → result.memory storage i=caller storage i) ∧
      (∀ t∈sourceTiles,IndexDomain r.shape t ∧ StrideBinding r.arguments t) := by
  let bytes:=nativeWrapper r.shape p trials chosen (matrixBytes r alloc.tensor alloc.memory)
  refine ⟨⟨alloc.tensor,mergeBytes alloc.tensor alloc.memory bytes,callerDevice⟩,?_,rfl,alloc.descriptor,rfl,
    (fun t ht=>pointer_binding base t (inputLaws t ht)),pointer_binding base alloc.tensor outputLaws,chosen.sourceIdentity,?_,?_,?_⟩
  · simp [runMatmulNative,(checkMatmul_iff _ _ _ _ _ _).mpr r.contract,Except.bind,Except.pure,bind,pure,bytes]
  · intro row col hr hc i
    change mergeBytes alloc.tensor alloc.memory bytes alloc.tensor.descriptor.allocation (4*logicalIndex alloc.tensor row col+i.val)=_
    simp only [mergeBytes,ite_true,output_index r alloc.tensor alloc.descriptor]
    dsimp only [bytes]
    rw [native_result r.shape r.admitted.shape p trials chosen _ row col hr hc i]
    rw [referenceOutput_current _ _ _ _ _ (native_allocated_inputs r caller live inputs alloc),matrixBytes_load,matrixState_original]
    rw [Native.correspondence_eq p reference numeric]
    rfl
  · intro storage i owned outside
    change mergeBytes alloc.tensor alloc.memory bytes storage i=caller storage i
    by_cases eq : storage=alloc.tensor.descriptor.allocation
    · have bound : r.shape.m*r.shape.n≤ i/4:=by
        change ¬(storage=alloc.tensor.descriptor.allocation ∧ i/4<r.shape.m*r.shape.n) at outside
        omega
      simp only [mergeBytes,eq,ite_true]
      dsimp only [bytes]
      rw [native_frame r.shape r.admitted.shape p trials chosen _ (.value .output) i (Or.inr bound)]
      change alloc.memory alloc.tensor.descriptor.allocation i=caller alloc.tensor.descriptor.allocation i
      cases out : r.output with
      | none=>rw [eq] at owned;exact False.elim (alloc.fresh out owned)
      | some supplied=>exact alloc.frame _ _ (Or.inr (by simp [out]))
    · simp only [mergeBytes,eq,ite_false]
      exact alloc.frame _ _ (Or.inl eq)
  · intro tile member
    exact ⟨guarded_indices r.shape r.admitted.indices tile member,contract_strides r.arguments r.admitted tile member⟩

#axiomsClean matrixBytes_load
#axiomsClean native_allocated_inputs
#axiomsClean matmul_native_if_local
#specNonCircular wordTarget avoiding [runMatmulNative,nativeWrapper,byteCall,partialValue]
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
