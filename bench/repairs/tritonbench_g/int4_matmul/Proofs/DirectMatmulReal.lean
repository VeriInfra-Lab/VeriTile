import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectRun

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host
noncomputable section
set_option maxHeartbeats 2000000

structure RealResult where
  tensor : TensorInfo
  memory : HeapState ℝ
  currentDevice : Nat

def mergeState (output : TensorInfo) (initial : HeapState α) (result : State α) : HeapState α :=
  ⟨(fun storage i=>if storage=output.descriptor.allocation then result.values .output i else initial.values storage i),initial.packed⟩
def runMatmulReal (r : MatmulRequest) (callerDevice : Nat) (trials : List (Invocation r.shape)) (chosen : Invocation r.shape)
    (caller : HeapState ℝ) (live : Nat → Prop) (alloc : OutputAllocation r caller live) : Except InputError RealResult := do
  checkMatmul r.input r.weight r.scales r.zeros r.group r.output
  let result:=wrapperRun r.shape realOps trials chosen (matrixState r alloc.tensor alloc.memory)
  return ⟨alloc.tensor,mergeState alloc.tensor alloc.memory result,callerDevice⟩

def OutputDescriptor (r : MatmulRequest) (out : TensorInfo) : Prop :=
  TensorContract out .fp32 Bool.false ∧ out.descriptor.rows=r.shape.m ∧ out.descriptor.columns=r.shape.n ∧
    out.descriptor.device=r.input.descriptor.device ∧ (∀ supplied,r.output=some supplied → out=supplied)
theorem OutputAllocation.geometry (alloc : OutputAllocation r caller live) : OutputDescriptor r alloc.tensor :=
  ⟨alloc.descriptor,alloc.shape.1,alloc.shape.2,alloc.device,alloc.identity⟩
theorem output_index (r : MatmulRequest) (out : TensorInfo) (h : OutputDescriptor r out) (row col : Nat) :
    logicalIndex out row col=cAddress r.shape row col := by
  simp only [logicalIndex,Tensor.elementOffset,h.1.offset,h.1.strides.1,h.1.strides.2,h.2.2.1,Nat.zero_add,Nat.mul_one,cAddress]

def OutputRegion (r : MatmulRequest) (out : TensorInfo) (storage i : Nat) := storage=out.descriptor.allocation ∧ i<r.shape.m*r.shape.n

theorem matmul_real_complete (r : MatmulRequest) (callerDevice : Nat)
    (trials : List (Invocation r.shape)) (chosen : Invocation r.shape) (caller : HeapState ℝ) (live : Nat → Prop)
    (inputs : ∀ t∈r.views,live t.descriptor.allocation) (alloc : OutputAllocation r caller live) :
    ∃ result,runMatmulReal r callerDevice trials chosen caller live alloc=.ok result ∧
      result.tensor=alloc.tensor ∧ OutputDescriptor r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ row col,row<r.shape.m → col<r.shape.n → result.memory.values result.tensor.descriptor.allocation
        (logicalIndex result.tensor row col)=realTarget r alloc.tensor caller row col) ∧
      (∀ storage i,live storage → ¬OutputRegion r result.tensor storage i → result.memory.values storage i=caller.values storage i) ∧
      result.memory.packed=caller.packed ∧
      (∀ t∈sourceTiles,IndexDomain r.shape t ∧ StrideBinding r.arguments t) := by
  let state:=wrapperRun r.shape realOps trials chosen (matrixState r alloc.tensor alloc.memory)
  refine ⟨⟨alloc.tensor,mergeState alloc.tensor alloc.memory state,callerDevice⟩,?_,rfl,alloc.geometry,rfl,?_,?_,alloc.framePacked,?_⟩
  · simp [runMatmulReal,(checkMatmul_iff _ _ _ _ _ _).mpr r.contract,Except.bind,Except.pure,bind,pure,state]
  · intro row col hr hc
    change (mergeState alloc.tensor alloc.memory state).values alloc.tensor.descriptor.allocation (logicalIndex alloc.tensor row col)=_
    simp only [mergeState,ite_true,output_index r alloc.tensor alloc.geometry]
    have h:=wrapper_real r.shape r.admitted.shape trials chosen (matrixState r alloc.tensor caller)
      (matrixState r alloc.tensor alloc.memory) (allocated_inputs r caller live inputs alloc) row col hr hc
    exact h.trans (congrArg (fun m=>mathematical r.shape m row col) (matrixState_original r alloc.tensor caller))
  · intro storage i owned outside
    change (mergeState alloc.tensor alloc.memory state).values storage i=caller.values storage i
    by_cases eq : storage=alloc.tensor.descriptor.allocation
    · have outsideIndex : r.shape.m*r.shape.n≤ i:=by
        change ¬(storage=alloc.tensor.descriptor.allocation ∧ i<r.shape.m*r.shape.n) at outside
        omega
      simp only [mergeState,eq,ite_true]
      rw [wrapper_frame r.shape r.admitted.shape realOps trials chosen _ .output i (Or.inr outsideIndex)]
      change alloc.memory.values alloc.tensor.descriptor.allocation i=caller.values (alloc.tensor.descriptor.allocation) i
      cases out : r.output with
      | none=>rw [eq] at owned;exact False.elim (alloc.fresh out owned)
      | some supplied=>exact alloc.frameValues _ _ (Or.inr (by simp [out]))
    · simp only [mergeState,eq,ite_false]
      exact alloc.frameValues _ _ (Or.inl eq)
  · intro tile member
    exact ⟨guarded_indices r.shape r.admitted.indices tile member,contract_strides r.arguments r.admitted tile member⟩

#axiomsClean matmul_real_complete
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
