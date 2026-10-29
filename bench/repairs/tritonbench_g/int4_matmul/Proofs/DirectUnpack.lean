import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectHelpersData

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host Helpers HelperHost HelperBytes Physical
noncomputable section
set_option maxHeartbeats 2000000

structure UnpackSource where
  parsedProduction : String
  sourceIdentity : parsedProduction=SourceBinding.productionAST

def unpackRealTarget (r : UnpackRequest) (caller : HeapState ℝ) (k n : Nat) : ℝ :=
  ((logicalQ r.shape (originalUnpackState r caller) k n-logicalZ r.shape (originalUnpackState r caller) k n:Int):ℝ)*
    logicalScale r.shape (originalUnpackState r caller) k n

def unpackRealMemory (r : UnpackRequest) (caller : HeapState ℝ) (alloc : UnpackAllocation r live) : HeapState ℝ :=
  ⟨(fun storage i=>if storage=alloc.tensor.descriptor.allocation ∧ i<r.shape.n*r.shape.k then
    unpackLoop r.shape HelperReal.halfReal (unpackState r caller) (i%r.shape.k) (i/r.shape.k) (r.shape.k/r.shape.group)
    else caller.values storage i),caller.packed⟩
def runUnpackReal (r : UnpackRequest) (callerDevice : Nat) (caller : HeapState ℝ) (live : Nat → Prop)
    (alloc : UnpackAllocation r live) : Except InputError RealResult := do
  checkPacked r.weight r.scales r.zeros
  return ⟨alloc.tensor,unpackRealMemory r caller alloc,callerDevice⟩

theorem unpack_real_complete (r : UnpackRequest) (callerDevice : Nat) (caller : HeapState ℝ) (live : Nat → Prop)
    (inputs : ∀ t∈r.views,live t.descriptor.allocation) (alloc : UnpackAllocation r live) :
    ∃ result,runUnpackReal r callerDevice caller live alloc=.ok result ∧
      result.tensor=alloc.tensor ∧ UnpackTensor r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ k n,k<r.shape.k → n<r.shape.n → result.memory.values result.tensor.descriptor.allocation
        (logicalIndex result.tensor k n)=unpackRealTarget r caller k n) ∧
      (∀ storage i,live storage → result.memory.values storage i=caller.values storage i) ∧ result.memory.packed=caller.packed := by
  refine ⟨⟨alloc.tensor,unpackRealMemory r caller alloc,callerDevice⟩,?_,rfl,alloc.descriptor,rfl,?_,?_,rfl⟩
  · simp [runUnpackReal,(checkPacked_iff _ _ _).mpr r.contract,Except.bind,Except.pure,bind,pure]
  · intro k n hk hn
    have bound:=rowMajor_bound r.shape.n r.shape.k n k hn hk
    change (unpackRealMemory r caller alloc).values alloc.tensor.descriptor.allocation (logicalIndex alloc.tensor k n)=_
    simp only [unpackTensor_index _ _ alloc.descriptor,unpackRealMemory,bound,true_and,ite_true,
      div_coordinate n r.shape.k k r.valid.kpos hk,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hk]
    rw [HelperReal.unpack_independent r.shape r.valid (unpackState r caller) k n hk,unpackState_original]
    rfl
  · intro storage i owned
    have ne : storage≠alloc.tensor.descriptor.allocation:=by intro eq;rw [eq] at owned;exact alloc.fresh owned
    simp only [unpackRealMemory,ne,false_and,ite_false]

def unpackWordTarget (r : UnpackRequest) (p : HelperNative.Instructions) (caller : ByteHeap) (k n : Nat) : Half :=
  unpackSpec r.shape (HelperNative.halfArithmetic p) (originalUnpackState r (heapWords caller)) k n

def unpackNativeMemory (r : UnpackRequest) (p : HelperNative.Instructions) (caller : ByteHeap) (alloc : UnpackAllocation r live) : ByteHeap :=
  let result:=unpack r.shape (HelperNative.halfArithmetic p) (unpackBytesInput r caller) (caller alloc.tensor.descriptor.allocation)
  fun storage i=>if storage=alloc.tensor.descriptor.allocation then result.payload i else caller storage i

def runUnpackNative (r : UnpackRequest) (p : HelperNative.Instructions) (call : UnpackSource) (callerDevice : Nat)
    (caller : ByteHeap) (live : Nat → Prop) (alloc : UnpackAllocation r live) : Except InputError NativeResult := do
  checkPacked r.weight r.scales r.zeros
  return ⟨alloc.tensor,unpackNativeMemory r p caller alloc,callerDevice⟩

theorem unpack_native_if_local (base : Nat → Nat) (r : UnpackRequest) (inputLaws : ∀ t∈r.views,TensorLaws base t)
    (p reference : HelperNative.Instructions) (numeric : HelperNative.Correspondence p reference)
    (call : UnpackSource) (callerDevice : Nat) (caller : ByteHeap) (live : Nat → Prop)
    (inputs : ∀ t∈r.views,live t.descriptor.allocation) (alloc : UnpackAllocation r live) (outputLaws : TensorLaws base alloc.tensor) :
    ∃ result,runUnpackNative r p call callerDevice caller live alloc=.ok result ∧
      result.tensor=alloc.tensor ∧ UnpackTensor r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ t∈r.views,PointerBinding base t) ∧ PointerBinding base result.tensor ∧
      (∀ k n,k<r.shape.k → n<r.shape.n → ∀ i : Fin 2,
        result.memory result.tensor.descriptor.allocation (2*logicalIndex result.tensor k n+i.val)=
          encode16 (unpackWordTarget r reference caller k n) i) ∧
      (∀ storage i,live storage → result.memory storage i=caller storage i) ∧ call.parsedProduction=SourceBinding.productionAST := by
  refine ⟨⟨alloc.tensor,unpackNativeMemory r p caller alloc,callerDevice⟩,?_,rfl,alloc.descriptor,rfl,
    (fun t ht=>pointer_binding base t (inputLaws t ht)),pointer_binding base alloc.tensor outputLaws,?_,?_,call.sourceIdentity⟩
  · simp [runUnpackNative,(checkPacked_iff _ _ _).mpr r.contract,Except.bind,Except.pure,bind,pure]
  · intro k n hk hn i
    change unpackNativeMemory r p caller alloc alloc.tensor.descriptor.allocation (2*logicalIndex alloc.tensor k n+i.val)=_
    simp only [unpackNativeMemory,ite_true,unpackTensor_index _ _ alloc.descriptor]
    rw [unpack_bytes r.shape r.valid _ _ _ k n hk hn i,unpackBytesInput_load,unpackState_original,HelperNative.correspondence_eq p reference numeric]
    rfl
  · intro storage i owned
    have ne : storage≠alloc.tensor.descriptor.allocation:=by intro eq;rw [eq] at owned;exact alloc.fresh owned
    simp only [unpackNativeMemory,ne,ite_false]

#axiomsClean unpack_real_complete
#axiomsClean unpack_native_if_local
#specNonCircular unpackRealTarget avoiding [runUnpackReal,unpackRealMemory,unpackLoop]
#specNonCircular unpackWordTarget avoiding [runUnpackNative,unpackNativeMemory,unpackLoop]
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
