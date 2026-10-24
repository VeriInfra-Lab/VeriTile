import bench.repairs.tritonbench_g.attention_score.Proofs.AcceptedFP
import bench.repairs.tritonbench_g.attention_score.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionScore.FPEquiv
open VeriTile Triton Geometry Physical Whole002 NativeInstructions
noncomputable section
abbrev reference := NativePhysical.target

/-- Exact native instruction graph from every current FP32 input word.
Default TF32 MMA, ordered reductions, exp2, casts and fusion are retained. -/
specification score_native (c : Shape) (dispatch : Dispatch c) (abi : ABI c)
    (allocation : Host.Allocation c abi) (physical : Nat → BitVec 32)
    (observed : Contract.Observations) (offset size : Int) (hostScale : ℝ)
    (window complement : Bool) (admitted : Contract.InputContract c observed offset size window)
    (pointers : Contract.PointerBinding c abi observed) :
    Spec.Real (Machine.Closed c abi offset size window ∧ ∃ result,
      Host.nativeReturn c dispatch abi allocation physical offset size hostScale window complement=some result ∧
      result.tensor=allocation.output ∧ result.tensor.shape=[c.Z,c.H,c.NK] ∧
      result.tensor.device=allocation.inputDevice ∧ result.tensor.fp32=Bool.true ∧
      result.blockM=c.B ∧ result.blockN=c.B ∧
      (∀ z h j,result.memory ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=
        reference c abi physical offset size hostScale window complement z h j) ∧
      (∀ a,¬(abi.alloc "out").Contains a → result.memory a=Graph.bits (physical a))) := by
  obtain ⟨closed,block,memory,he,hv,_,hf⟩ := Whole002.FPEquiv.native_return c dispatch abi physical
    offset size hostScale window complement (Contract.domain c abi observed offset size window admitted pointers)
  refine ⟨closed,⟨allocation.output,memory,persistentBlock dispatch,persistentBlock dispatch⟩,?_,
    rfl,allocation.shape,allocation.device,allocation.precision,block,block,hv,hf⟩
  simp [Host.nativeReturn,he]

/-- Conditional transport to another instruction interpretation uses only
scalar and fixed m16n8k8 coordinate laws; it asserts no exact-real GPU law. -/
theorem score_native_if_local {α β : Type} (f : α→β) (actual : ISA α) (standard : ISA β)
    (laws : LocalTransport f actual standard) (c : Shape) (dispatch : Dispatch c) (abi : ABI c)
    (allocation : Host.Allocation c abi) (physical : Nat → BitVec 32)
    (observed : Contract.Observations) (offset size : Int) (hostScale : ℝ)
    (window complement : Bool) (admitted : Contract.InputContract c observed offset size window)
    (pointers : Contract.PointerBinding c abi observed) :
    ∃ result,Host.nativeReturn c dispatch abi allocation physical offset size hostScale window complement=some result ∧
      result.tensor=allocation.output ∧ result.blockM=c.B ∧ result.blockN=c.B ∧
      (∀ z h j,f ((result.memory ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)).eval actual)=
        (reference c abi physical offset size hostScale window complement z h j).eval standard) ∧
      (∀ a,¬(abi.alloc "out").Contains a → result.memory a=Graph.bits (physical a)) := by
  obtain ⟨_,result,he,ht,_,_,_,hm,hn,hv,hf⟩ := score_native c dispatch abi allocation physical observed offset size hostScale window complement admitted pointers
  refine ⟨result,he,ht,hm,hn,?_,hf⟩
  intro z h j
  rw [hv]
  exact transport f actual standard laws _

#specNonCircular reference avoiding [Host.nativeReturn,FP.Structural.exec]
#axiomsClean score_native
#axiomsClean score_native_if_local
end
end VeriTile.Bench.Repairs.AttentionScore.FPEquiv
