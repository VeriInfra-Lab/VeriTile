import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawCorrect
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawNative
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Memory

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.TensorBridge
open VeriTile Triton Reused Geometry Physical NativeInstructions
noncomputable section

/-- Independent Real specification: N query terms, the supplied normalizer,
grouped KV heads, and the source's one-sided complement-window convention. -/
def mathematical (c : Shape) (t : Tensors c) (read : Nat → ℝ) (offset size : Int)
    (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : ℝ :=
  RealPhysical.logicalTarget c (logicalInputs c t read) offset size
    (LogicalFormula.hostScale c.D) window complement z h j

/-- Kernel/grid result for each actual output Tensor element. The wrapper's
allocation supplies the prepared objects; it is not a host-success premise.
Nonempty launch is explicit because empty calls dispatch no kernel. -/
theorem kernel_correct (c : Shape) (t : Tensors c) (abi : ABI c) (read : Nat → ℝ)
    (offset size : Int) (window complement : Bool)
    (admitted : InputContract c t offset size window) (launch : NonemptyLaunch c)
    (binding : Binding c abi t) :
    Spec.Real (Whole002.Machine.Closed c abi offset size window ∧
    (∀ shape ∈ statementListShapes (kernel c offset size (LogicalFormula.hostScale c.D) window complement).surfaceBody, shape.prod ≤ 1048576) ∧
    ∃ result, realExecute c abi (physicalReal read) offset size
        (LogicalFormula.hostScale c.D) window complement = some result ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        logical (realMemory c t result read) t "out" (c.outputBase z.val h.val+j.val) =
          mathematical c t read offset size window complement z h j) ∧
      (∀ address, OutsideOutput c t address → realMemory c t result read address=read address)) := by
  have domain := Reused.Contract.domain c abi (observations t) offset size window admitted.1 binding
  obtain ⟨closed,tiles,result,he,hv,_,_⟩ := RawCorrect.kernel_correct c abi
    (physicalReal read) (logicalInputs c t read)
    (inputs_direct c t abi read offset size window admitted launch binding)
    offset size (LogicalFormula.hostScale c.D) window complement domain (admitted.2 launch)
  refine ⟨closed,tiles,result,he,?_,?_⟩
  · intro z h j
    rw [real_read c t abi read result offset size window admitted launch binding _ (output_bound c z h j)]
    simpa only [mathematical,RawCorrect.kernelMathematical,Nat.add_assoc] using hv z h j
  · exact real_frame c t result read

/-- Empty query input has the independently specified zero result; Python's
existing torch.zeros return supplies it without any tile or pointer checks. -/
theorem empty_query_target (c : Shape) (t : Tensors c) (read : Nat → ℝ) (offset size : Int)
    (window complement : Bool) (empty : c.N=0) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    mathematical c t read offset size window complement z h j = 0 := by
  unfold mathematical RealPhysical.logicalTarget
  apply Finset.sum_eq_zero
  intro i _
  have hi := i.isLt
  omega

/-- Independent input recurrence in fixed scalar/m16n8k8 nodes. It contains no
interpreter execution or chosen successful result. Scale is the actual launch
scalar; no equality with ideal reciprocal sqrt is assumed. -/
def nativeTarget (I : ISA (BitVec 32)) (c : Shape) (abi : ABI c) (heap : Bytes.Memory)
    (offset size : Int) (hostScale : ℝ) (window complement : Bool)
    (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : BitVec 32 :=
  (NativePhysical.target c abi (Bytes.words heap) offset size hostScale window complement z h j).eval (nativeISA I)

/-- Complete source execution, native instruction graph, actual FP32 Tensor
words and exterior byte frame. Primitive meanings and compiler-to-model
conformance are explicit model boundaries, not numerical experiment TODOs. -/
theorem kernel_native (I : ISA (BitVec 32)) (c : Shape) (t : Tensors c) (abi : ABI c)
    (heap : Bytes.Memory) (offset size : Int) (hostScale : ℝ) (window complement : Bool)
    (admitted : InputContract c t offset size window) (launch : NonemptyLaunch c)
    (binding : Binding c abi t) :
    Whole002.Machine.Closed c abi offset size window ∧
    (∀ shape ∈ statementListShapes (kernel c offset size hostScale window complement).surfaceBody, shape.prod ≤ 1048576) ∧
    ∃ result, nativeExecute c abi (Bytes.words heap) offset size hostScale window complement=some result ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        (Torch.FP32.logicalValue (fun a => ⟨Bytes.read (nativeMemory I c t abi heap result) a⟩)
          (t.get "out") (t.api "out" (by simp [tracked])) (c.outputBase z.val h.val+j.val)).bits =
          nativeTarget I c abi heap offset size hostScale window complement z h j) ∧
      (∀ address, OutsideOutput c t address → nativeMemory I c t abi heap result address=heap address) := by
  have domain := Reused.Contract.domain c abi (observations t) offset size window admitted.1 binding
  obtain ⟨closed,tiles,result,he,hv,_,_⟩ := RawNative.kernel_native I c abi
    (Bytes.words heap) offset size hostScale window complement domain (admitted.2 launch)
  refine ⟨closed,tiles,result,he,?_,?_⟩
  · intro z h j
    rw [native_read I c t abi heap result offset size window admitted _ (output_bound c z h j)]
    simpa only [nativeTarget,RawNative.kernelNativeTarget,nativeISA,Nat.add_assoc] using hv z h j
  · exact native_frame I c t abi heap result

#specNonCircular mathematical avoiding [realExecute,VeriTile.Triton.exec]
#specNonCircular nativeTarget avoiding [nativeExecute,FP.Structural.exec,NativePhysical.execute]
#axiomsClean kernel_correct
#axiomsClean empty_query_target
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.TensorBridge
