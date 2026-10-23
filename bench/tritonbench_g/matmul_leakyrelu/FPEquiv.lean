import bench.tritonbench_g.matmul_leakyrelu.Proofs.ContractedWhole

/-! Complete default FP32 host/grid result. The specification has independent
logical inputs and a scalar recurrence. Its dot is the source default dot,
followed by separate FP32 addition; neither exact Real summation nor a fused
dot-with-accumulator is substituted. Native numerical primitive conformance
is an explicit, separately registered deferred assumption.
-/
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000

/-- Logical values only: this specification has no memory, source registers,
program IDs, launch geometry, or source execution in its definition. -/
def logicalBlock {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) (block : Nat) : α :=
  Q.dot (fun k => if hk : block*32+k.val<r.inner then A i ⟨block*32+k.val,hk⟩ else Q.zero)
    (fun k => if hk : block*32+k.val<r.inner then B ⟨block*32+k.val,hk⟩ j else Q.zero)

def logicalReference {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) : α :=
  let value := foldRange (fun block x => Q.add x (logicalBlock Q r A B i j block))
    0 ((r.inner+31)/32) 1 Q.zero
  Q.castHalf (Q.activate (r.activation == "leaky_relu") value)

theorem reference_logical {α : Type} [Inhabited α]
    (Q : Numerics α) (r : Request) (initial : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (i : Fin r.rows) (j : Fin r.columns) :
    reference Q r initial i.val j.val=logicalReference Q r A B i j := by
  have hb (block : Nat) : blockValue Q r initial i.val j.val block=logicalBlock Q r A B i j block := by
    unfold blockValue logicalBlock
    apply congrArg₂ Q.dot
    · funext k
      by_cases hk : block*32+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hA i ⟨block*32+k.val,hk⟩
      · simp [hk]
    · funext k
      by_cases hk : block*32+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hB ⟨block*32+k.val,hk⟩ j
      · simp [hk]
  unfold reference accumulator logicalReference
  simp_rw [hb]

/-- K=0 executes no dot or load. Any required native zero/cast identity is a
numerical leaf fact; the complete structural statement retains these operations. -/
theorem logicalReference_zero_inner {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) (hk : r.inner=0) :
    logicalReference Q r A B i j=Q.castHalf (Q.activate (r.activation == "leaky_relu") Q.zero) := by
  simp [logicalReference,hk,foldRange]

/-- Whole original host, full grouped grid, both constexpr activation cases,
actual FP16 return, exact output frame, and source address/race safety.
Only leaf numerical behavior and ordinary tensor/API contracts are premises. -/
specification matmul_leakyrelu_fp32_host_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (tensor : TensorContract r initial) (indices : IndexDomain r) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (logicalReference Q r A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=initial.mem region offset) ∧
      SourceSafety r initial tensor := by
  obtain ⟨final,he,hv,hf⟩ := host_correct M D Q law r initial
    tensor.freshA tensor.freshB tensor.dimensions tensor.contiguousA
  refine ⟨final,he,?_,hf,source_safety r indices initial tensor⟩
  intro i j
  simpa only [reference_logical Q r initial A B hA hB i j] using hv i.val i.isLt j.val j.isLt

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole


namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.Contracted
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete selected FP32 compiled arithmetic profile: live-accumulator dot,
right-side slope, actual FP16 result, independent logical values, full frame,
and the unchanged proved finite source geometry. Primitive native numerical
conformance remains explicit. This is not source/fused floating equivalence. -/
specification matmul_leakyrelu_contracted_fp32_host_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPFused.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (tensor : TensorContract r initial) (indices : IndexDomain r) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (logicalReference Q r A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=initial.mem region offset) ∧
      SourceSafety r initial tensor := by
  obtain ⟨final,he,hv,hf⟩ := host_correct M D Q law r initial
    tensor.freshA tensor.freshB tensor.dimensions tensor.contiguousA
  refine ⟨final,he,?_,hf,source_safety r indices initial tensor⟩
  intro i j
  simpa only [reference_logical Q r initial A B hA hB i j] using hv i.val i.isLt j.val j.isLt

#specNonCircular logicalReference avoiding [host, runPrograms, contractedKernel, Whole.host, Whole.kernel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.Contracted
