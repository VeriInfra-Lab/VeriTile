import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IOBinding
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 40000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The target reads only the mathematical K-domain, including in padded K8
blocks. This fact does not restrict FP values or assume a numerical identity. -/
theorem native_congr (R : Instructions.Reference) (K BK : Nat)
    (A B A' B' : Nat → Nat → Word) (i j : Nat)
    (ha : ∀ k, k<K → A i k=A' i k) (hb : ∀ k, k<K → B k j=B' k j) :
    Target.native R K BK A B i j=Target.native R K BK A' B' i j := by
  have blocks : Target.block (Instructions.referenceInstructions R) K BK A B i j=
      Target.block (Instructions.referenceInstructions R) K BK A' B' i j := by
    funext t seed
    have aeq : (fun k : Fin BK => if t*BK+k.val<K then A i (t*BK+k.val) else FP.Value.zero .fp32)=
        (fun k : Fin BK => if t*BK+k.val<K then A' i (t*BK+k.val) else FP.Value.zero .fp32) := by
      funext k
      split
      · exact ha _ (by assumption)
      · rfl
    have beq : (fun k : Fin BK => if t*BK+k.val<K then B (t*BK+k.val) j else FP.Value.zero .fp32)=
        (fun k : Fin BK => if t*BK+k.val<K then B' (t*BK+k.val) j else FP.Value.zero .fp32) := by
      funext k
      split
      · exact hb _ (by assumption)
      · rfl
    simp only [Target.block,aeq,beq]
  simp only [Target.native,blocks]

theorem real_formula (rest : Algebra ℝ) (g : Kernel.Geometry)
    (vals : ChannelValues ℝ (IO.inputs g)) (s : State ℝ)
    (h : (IO.matmulIO (RealExecution.backend rest) .source g).InputBinding vals s)
    (i j : Nat) (hi : i<g.rows) (hj : j<g.columns) :
    RealTarget.mathematical g.inner (Ideal.inputA g.request s) (Ideal.inputB g.request s) i j=
      Target.mathematical g.inner (IO.matrixA g vals) (IO.matrixB g vals) i j := by
  apply Finset.sum_congr rfl
  intro k _
  rw [show Ideal.inputA g.request s i k.val=IO.matrixA g vals i k.val from
      inputA _ _ _ _ _ h i k hi k.isLt,
    show Ideal.inputB g.request s k.val j=IO.matrixB g vals k.val j from
      inputB _ _ _ _ _ h k j k.isLt hj]

theorem native_formula (M : Algebra Word) (g : Kernel.Geometry)
    (vals : ChannelValues Word (IO.inputs g)) (s : State Word)
    (h : (IO.matmulIO M .accumulator g).InputBinding vals s)
    (R : Instructions.Reference) (i j : Nat) (hi : i<g.rows) (hj : j<g.columns) :
    Target.native R g.inner g.blockInner (Target.inputA g.request s) (Target.inputB g.request s) i j=
      Target.native R g.inner g.blockInner (IO.matrixA g vals) (IO.matrixB g vals) i j := by
  apply native_congr
  · intro k hk; exact inputA _ _ _ _ _ h i k hi hk
  · intro k hk; exact inputB _ _ _ _ _ h k j hk hj

#axiomsClean native_congr
#axiomsClean real_formula
#axiomsClean native_formula
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IOBinding
