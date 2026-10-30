import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.RealReference
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.WholeVJP
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Physical
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Public

/-- Values depend only on logical entry inputs. Absent seed/cotangents are
zero-valued logical inputs, as defined by HostABI.effectiveInputs. -/
structure MathematicalResult (c : Shape) (entry result : Heap ℝ) (scale : ℝ)
    (returnFinal returnInitial : Bool) : Prop where
  output : ∀ h t j,h<c.heads → t<c.T → j<c.V →
    result ⟨.output,h,0,t,j⟩=MathTarget.output c (inputsOfHeap entry) scale Bool.true h t j
  query : ∀ h t e,h<c.heads → t<c.T → e<c.K →
    result ⟨.dq,h,0,t,e⟩=MathTarget.queryGradient c (inputsOfHeap entry) scale Bool.true h t e
  key : ∀ h t e,h<c.heads → t<c.T → e<c.K →
    result ⟨.dk,h,0,t,e⟩=MathTarget.keyGradient c (inputsOfHeap entry) scale h t e
  value : ∀ h t j,h<c.heads → t<c.T → j<c.V →
    result ⟨.dv,h,0,t,j⟩=MathTarget.valueGradient c (inputsOfHeap entry) scale h t j
  final : returnFinal=Bool.true → ∀ h e j,h<c.heads → e<c.K → j<c.V →
    result ⟨.finalState,h,0,e,j⟩=MathTarget.finalState c (inputsOfHeap entry) Bool.true h e j
  initial : returnInitial=Bool.true → ∀ h e j,h<c.heads → e<c.K → j<c.V →
    result ⟨.initialGradient,h,0,e,j⟩=MathTarget.initialGradient c (inputsOfHeap entry) scale h e j
  inputs : ∀ p,p.region∈inputRegions → result p=entry p
  frame : ∀ p,p∉Source.footprint c returnFinal →
    p∉(if returnInitial then Source.eventsInitial c else []) → result p=entry p

/-- Complete real interpretation of all four kernels, host dV sum, optional
final/initial states, preserved inputs and frame. This is not an FP=Real claim. -/
theorem host_real (c : Shape) (_abi : HostABI.LaunchABI c) (scale : ℝ) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (rf ri : Bool) (entry : Heap ℝ) :
    MathematicalResult c entry (Source.runAPI realOps c scale tree rf ri entry) scale rf ri := by
  have same := Source.runAPI_agrees realOps c scale tree rf ri entry
  constructor
  · intro h t j hh ht hj
    rw [same _ (by simp),Source.run_output realOps c scale tree rf entry h t j hh ht hj,
      RealScalar.reference_output c _ scale h t j ht hj]
  · intro h t e hh ht he
    rw [same _ (by simp),Source.run_dq realOps c scale tree rf entry h t e hh ht he,
      RealScalar.reference_query c _ scale h t e ht he]
  · intro h t e hh ht he
    rw [same _ (by simp),Source.run_dk realOps c scale tree rf entry h t e hh ht he,
      RealScalar.reference_key c _ scale h t e ht he]
  · intro h t j hh ht hj
    rw [same _ (by simp),Source.run_dv realOps c scale tree rf entry complete h t j hh ht hj,
      RealScalar.reference_value c _ scale tree complete h t j ht hj]
  · intro flag h e j hh he hj
    subst rf
    rw [same _ (by simp),Source.run_final realOps c scale tree entry h e j hh he hj,
      RealScalar.reference_final c _ h e j he hj]
  · intro flag h e j hh he hj
    subst ri
    rw [Source.runAPI_initial realOps c scale tree rf entry h e j hh he hj,
      RealScalar.real_cast,RealScalar.reverse_scalar c _ scale h c.chunks e j le_rfl he hj,
      ScalarSource.initial_gradient_correct]
  · intro p hp
    rw [same p (by intro he; simp [he,inputRegions] at hp)]
    exact Source.run_inputs realOps c scale tree rf entry p hp
  · exact Source.runAPI_frame realOps c scale tree rf ri entry

theorem host_physical_real (c : Shape) (abi : HostABI.LaunchABI c) (binding : Binding c)
    (scale : ℝ) (tree : SumTree) (complete : tree.Complete c.keyBlocks) (rf ri : Bool)
    (entry : Memory ℝ) :
    MathematicalResult c (binding.view realOps.zero entry)
      (binding.view realOps.zero (Source.runPhysicalAPI realOps c binding scale tree rf ri entry))
      scale rf ri := by
  rw [Source.run_physical]
  exact host_real c abi scale tree complete rf ri _

/-- The complete input-only differential is the pairing with all returned
mathematical gradients; seed absence has zero seed variation. -/
theorem mathematical_vjp (c : Shape) (xs : Inputs ℝ) (variation : MathTarget.Variation)
    (scale : ℝ) (hasSeed : Bool) :
    MathTarget.fullVariation c xs variation scale hasSeed=
      MathTarget.gradientPairing c xs variation scale hasSeed :=
  MathTarget.whole_vjp c xs variation scale hasSeed

#axiomsClean host_real
#axiomsClean host_physical_real
#axiomsClean mathematical_vjp
#specNonCircular MathematicalResult avoiding [Source.runAPI,Source.runPhysicalAPI,host_real]
end Public
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

