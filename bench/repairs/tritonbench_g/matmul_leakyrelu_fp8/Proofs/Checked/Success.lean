import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Host
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Reference

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
set_option maxHeartbeats 8000000

/-- Actual allocation descriptor and caller-indexed values, obtained from the
checked whole host call using only one-event machine laws. -/
theorem values (allocate : Allocator σ) (abi : LocalABI α σ ops) (args : Host.Arguments)
    (law : AllocationLaw allocate abi live) (trials : Nat) (failures : Failures)
    (before after : DeviceState σ) (out : Tensor)
    (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live before.payload args.A.storage) (bLive : live before.payload args.B.storage)
    (success : call allocate abi (.tensors args) trials failures before = .ok out after) :
    MetadataContract args ∧ (∃ ready,
      allocate.empty args.A.rows args.B.columns .fp16 args.A.device before = .ok out ready ∧
      Host.Allocation live args.A.rows args.B.columns .fp16 args.A.device before.payload out ready.payload) ∧
    (∀ row column, row < args.A.rows → column < args.B.columns →
      abi.observe after.payload (out.address row column) = reference ops args (abi.observe before.payload) row column) ∧
    after.currentDevice = before.currentDevice := by
  obtain ⟨admitted,ready,allocated,allocation,payload,device⟩ :=
    call_success allocate abi args law trials failures before after out success
  have input := checked_input live args before.payload ((checks_iff args).mpr admitted) aBound bBound aLive bLive
  have layout := allocation_layout allocation input
  have native := native_refines abi (computationCode args out trials) ready.payload
  rw [eval_computation ops args out layout] at native
  have memory : abi.observe after.payload = resultMemory ops args out (abi.observe ready.payload) := by
    rw [payload]
    exact congrArg Prod.snd native
  refine ⟨admitted,⟨ready,allocated,allocation⟩,?_,device⟩
  intro row column hr hc
  rw [memory,result_reference ops args out layout _ row column hr hc]
  apply reference_stable
  intro point read
  apply law.frame _ _ _ _ _ _ _ allocated
  rcases read with a | b
  · simpa only [a] using aLive
  · simpa only [b] using bLive

#axiomsClean values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
