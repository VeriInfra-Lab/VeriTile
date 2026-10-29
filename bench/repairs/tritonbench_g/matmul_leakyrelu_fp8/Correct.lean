import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Success
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Real

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
set_option maxHeartbeats 8000000

theorem reference_real (args : Host.Arguments) (out : Tensor) (layout : Layout args.problem ⟨args.A,args.B,out⟩)
    (memory : Memory ℝ) (row column : Nat) (hr : row < args.A.rows) (hc : column < args.B.columns) :
    reference realOps args memory row column = Real.formula args.problem ⟨args.A,args.B,out⟩ memory row column := by
  by_cases zero : args.A.columns = 0
  · simp [reference,zero,realOps,Real.formula,Real.activate,RealMath.formula,Host.Arguments.problem]
  · have source := grid_reference realOps args out layout memory zero row column hr hc
    rw [source_run_real] at source
    exact source.symm.trans (Real.output args.problem ⟨args.A,args.B,out⟩ layout memory row column hr hc)

/-- Complete independent dense Real result for the actual checked call.
Metadata describes the Tensor actually returned by the allocator; values
are relative to the actual caller memory before allocation. -/
specification correct (allocate : Allocator σ) (abi : LocalABI ℝ σ realOps) (args : Host.Arguments)
    (law : AllocationLaw allocate abi live) (trials : Nat) (failures : Failures)
    (before after : DeviceState σ) (out : Tensor)
    (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live before.payload args.A.storage) (bLive : live before.payload args.B.storage)
    (success : call allocate abi (.tensors args) trials failures before = .ok out after) :
    MetadataContract args ∧ (∃ ready,
      allocate.empty args.A.rows args.B.columns .fp16 args.A.device before = .ok out ready ∧
      Host.Allocation live args.A.rows args.B.columns .fp16 args.A.device before.payload out ready.payload) ∧
    (∀ row column, row < args.A.rows → column < args.B.columns →
      abi.observe after.payload (out.address row column) =
        Real.formula args.problem ⟨args.A,args.B,out⟩ (abi.observe before.payload) row column) ∧
    after.currentDevice = before.currentDevice := by
  obtain ⟨admitted,allocated,words,device⟩ := values allocate abi args law trials failures before after out aBound bBound aLive bLive success
  obtain ⟨ready,returned,allocation⟩ := allocated
  have input := checked_input live args before.payload ((checks_iff args).mpr admitted) aBound bBound aLive bLive
  refine ⟨admitted,⟨ready,returned,allocation⟩,?_,device⟩
  intro row column hr hc
  rw [words row column hr hc,reference_real args out (allocation_layout allocation input) _ row column hr hc]

#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
