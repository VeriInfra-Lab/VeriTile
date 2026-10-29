import bench.tritonbench_g.triton_mul2.Proofs.FiniteRefinement

namespace TritonMul2Finite
open VeriTile.Triton
noncomputable section

/-- Full finite-source contract for both original wrappers and kernels. The
reference runs all source-first failures and every finite host schedule against
actual signed-i32 offsets, signed comparison, wrapped-u64 extracted ABI
pointers, launch-context byte permissions and shared alias-backing translation.
Only the four floating multiply operations are numerical parameters. Missing
bytes/access/ownership outcomes are observations without native failure-state
claims. No narrow-size, logical-contiguity, or arithmetic-refinement premise
removes actual wrapped executions from this theorem. -/
specification triton_mul2_finite_source_outcomes (b : Numerics) (rt : Runtime)
    (contract : RuntimeContract rt) (mode : Mode) (x : Tensor) (block : Int)
    (initial : ByteMemory) :
    executeHost b rt mode x block initial=referenceHost b rt mode x block initial ∧
    (∀ ret, (executeHost b rt mode x block initial).outcome=.ok ret →
      ReturnedMetadata rt mode x initial ret) ∧
    (∀ d p input output n B pids before final,
      sourceLaunch b d p mode input output n B pids before=.ok final →
      LaunchResult b d p n B pids before final) ∧
    (∀ d p input output B pids,
      sourceLaunch b d p mode input output (2^31) B pids initial=.ok initial) ∧
    (∀ d p input output n B,
      sourceLaunch b d p mode input output n B [] initial=.ok initial) ∧
    (∀ output current setup, rt.pointers x output current=.ok setup →
      PointerSetupBinding current x output setup ∧
      (x.dataPointer≠0 → ∀ address width, InRegion (layout x output setup).deviceInput address width →
        setup.access.Resolves address ((layout x output setup).inputBacking address) width) ∧
      (output.dataPointer≠0 → ∀ address width, InRegion (layout x output setup).deviceOutput address width →
        setup.access.Resolves address ((layout x output setup).outputBacking address) width)) ∧
    (∀ input output n B,
      lowerBody (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_kernel input output n B).body=
        some (sourceCode input output n B)) ∧
    (∀ ptr n B,
      lowerBody (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_inplace_kernel ptr n B).body=
        some (sourceCode ptr ptr n B)) := by
  refine ⟨host_correspondence b rt mode x block initial,?_,?_,?_,?_,?_,original_ast_lowering,inplace_ast_lowering⟩
  · exact host_returned_metadata b rt contract mode x block initial
  · intro d p input output n B pids before final h
    exact successful_launch_result b d p mode input output n B pids before final h
  · intro d p input output B pids
    exact minimum_launch_no_access b d p mode input output B pids initial
  · intro d p input output n B
    exact zero_grid_no_access b d p mode input output n B initial
  · intro output current setup h
    have binding := contract.pointerBindings x output current setup h
    exact ⟨binding,extracted_input_resolution current x output setup binding,
      extracted_output_resolution current x output setup binding⟩

#axiomsClean triton_mul2_finite_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end TritonMul2Finite
