import bench.repairs.tritonbench_g.max_reduction.Correct

/-! Optional Python-check correspondence. The three kernel stages retain their
independent mathematical and equivalence specifications in the primary entries. -/
namespace VeriTile.Bench.Repairs.MaxReduction.TensorBridge

theorem python_global_checks (x : VeriTile.Torch.TensorMetadata) :
    checkGlobal (.tensor x) = .ok () ↔ GlobalContract x := check_global_iff x

theorem python_dimension_checks (a : DimArguments) :
    checkDimension a = .ok () ↔ DimContract a := check_dimension_iff a

#axiomsClean python_global_checks
#axiomsClean python_dimension_checks
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MaxReduction.TensorBridge
