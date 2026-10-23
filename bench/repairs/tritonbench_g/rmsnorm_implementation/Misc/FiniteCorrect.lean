import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.Native

/-! Optional finite-machine companion: the checked guard maxima imply every
signed-i32 intermediate, final loop increment, padded 64-bit pointer, active
four-byte access, and full output-address coverage. These are structural API
facts, with no numerical assumptions. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.FiniteCorrect
abbrev checked_native_addresses := @native_correspondence
#axiomsClean checked_native_addresses
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsnormImplementation.FiniteCorrect
