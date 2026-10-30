import bench.repairs.tritonbench_g.cross_entropy_ops.Kernels

namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelAudit
open VeriTile.Triton

/-- The 14/16 original top-level statements are retained. Lowering adds one
internal load temporary to each body, yielding 15/17 compute statements; the
raw Python/Lean statement correspondence is checked separately. -/
theorem forward_statements (p : Parameters) : (forwardSyntax p).surfaceBody.length=15 := by rfl
theorem backward_statements (p : Parameters) : (backwardSyntax p).surfaceBody.length=17 := by rfl

/-- Selected checked source is a guard wrapper over these exact resident JITs. -/
def checkedSHA256 : String := "5f106a09beaeb4807b9a8f6409e0f015478be518e8a31d9ca889a32f9cd7e768"
def residentSHA256 : String := "b690a712bd968686ee78818e81dbbb21a8d15331e132eb1c994d1ed3eb2b4ba5"
def originalSHA256 : String := "a06de9e0f3d5fa4d785de59bcf988d2945ec20c98e47a05cdc358e6f504aa11a"

#axiomsClean forward_statements
#axiomsClean backward_statements
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelAudit
