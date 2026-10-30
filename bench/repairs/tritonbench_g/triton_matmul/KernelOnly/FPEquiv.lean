import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.Pair

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.FPEquiv
open VeriTile Triton FP.Structural FPValue
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def original (rest : Algebra Word) (I : Native.Instructions) (g : Geometry) (ids : List Nat) :=
  ProgramIO.Implementation.original (Pair.io rest I g ids) (fun _ => True)
def shifted (rest : Algebra Word) (I : Native.Instructions) (g : Geometry) (ids : List Nat) :=
  ProgramIO.Implementation.ofVariant (Pair.variant rest I g ids) (fun _ => True)

/-- Complete source schedules with the original FP16/E4M3 storage, FP32 carry,
K order, tails and any complete grid permutation. Only exact group-width integer
arithmetic changes, with a generic fallback for GROUP_SIZE_M other than 8. -/
specification grid_equiv (rest : Algebra Word) (I : Native.Instructions) (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    original rest I g ids ≡[[]] shifted rest I g ids :=
  ProgramIO.Implementation.of_equivalent (Pair.variant rest I g ids)
    (Pair.equivalent rest I g ids order sepA sepB)

#axiomsClean grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.FPEquiv
