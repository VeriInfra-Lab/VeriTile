import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.CacheTransform.NativeEndpoints
open VeriTile.Triton FP.Structural
noncomputable section

/-- Every native output byte comes from the independent row-gather reference;
all byte words and their FP32 groupings are preserved exactly by the source. -/
theorem kernel_native (M : Algebra Unit) (p : Parameters)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit)
    (s : State Unit) (n capacity : Nat) (rows cache : Nat → Nat)
    (input : 0<p.count → GatherInput p s n capacity rows cache) :
    ∃ t,KernelSource.launch M p s (blocks p.count)=some t ∧
      (∀ o,o<p.count → t.mem p.output o=.mk .nat (byteTarget p rows cache o)) ∧
      (∀ (r : RegionName) o,r≠p.output ∨ p.count≤o → t.mem r o=s.mem r o) :=
  KernelInterface.complete M p start block grid s n capacity rows cache input

#axiomsClean kernel_native
#specNonCircular byteTarget avoiding [source,KernelSource.launch,gather]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.NativeEndpoints
