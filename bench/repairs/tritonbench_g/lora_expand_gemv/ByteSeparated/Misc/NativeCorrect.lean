import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.IO
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.FPEquiv
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section

/-- Full source-to-native observation for the original FP32 X/Y, FP16 weight,
conversion/product/reduction and FP32-add graph. Scalar primitives, shape-only
reduction plans and fusion choices retain their prior explicit model boundary.
This is not a new optimization or an assumed floating algebraic equality. -/
theorem kernel_native (rest : Algebra Word) (ops : Native.ScalarOps)
    (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (indices : Nat → Int) (bounds : Valid g indices)
    (bits : RegionName → Nat → Native.Bits) (order : List Nat)
    (coverage : order.Perm (List.range g.count)) :
    (LoraIO.io rest plans g order).ObservesOnCells
      (LoraIO.nativeObservation ops schedule bits g) (LoraIO.domain g indices)
      (LoraIO.nativeTarget ops schedule plans g indices bits) :=
  IOCorrect.native rest ops schedule plans g indices bounds bits order coverage

#axiomsClean kernel_native
#specNonCircular LoraIO.nativeTarget avoiding [execute,Branches.source,FP.Structural.exec]
#stmtSurfaceSubset kernel_native ⊆
  [ProgramIO.ObservesOnCells,LoraIO.io,LoraIO.domain,LoraIO.nativeObservation,
   LoraIO.nativeTarget,FP.Structural.Algebra,Word,FPValue.instInhabitedWord,
   ReductionPlan,Geometry,Valid,Geometry.count,RegionName,Native.ScalarOps,Native.Schedule,Native.Bits]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.FPEquiv
