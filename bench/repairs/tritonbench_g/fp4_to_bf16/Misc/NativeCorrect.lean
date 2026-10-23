import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Fp4ToBf16.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Auxiliary native conversion to the independent BF16 table, retaining both
zero signs and every packed uint8 value. The source's signed32 operations,
FP32 bitcast and BF16 cast remain explicit in IO.lean. This is an exact native
observation, with no second kernel or deferred numerical transformation law. -/
theorem fp4_to_bf16_native (rest : Algebra Word) (h : Input)
    (storage : Finite.Storage h.shape.count) (jobs : List Nat)
    (fresh : h.input≠h.output) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    (ConversionIO.nativeIO rest h jobs).ImplementsOn
      (ConversionIO.ByteDomain h) (ConversionIO.nativeTarget h) :=
  ConversionIO.native_implements rest h storage jobs fresh gridSafe order

#stmtSurfaceSubset fp4_to_bf16_native ⊆
  [FP.Structural.Algebra,Word,FP.Value,FP.Value.zero,FloatDType.fp32,
   RegionName,instInhabitedWord,Input,Input.input,Input.output,Input.shape,Shape.count,Finite.Storage,grid,
   ConversionIO.nativeIO,ProgramIO.ImplementsOn,ConversionIO.ByteDomain,ConversionIO.nativeTarget]
#specNonCircular ConversionIO.nativeTarget avoiding [source,Kernels.source,Launch.run,FP.Structural.exec]
#axiomsClean fp4_to_bf16_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4ToBf16.FPEquiv
