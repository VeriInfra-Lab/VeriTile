import bench.tritonbench_g.fp4_to_bf16.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
local instance inhabitedWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent E2M1 value of one nibble: sign, biased exponent and
one fraction bit. The high nibble is output first; exponent zero is subnormal. -/
def nibbleReal (q : Bits.Nibble) : ℝ :=
  let u := q.toNat % 8
  let e := u / 2
  let m : ℝ := (u % 2 : Nat)
  let magnitude := if e = 0 then m / 2 else (1 + m / 2) * 2 ^ (e - 1)
  if q.toNat < 8 then magnitude else -magnitude

/-- Complete finite-ABI grid: every packed byte yields its high then low
E2M1 Real value. The observed BF16 value is proved finite (`some`); every actual
source access is safe and the full exterior cell frame is preserved. -/
specification fp4_to_bf16_real (rest : Algebra Word) (h : Input)
    (storage : Finite.Storage h.shape.count) (jobs : List Nat)
    (fresh : h.input≠h.output) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    VeriTile.Spec.Real (ConversionIO.logicalRealIO rest h jobs ⊨{ConversionIO.LogicalByteDomain h}
      fun xs j => some (nibbleReal (ConversionIO.logicalNibble h xs j))) :=
  ConversionIO.logical_real_implements rest h storage jobs fresh gridSafe order

#stmtSurfaceSubset fp4_to_bf16_real ⊆
  [VeriTile.Spec.Real,FP.Structural.Algebra,Word,FP.Value,FP.Value.zero,FloatDType.fp32,
   RegionName,inhabitedWord,Input,Input.input,Input.output,Input.shape,Shape.count,Finite.Storage,grid,
   ConversionIO.logicalRealIO,LogicalProgramIO.ImplementsOn,ConversionIO.LogicalByteDomain,
   ConversionIO.logicalNibble,nibbleReal]
#specNonCircular ConversionIO.mathematical avoiding [source,Kernels.source,Launch.run,FP.Structural.exec]
#axiomsClean fp4_to_bf16_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4ToBf16.Correct
