import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSoftware
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Spec

/-! An admitted, exactly normalized FP32 row whose rounded second prefix
crosses the canonical seed=0/offset=0 sample. This distinguishes the exact
Real inverse CDF from the selected source-precision target. The CPU source
replay checks the resulting label in the complete kernel. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling.Boundary
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def u : FP.Value .fp32 := ⟨0x3f4c4fd1⟩
def a : FP.Value .fp32 := ⟨0x3f4c4fd0⟩
def b : FP.Value .fp32 := ⟨0x33400000⟩
def c : FP.Value .fp32 := ⟨0x3e4ec0bd⟩

theorem rounded_boundary :
    (Random.rand 0 0 .bits32 10).bits=u.bits ∧
    a.toRat?=some (836861/1048576) ∧ b.toRat?=some (3/67108864) ∧
    c.toRat?=some (13549757/67108864) ∧ u.toRat?=some (13389777/16777216) ∧
    (FP.add .ieeeValue a b).bits=u.bits ∧
    FP.lt .ieeeValue (FP.sub .ieeeValue u b) u=Bool.true ∧
    FP.eq .ieeeValue u u=Bool.true := by decide +kernel

theorem exactly_normalized :
    (836861/1048576 : ℝ)+3/67108864+13549757/67108864=1 ∧
    (836861/1048576 : ℝ)+3/67108864<13389777/16777216 := by norm_num

#axiomsClean rounded_boundary
#axiomsClean exactly_normalized
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Boundary
