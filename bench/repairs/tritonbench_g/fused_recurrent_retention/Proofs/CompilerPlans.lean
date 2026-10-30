import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Numeric
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
namespace CompilerPlans
def dot1lanes1 : DotPlan := (.product 0)
theorem dot1lanes1_covers : dot1lanes1.Covers 1 := by unfold DotPlan.Covers; decide
#axiomsClean dot1lanes1_covers
def dot16lanes2 : DotPlan := (.add (.fma 14 (.fma 12 (.fma 10 (.fma 8 (.fma 6 (.fma 4 (.fma 0 (.product 2)))))))) (.fma 15 (.fma 13 (.fma 11 (.fma 9 (.fma 7 (.fma 5 (.fma 1 (.product 3)))))))))
theorem dot16lanes2_covers : dot16lanes2.Covers 16 := by unfold DotPlan.Covers; decide
#axiomsClean dot16lanes2_covers
def dot16lanes16 : DotPlan := (.add (.add (.add (.fma 0 (.product 8)) (.fma 4 (.product 12))) (.add (.fma 2 (.product 10)) (.fma 6 (.product 14)))) (.add (.add (.fma 1 (.product 9)) (.fma 5 (.product 13))) (.add (.fma 3 (.product 11)) (.fma 7 (.product 15)))))
theorem dot16lanes16_covers : dot16lanes16.Covers 16 := by unfold DotPlan.Covers; decide
#axiomsClean dot16lanes16_covers
def dot32lanes1 : DotPlan := (.fma 31 (.fma 30 (.fma 29 (.fma 28 (.fma 27 (.fma 26 (.fma 25 (.fma 24 (.fma 23 (.fma 22 (.fma 21 (.fma 20 (.fma 19 (.fma 18 (.fma 17 (.fma 16 (.fma 15 (.fma 14 (.fma 13 (.fma 12 (.fma 11 (.fma 10 (.fma 9 (.fma 8 (.fma 7 (.fma 6 (.fma 5 (.fma 4 (.fma 3 (.fma 2 (.fma 0 (.product 1))))))))))))))))))))))))))))))))
theorem dot32lanes1_covers : dot32lanes1.Covers 32 := by unfold DotPlan.Covers; decide
#axiomsClean dot32lanes1_covers
def dot32lanes32 : DotPlan := (.add (.add (.add (.add (.fma 0 (.product 16)) (.fma 8 (.product 24))) (.add (.fma 4 (.product 20)) (.fma 12 (.product 28)))) (.add (.add (.fma 2 (.product 18)) (.fma 10 (.product 26))) (.add (.fma 6 (.product 22)) (.fma 14 (.product 30))))) (.add (.add (.add (.fma 1 (.product 17)) (.fma 9 (.product 25))) (.add (.fma 5 (.product 21)) (.fma 13 (.product 29)))) (.add (.add (.fma 3 (.product 19)) (.fma 11 (.product 27))) (.add (.fma 7 (.product 23)) (.fma 15 (.product 31))))))
theorem dot32lanes32_covers : dot32lanes32.Covers 32 := by unfold DotPlan.Covers; decide
#axiomsClean dot32lanes32_covers
end CompilerPlans
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
