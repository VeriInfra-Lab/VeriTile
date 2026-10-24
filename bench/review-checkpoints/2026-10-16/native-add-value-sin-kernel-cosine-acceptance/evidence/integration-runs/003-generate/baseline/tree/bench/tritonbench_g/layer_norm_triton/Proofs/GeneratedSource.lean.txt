import bench.tritonbench_g.layer_norm_triton.Proofs.SourcePlan

namespace VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend

/-- Generated from every statement of the hash-pinned original three JITs.
The external typed frontend validates the interior of compound nodes. -/
def originalTypedSource : TypedSource where
  forward := [
    .setup,
    .initializeMean,
    .meanLoop .x,
    .meanResult (.div (.slot .meanSum) (.denominator)),
    .initializeVariance,
    .varianceLoop (.mul (.mask (.sub (.x) (.slot .mean))) (.mask (.sub (.x) (.slot .mean)))),
    .varianceResult (.div (.slot .varSum) (.denominator)),
    .inverseResult (.div (.one) (.sqrt (.add (.slot .variance) (.slot .eps)))),
    .storeMean,
    .storeRstd,
    .outputLoop (.add (.mul (.mul (.sub (.x) (.slot .mean)) (.slot .rstd)) (.promoteW .w)) (.promoteBias .bias))]
  backward := [
    .setup,
    .assign 0 (.read .x),
    .assign 1 (.read .dy),
    .assign 2 (.read .w),
    .assign 3 (.read .mean),
    .assign 4 (.read .rstd),
    .assign 5 (.mul (.sub (.reg 0) (.reg 3)) (.reg 4)),
    .assign 6 (.mul (.reg 2) (.reg 1)),
    .assign 5 (.mask (.reg 5)),
    .assign 6 (.mask (.reg 6)),
    .assign 7 (.div (.sum (.mul (.reg 5) (.reg 6))) (.denominator)),
    .assign 8 (.div (.sum (.reg 6)) (.denominator)),
    .assign 9 (.mul (.sub (.reg 6) (.add (.mul (.reg 5) (.reg 7)) (.reg 8))) (.reg 4)),
    .storeDX (.reg 9),
    .assign 10 (.cast .fp32 .fp32 (.mul (.reg 1) (.reg 5))),
    .assign 11 (.cast .fp32 .fp32 (.reg 1)),
    .whileCAS 0 1 1,
    .loadCount,
    .countBranch 0 1,
    .storeDW,
    .storeDB,
    .release 0]
  finalReduce := [.setup, .zeroDW, .zeroDB, .streamDWDB, .reduceDW, .reduceDB, .storeDW, .storeDB]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend
