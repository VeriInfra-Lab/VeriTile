-- GENERATED from the unchanged pinned Python AST by translate_source.py.
-- Do not replace this tree with a ready-lock stage or an unrolled poll count.
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_SourcePlan
namespace VeriTile.Bench.TritonBenchG.SpinningLock.Frontend

def originalTypedSource : List Statement :=
  [.integer "pid" .pid,
  .integer "pid_m" (.sdiv .pid (.param .numSms)),
  .integer "pid_n" (.srem .pid (.param .numSms)),
  .zeroFP32 "acc" .bm .bn,
  .forRange "iters" 1 10 [.branch (.eq (.srem .pid (.param .groupK)) (.literal 0)) [.integer "next_pid" (.add .pid (.literal 1)),
  .whileLoop (.pure (.and (.lt .next (.add .pid (.param .groupK))) (.lt .next (.param .numSms)))) [.whileLoop (.casNe ⟨.locks, [.next]⟩ (.literal 1) (.literal 1) (.literal 1)) [.pass],
  .range "rm1" .bm,
  .range "rn1" .bn,
  .pointer "P_" ⟨.partials, [(.mul (.mul .next (.param .bm)) (.param .bn)), (.mul .row (.param .bn)), .col]⟩,
  .loadFP32 "acc1" ⟨.partials, [(.mul (.mul .next (.param .bm)) (.param .bn)), (.mul .row (.param .bn)), .col]⟩,
  .addFP32 "acc" "acc1",
  .integer "next_pid" (.add .next (.literal 1))]] [.range "rm1" .bm,
  .range "rn1" .bn,
  .pointer "P_" ⟨.partials, [(.mul (.mul .pid (.param .bm)) (.param .bn)), (.mul .row (.param .bn)), .col]⟩,
  .storeFP32 ⟨.partials, [(.mul (.mul .pid (.param .bm)) (.param .bn)), (.mul .row (.param .bn)), .col]⟩ "acc" none,
  .xchg ⟨.locks, [.pid]⟩ (.literal 1)],
  .integer "rm" (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row),
  .integer "rn" (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col),
  .pointer "C_" ⟨.output, [(.mul (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row) (.param .strideM)), (.mul (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col) (.param .strideN))]⟩,
  .mask "mask" (.and (.lt (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row) (.param .m)) (.lt (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col) (.param .n))),
  .storeFP32 ⟨.output, [(.mul (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row) (.param .strideM)), (.mul (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col) (.param .strideN))]⟩ "acc" (some (.and (.lt (.add (.mul (.sdiv .pid (.param .numSms)) (.param .bm)) .row) (.param .m)) (.lt (.add (.mul (.srem .pid (.param .numSms)) (.param .bn)) .col) (.param .n))))]]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock.Frontend
