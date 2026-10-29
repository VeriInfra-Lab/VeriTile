import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.PairInter
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.PairIntra
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.PairSplit
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.PairMerge
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.PairOutput

/-! The five actual stages of the selected FP32 source, each compared with its
existing complete candidate in Kernels.lean. Reached finite-operand conditions
are extracted from the actual repaired source. Source zero padding, both K
branches, casts, reductions, launch IDs and all other operations are retained.
The complete producer/consumer native IO results stay in Misc/NativeCorrect.lean. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []

abbrev interOriginal := PairInter.original .fp32 .fp32 .fp32 .fp32
abbrev interDotFirst := PairInter.dotFirst .fp32 .fp32 .fp32 .fp32
specification inter_equiv (q k g A : RegionName) (headStride timeStride : Nat)
    (scale : ℝ) (T K BT BC BK NC : Nat) :
    interOriginal q k g A headStride timeStride scale T K BT BC BK NC ≡[assumptions]
      interDotFirst q k g A headStride timeStride scale T K BT BC BK NC :=
  PairInter.inter_equiv .fp32 .fp32 .fp32 .fp32 q k g A headStride timeStride scale T K BT BC BK NC []

abbrev intraOriginal := PairIntra.original .fp32 .fp32 .fp32
abbrev intraScaleFirst := PairIntra.scaleFirst .fp32 .fp32 .fp32
specification intra_equiv (q k g A : RegionName) (headStride timeStride : Nat)
    (scale : ℝ) (T K BT BC BK : Nat) :
    intraOriginal q k g A headStride timeStride scale T K BT BC BK ≡[assumptions]
      intraScaleFirst q k g A headStride timeStride scale T K BT BC BK :=
  PairIntra.intra_equiv .fp32 .fp32 .fp32 q k g A headStride timeStride scale T K BT BC BK []

abbrev splitOriginal := PairSplit.original .fp32 .fp32 .fp32
abbrev splitScaleFirst := PairSplit.scaleFirst .fp32 .fp32 .fp32
specification split_equiv (q k g A : RegionName) (headStride timeStride : Nat)
    (scale : ℝ) (T K BT BC BK NC : Nat) :
    splitOriginal q k g A headStride timeStride scale T K BT BC BK NC ≡[assumptions]
      splitScaleFirst q k g A headStride timeStride scale T K BT BC BK NC :=
  PairSplit.split_equiv .fp32 .fp32 .fp32 q k g A headStride timeStride scale T K BT BC BK NC []

abbrev mergeOriginal := PairMerge.originalProgram .fp32
abbrev mergeLoadFirst := PairMerge.loadFirstProgram .fp32
specification merge_equiv (A A2 : RegionName) (T BT BC NK : Nat) :
    mergeOriginal A A2 T BT BC NK ≡[assumptions] mergeLoadFirst A A2 T BT BC NK :=
  PairMerge.merge_equiv .fp32 A A2 T BT BC NK []

abbrev outputOriginal := PairOutput.original .fp32 .fp32 .fp32 .fp32 .fp32
abbrev outputSumLeft := PairOutput.sumLeft .fp32 .fp32 .fp32 .fp32 .fp32
specification output_equiv (q v g h o A : RegionName)
    (keyHead keyTime valueHead valueTime historyHead historyKey : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    outputOriginal q v g h o A keyHead keyTime valueHead valueTime historyHead historyKey scale T K V BT BK BV ≡[assumptions]
      outputSumLeft q v g h o A keyHead keyTime valueHead valueTime historyHead historyKey scale T K V BT BK BV :=
  PairOutput.output_equiv .fp32 .fp32 .fp32 .fp32 .fp32 q v g h o A
    keyHead keyTime valueHead valueTime historyHead historyKey scale T K V BT BK BV []

#print_fp_assumptions inter_equiv
#print_fp_assumptions intra_equiv
#print_fp_assumptions split_equiv
#print_fp_assumptions merge_equiv
#print_fp_assumptions output_equiv
#axiomsClean inter_equiv
#axiomsClean intra_equiv
#axiomsClean split_equiv
#axiomsClean merge_equiv
#axiomsClean output_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv
