import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealProgram

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealArithmetic
open VeriTile.Triton

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)

theorem original_arithmetic :
    (loopBody? (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body).map
        (fun code => (code.drop 3).take 9) =
      some (body BN BD N offset size window complement evenM Bool.false) := by
  rw [RealProgram.original_source]
  rfl

theorem candidate_arithmetic :
    (loopBody? (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body).map
        (fun code => (code.drop 3).take 9) =
      some (body BN BD N offset size window complement evenM Bool.true) := by
  rw [RealProgram.candidate_source]
  rfl


end

end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealArithmetic

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealBody
open VeriTile.Triton

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)
theorem original_loop :
    (RealArithmetic.loopBody? (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body) =
      some (body BN BD N offset size window complement evenM evenN Bool.false) := by
  rw [RealProgram.original_source]
  rfl

theorem candidate_loop :
    (RealArithmetic.loopBody? (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body) =
      some (body BN BD N offset size window complement evenM evenN Bool.true) := by
  rw [RealProgram.candidate_source]
  rfl


end

end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealBody

