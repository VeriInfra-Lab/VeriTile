import bench.tritonbench_g.chunk_retention.Proofs.Decay
namespace VeriTile.Bench.TritonBenchG.ChunkRetention
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 5000000
private theorem signed_lane_eval (width n : Nat) (u : BlockState) :
    evalOp (.intToSigned width (.castNatToInt (.arange n))) u =
      some (Tile.vec (fun i : Fin n =>
        let bits : Nat := i.val % 2^width
        if 2*bits < 2^width then (bits : Int) else (bits : Int)-2^width)) := by
  simp only [Op.intToSigned,Op.unsignedToInt,Op.intToUnsigned,Op.natToUnsigned,
    evalOp,bind,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [Tile.cop,Tile.bop,Tile.select,Tile.scalar,Tile.vec,
    IntegerExpr.sameShape,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.sub,IntegralDType.mod,ComparableDType.lt]
  have hp : (2^width : Int) = ((2^width : Nat) : Int) := by norm_cast
  have hr : Int.toNat (Int.ofNat i.val % (2^width : Int)) = i.val % 2^width := by
    rw [hp]
    norm_cast
  simp only [decide_eq_true_eq]
  simp only [hr,Nat.mod_mod]
  rfl

theorem signed32_lane_eval (n : Nat) (u : BlockState) :
    evalOp (.intToSigned 32 (.castNatToInt (.arange n))) u =
      some (Tile.vec (fun i : Fin n => Decay.index32 i.val)) := signed_lane_eval 32 n u

#axiomsClean signed32_lane_eval
end VeriTile.Bench.TritonBenchG.ChunkRetention
