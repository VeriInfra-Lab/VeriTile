import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SignedTile

/-!
Positive-stride block-pointer bridge for the selected FP32 tile body.
`start` is the already computed `tile_id * tile_size0`. The old source's
`tile_id0`/`offset0` bookkeeping and host launch are deliberately not hidden
inside this local access equivalence.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedTile
open VeriTile.Triton
open SignedView

def natBlockPtr (region : RegionName) (base stride n start B : Nat) : Op .blockPtr [B] :=
  .makeBlockPtrDynOffsets region (.constNat base) [n] [B] [stride] [.constNat start]

theorem natBlockPtr_eval (region : RegionName) (base stride n start B : Nat) (s : BlockState) :
    evalOp (natBlockPtr region base stride n start B) s = some
      (⟨fun _ => { region := region, baseOffset := base, parentShape := [n], blockShape := [B], strides := [stride], offsets := [start] }⟩ : Tile .blockPtr [B]) := by
  simp only [natBlockPtr, evalOp, List.mapM_cons, List.mapM_nil, bind, Option.bind,
    Option.pure_def]
  rfl

def natBlockLoad (region : RegionName) (base stride n start B : Nat) : Op .fp32 [B] :=
  .load .fp32 (.blockPtr (natBlockPtr region base stride n start B) [0]) .none .unspecified

theorem natBlockLoad_eq (region : RegionName) (base stride n start B : Nat) (s : BlockState) :
    evalOp (natBlockLoad region base stride n start B) s =
      evalOp (load32 ⟨region, base, stride⟩ n start B) s := by
  rw [load32_eval]
  simp only [natBlockLoad, evalOp, natBlockPtr_eval, bind, Option.bind]
  refine congrArg some ?_
  ext i
  rcases i with ⟨i, u⟩
  simp [TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d, nat_address]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedTile
