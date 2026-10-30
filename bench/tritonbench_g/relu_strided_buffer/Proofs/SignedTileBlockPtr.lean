import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedTile

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

def natBlockStore (region : RegionName) (base stride n start B : Nat) : Stmt :=
  .store .fp32 [B] (.blockPtr (natBlockPtr region base stride n start B) [0])
    (.castFloat .fp32 .fp32 (.ref .fp32 [B] "out0")) .none

theorem natBlockStore_eq (region : RegionName) (base stride n start B : Nat) (s : BlockState) :
    stepStmt (natBlockStore region base stride n start B) s =
      stepStmt (tileStore ⟨region, base, stride⟩ n start B) s := by
  simp only [natBlockStore, tileStore, stepStmt, natBlockPtr_eval,
    addressOp_eval, maskOp_eval, bind, Option.bind, Option.map]
  congr 1
  funext values
  refine congrArg some ?_
  apply List.foldl_ext
  intro acc i _
  rcases i with ⟨i, u⟩
  simp [TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d, nat_address]

def natBlockBody (input output : RegionName)
    (inputBase outputBase inputStride outputStride n start B : Nat) : List Stmt :=
  [.assign .fp32 [B] "in0" (.castFloat .fp32 .fp32
    (natBlockLoad input inputBase inputStride n start B)),
   .assign .fp32 [B] "out0" (reluOp B),
   natBlockStore output outputBase outputStride n start B]

/-- Entire three-statement typed tile body agrees with the block-pointer body
on natural strides, including padding, both casts, exact state and failures. -/
theorem natBlockBody_eq (input output : RegionName)
    (inputBase outputBase inputStride outputStride n start B : Nat) (s : BlockState) :
    stepStmts (natBlockBody input output inputBase outputBase inputStride outputStride n start B) s =
      stepStmts (tileBody ⟨input, inputBase, inputStride⟩
        ⟨output, outputBase, outputStride⟩ n start B) s := by
  have hl : stepStmt (.assign .fp32 [B] "in0" (.castFloat .fp32 .fp32
      (natBlockLoad input inputBase inputStride n start B))) s =
      stepStmt (.assign .fp32 [B] "in0" (.castFloat .fp32 .fp32
        (load32 ⟨input, inputBase, inputStride⟩ n start B))) s := by
    have he := (cast32_identity (natBlockLoad input inputBase inputStride n start B) s).trans
      ((natBlockLoad_eq input inputBase inputStride n start B s).trans
        (cast32_identity (load32 ⟨input, inputBase, inputStride⟩ n start B) s).symm)
    simp only [stepStmt]
    exact congrArg (fun value : Option (Tile .fp32 [B]) =>
      value.bind (fun value => some (s.setReg "in0" .fp32 [B] value))) he
  simp only [natBlockBody, tileBody, stepStmts, hl, natBlockStore_eq]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedTile
