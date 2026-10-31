import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealSource
import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealMemory

/-! One complete FP8 attention iteration, related to the same real recurrence
on decoded input memory. The relation preserves all state channels and failure.
Only the actual masked K/V lanes need finite signed-byte encodings. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealIteration
open VeriTile.Triton RealKernel FP8RealMemory
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

private theorem decode_cons (K V : RegionName) (s : BlockState) (name : RegName)
    (ptr : Op .ptr sh) (mask : Op .bool sh) (ptrs : Tile .ptr sh) (masks : Tile .bool sh)
    (lhs rhs : List Stmt)
    (hp : evalOp ptr s = some ptrs) (hm : evalOp mask s = some masks)
    (hpure : ptr.SafeAt (fun _ => 0) s) (mpure : mask.SafeAt (fun _ => 0) s)
    (hregions : ∀ i, masks.data i = Bool.true → (ptrs.data i).1 = K ∨ (ptrs.data i).1 = V)
    (hfinite : ∀ i, masks.data i = Bool.true →
      FiniteByte (s.readMemValue .int (ptrs.data i).1 (ptrs.data i).2))
    (htail : ∀ tile : Tile .real sh,
      (stepStmts lhs (s.setReg name .real sh tile)).map (decoded K V) =
        stepStmts rhs (decoded K V (s.setReg name .real sh tile))) :
    (stepStmts (.assign .real sh name (.decodeFloat8E5 .real
      (.load .int (.ptr ptr) (.maskOther mask (.broadcast (.constInt 0) sh)))) :: lhs) s).map
        (decoded K V) =
      stepStmts (.assign .real sh name
        (.load .real (.ptr ptr) (.maskOther mask (.broadcast (.const 0) sh))) :: rhs)
        (decoded K V s) := by
  have he := masked_decode K V s ptr mask ptrs masks hp hm hpure mpure hregions hfinite
  simp only [stepStmts, stepStmt]
  rw [← he]
  cases hv : @evalOp .real sh (.decodeFloat8E5 .real
      (.load .int (.ptr ptr) (.maskOther mask (.broadcast (.constInt 0) sh)))) s with
  | none => simp [hv]
  | some tile => simpa [hv, decoded_setReg] using htail tile

/-- The source mask, shared by the K and V loads, tests the global key index. -/
def keyMask (N BN BD : Nat) : Op .bool [BN, BD] :=
  .remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
    (.lt .nat Broadcast.scalarR
      (.expandDim ⟨1, by simp⟩ (.ref .nat [BN] "block_n_offs")) (.constNat N))

private theorem load_cons (K V R : RegionName) (s : BlockState)
    (name ptrName : RegName) (base srow scol SN N BN BD : Nat) (lhs rhs : List Stmt)
    (hregion : R = K ∨ R = V)
    (hptr : s.regs .ptr [BN, BD] ptrName = some (alPtrTile R base srow scol SN BN BD))
    (hrows : s.regs .nat [BN] "block_n_offs" =
      some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hbytes : ∀ (j : Fin BN) (d : Fin BD), SN + j.val < N →
      FiniteByte (s.readMemValue .int R (base + (SN + j.val) * srow + d.val * scol)))
    (htail : ∀ tile : Tile .real [BN, BD],
      (stepStmts lhs (s.setReg name .real [BN, BD] tile)).map (decoded K V) =
        stepStmts rhs (decoded K V (s.setReg name .real [BN, BD] tile))) :
    (stepStmts (.assign .real [BN, BD] name (.decodeFloat8E5 .real
      (.load .int (.ptr (.ref .ptr [BN, BD] ptrName))
        (.maskOther (keyMask N BN BD) (.broadcast (.constInt 0) [BN, BD])))) :: lhs) s).map
        (decoded K V) =
      stepStmts (.assign .real [BN, BD] name
        (.load .real (.ptr (.ref .ptr [BN, BD] ptrName))
          (.maskOther (keyMask N BN BD) (.broadcast (.const 0) [BN, BD]))) :: rhs)
        (decoded K V s) := by
  apply decode_cons K V s name _ _ (alPtrTile R base srow scol SN BN BD)
    (⟨fun i => decide (SN + i.1.val < N)⟩ : Tile .bool [BN, BD]) lhs rhs
  · simpa only [evalOp_ref] using hptr
  · exact al_colmask_eval s SN N hrows
  · simp [Op.SafeAt.eq_def]
  · simp [keyMask, Op.SafeAt.eq_def]
  · intro i _; exact hregion
  · intro i hi
    exact hbytes i.1 i.2.1 (by simpa using hi)
  · exact htail

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealIteration
