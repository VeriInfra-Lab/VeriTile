import VeriTile.Triton.DSL
import bench.tritonbench_g.attention_llama.Proofs.Mathematical_RealKernel
import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealLoads
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Memory.TraceRewrite

/-! Connect the complete FP8 sources to their real execution bodies. The
reference real recurrence supplies the unchanged statements; each K/V load
is replaced by an actual int8 load and E5M2 decoding. The source's three
assignments at each conversion site may then be fused, preserving both the
complete execution result and address safety. Decoding is never erased.
This connection does not yet prove the independent attention formula. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSource
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false

structure ByteLoad where
  shape : TileShape
  name : RegName
  bytes : Op .int shape

/-- Locate K/V loads by their destination, preserving the pointer and mask.
The complete-source correspondence below checks that their padding is zero. -/
def byteLoad? : Stmt → Option ByteLoad
  | .assign .real sh name (.load .real (.ptr ptr) (.maskOther mask _)) =>
      if name == "k" || name == "v" then
        some ⟨sh, name, .load .int (.ptr ptr)
          (.maskOther mask (.broadcast (.constInt 0) sh))⟩
      else none
  | _ => none

def lowerLoad (fused : Bool) (st : Stmt) : List Stmt :=
  match byteLoad? st with
  | none => [st]
  | some load =>
      if fused then
        [.assign .real load.shape load.name (.decodeFloat8E5 .real load.bytes)]
      else FP8RealLoads.conversion load.name load.bytes

private theorem lowerLoad_fuse (st : Stmt) (s : BlockState) :
    stepStmts (lowerLoad Bool.false st) s = stepStmts (lowerLoad Bool.true st) s := by
  cases h : byteLoad? st with
  | none => simp [lowerLoad, h]
  | some load =>
    simp only [lowerLoad, h, Bool.false_eq_true, ↓reduceIte]
    exact FP8RealLoads.fuse_conversion ..

private theorem lowerList_fuse (code : List Stmt) (s : BlockState) :
    stepStmts (code.flatMap (lowerLoad Bool.false)) s =
      stepStmts (code.flatMap (lowerLoad Bool.true)) s := by
  induction code generalizing s with
  | nil => rfl
  | cons st rest ih =>
    simp only [List.flatMap_cons, ProgramRewrite.append, lowerLoad_fuse, ih]

private theorem ref_safe (bounds : RegionBounds) (s : BlockState) (d : TileDType)
    (sh : TileShape) (name : RegName) :
    (Op.ref d sh name).SafeAt bounds s = True := by rw [Op.SafeAt.eq_def]

private theorem decode_safe (bounds : RegionBounds) (s : BlockState)
    (bytes : Op .int shape) :
    @Op.SafeAt .real shape bounds s (.decodeFloat8E5 .real bytes) =
      bytes.SafeAt bounds s := by rw [Op.SafeAt.eq_def]

private theorem conversion_safe (bounds : RegionBounds) (name : RegName)
    (bytes : Op .int shape) (s : BlockState) :
    Stmt.TraceSafeList bounds (FP8RealLoads.conversion name bytes) s ↔
      bytes.SafeAt bounds s := by
  simp only [FP8RealLoads.conversion, Stmt.TraceSafeList, Stmt.TraceSafe,
    decode_safe, ref_safe, true_and, and_true]
  constructor
  · exact And.left
  · intro h
    refine ⟨h, ?_⟩
    split <;> try trivial
    split <;> try trivial
    split <;> trivial

private theorem lowerLoad_safe (bounds : RegionBounds) (st : Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (lowerLoad Bool.false st) s ↔
      Stmt.TraceSafeList bounds (lowerLoad Bool.true st) s := by
  cases h : byteLoad? st with
  | none => simp [lowerLoad, h]
  | some load =>
    simp only [lowerLoad, h, Bool.false_eq_true, ↓reduceIte, conversion_safe,
      Stmt.TraceSafeList, Stmt.TraceSafe, decode_safe, and_true]
    cases stepStmt (.assign .real load.shape load.name
      (.decodeFloat8E5 .real load.bytes)) s <;> simp

private theorem lowerList_safe (bounds : RegionBounds) (code : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (code.flatMap (lowerLoad Bool.false)) s ↔
      Stmt.TraceSafeList bounds (code.flatMap (lowerLoad Bool.true)) s := by
  induction code generalizing s with
  | nil => rfl
  | cons st rest ih =>
    simp only [List.flatMap_cons, TraceRewrite.append_iff, lowerLoad_safe, lowerLoad_fuse, ih]

/-- Erased source maximum is retained as max2; all remaining recurrence
statements are the same as the real mathematical body. -/
def sourceTail (skn svk N BM BN BD : Nat) (candidate : Bool) : List Stmt :=
  Stmt.assign .real [BM] "m_curr"
    (.max2 Broadcast.nil.consSame
      (.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (.ref .real [BM, BN] "qk"))
      (.ref .real [BM] "m_prev")) ::
    (alLoopTailG skn svk N BM BN BD candidate).drop 1

private theorem max_head_step {n : Nat} (x y : Op .real [n])
    (name : RegName) (rest : List Stmt) (s : BlockState) :
    stepStmts (.assign .real [n] name (.max2 (.consSame .nil) x y) :: rest) s =
      stepStmts (.assign .real [n] name (.maximumFloat .real (.consSame .nil) x y) :: rest) s := by
  simp only [stepStmts, stepStmt, FloatDType.toTileDType]
  erw [evalOp_maximumFloat_real]

/-- Only the exact real maximum representation changes; byte loads and
conversion ordering remain in the complete tail. -/
theorem sourceTail_lower_step (fused : Bool) (skn svk N BM BN BD : Nat)
    (candidate : Bool) (s : BlockState) :
    stepStmts ((sourceTail skn svk N BM BN BD candidate).flatMap (lowerLoad fused)) s =
      stepStmts ((alLoopTailG skn svk N BM BN BD candidate).flatMap (lowerLoad fused)) s := by
  exact max_head_step _ _ "m_curr"
    (((alLoopTailG skn svk N BM BN BD candidate).drop 1).flatMap (lowerLoad fused)) s

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal candidate : Bool)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSource
