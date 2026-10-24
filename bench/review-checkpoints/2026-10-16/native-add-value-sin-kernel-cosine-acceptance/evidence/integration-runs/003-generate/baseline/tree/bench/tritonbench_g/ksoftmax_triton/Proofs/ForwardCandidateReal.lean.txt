import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSource
import VeriTile.Triton
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Optional draft for later root validation. This proves the candidate's
actual erased real flat execution equality directly from the exact four
address assignments and the unchanged source suffix. It does not infer a
real erasure theorem from the separate abstract typed FP equivalence.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Candidate
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def source (v : Bool) : Typed.Source :=
  if v then Typed.fp32Candidate else Typed.fp32Original

theorem projects (v : Bool) (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool) :
    (source v Y X M ym yn xm xn sm K D log mode causal fp16).toAlgorithm? =
      .ok (source v Y X M ym yn xm xn sm K D log mode causal fp16).toAlgKernel := by
  cases v <;> rfl

/-- A proof-only prefix; typed_body below links it to the actual source.
The complete original suffix keeps every load, branch, cast and store. -/
def indexPrefix (v : Bool) (X : RegionName) (xm xn D : Nat) : List Stmt :=
  [.assign .nat [] "m" (.programId 0),
   .assign .nat [] "n" (.programId 1),
   .assign .nat [D] "k" (.arange D),
   .assign .ptr [D] "x_ptrs"
     (.ptrAdd .scalarL (.ptrBase X)
       (.add .nat .scalarL
         (.add .nat .nil
           (if v then .mul .nat .nil (.constNat xm) (.ref .nat [] "m")
            else .mul .nat .nil (.ref .nat [] "m") (.constNat xm))
           (.mul .nat .nil (.ref .nat [] "n") (.constNat xn)))
         (.ref .nat [D] "k")))]

theorem prefix_flat (A : FlatAlloc) (X : RegionName) (xm xn D : Nat)
    (s : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix Bool.true X xm xn D)) s =
      stepStmts (A.flattenStmts (indexPrefix Bool.false X xm xn D)) s := by
  simp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.ptrAdd, NumericDType.mul,
    Nat.mul_comm]

/-- Exact decomposition of the complete source, including the untouched
Option String control tree. No source flag is specialized for this lemma. -/
theorem typed_body (v : Bool) (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool) :
    (source v Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel.body =
      indexPrefix v X xm xn D ++
        (source Bool.false Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel.body.drop 4 := by
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases v
  all_goals
    simp only [source, indexPrefix, Bool.false_eq_true, ↓reduceIte]
    dsimp only [source, Typed.fp32Original, Typed.fp32Candidate, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, Except.pure, Bind.bind, Pure.pure, Kernel.eraseDType,
      ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
      Region.cast, indexPrefix, List.drop_succ_cons, List.drop_zero]
    rfl

/-- Whole flattened erased execution, for arbitrary initial flat state,
flags, mask option, allocation and dimensions; failure is preserved too. -/
theorem flat_exec (A : FlatAlloc) (v : Bool) (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (log : Bool) (mode : Option String)
    (causal fp16 : Bool) (s : BlockState) :
    exec (A.flattenKernel
      (source v Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel
      (source Bool.false Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel) s := by
  cases v
  · rfl
  · unfold exec FlatAlloc.flattenKernel
    conv_lhs => rw [typed_body Bool.true]
    conv_rhs => rw [typed_body Bool.false]
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat]

theorem erased_projects (v : Bool) (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool) :
    (source v Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgorithm? =
      .ok (source v Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, projects]
  simp

#axiomsClean projects
#axiomsClean prefix_flat
#axiomsClean typed_body
#axiomsClean flat_exec
#axiomsClean erased_projects
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Candidate
