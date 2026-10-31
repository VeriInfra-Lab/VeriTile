import bench.tritonbench_g.layer_norm_welfold.IO
import bench.tritonbench_g.layer_norm_welfold.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section

def code (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) : List ComputeStmt :=
  (SourceCode.rowKernel X W Bias Y Mean Rstd S D B (tree t)).surfaceBody

def phaseBody (phase : Bool) (X : RegionName) (S D B : Nat) (t : ReductionTree B) : List ComputeStmt :=
  let body : List ComputeStmt := code X "W" "Bias" "Y" "Mean" "Rstd" S D B t
  match body[if phase then 14 else 6]? with
  | some (ComputeStmt.forLoop _ _ body) => body
  | _ => []

def stats (X Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :=
  (code X "W" "Bias" "Y" Mean Rstd S D B t).take 23

def meanTail (Mean : RegionName) (S D B : Nat) (t : ReductionTree B) :=
  ((code "X" "W" "Bias" "Y" Mean "Rstd" S D B t).drop 8).take 5

def inverseTail (Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :=
  ((code "X" "W" "Bias" "Y" "Mean" Rstd S D B t).drop 16).take 7

def normalizeBody (reuse : Bool) (X W Bias Y : RegionName) (S D B : Nat)
    (t : ReductionTree B) : List ComputeStmt :=
  let body : List ComputeStmt :=
    (SourceCode.selected reuse X W Bias Y "Mean" "Rstd" S D B (tree t)).surfaceBody
  match body[23]? with
  | some (ComputeStmt.forLoop _ _ body) => body
  | _ => []

theorem sum_expression {B : Nat} (t : ReductionTree B) :
    DSL.WelfordSource.sumExpression (.ref .real [1,B] "acc") (tree t)=sumExpr t := by
  induction t with
  | leaf i => rfl
  | node l r hl hr => simp only [tree,SourceBinding.tree,DSL.WelfordSource.sumExpression,sumExpr,hl,hr]

theorem stats_split (X Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :
    stats X Mean Rstd S D B t =
      init S B ++ [.forLoop "k" (blockCount D B) (phaseBody Bool.false X S D B t)] ++
      sumCode t ++ meanTail Mean S D B t ++ reset B ++
      [.forLoop "k" (blockCount D B) (phaseBody Bool.true X S D B t)] ++
      sumCode t ++ inverseTail Rstd S D B t := by
  simp only [stats,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,List.take,
    List.drop,phaseBody,sumCode,←sum_expression]
  rfl

theorem source_split (reuse : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (t : ReductionTree B) :
    (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (tree t)).surfaceBody =
      stats X Mean Rstd S D B t ++
      [.forLoop "k" (blockCount D B) (normalizeBody reuse X W Bias Y S D B t)] := by
  cases reuse <;> rfl

#axiomsClean stats_split
#axiomsClean source_split
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
