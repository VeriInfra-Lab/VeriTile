import bench.tritonbench_g.layer_norm_welfold.IO

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section

def code (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) : List ComputeStmt :=
  (SourceCode.rowKernel X W Bias Y Mean Rstd S D B (tree t)).surfaceBody

def phaseBody (phase : Bool) (X : RegionName) (_S D B : Nat) (_t : ReductionTree B) : List ComputeStmt :=
  phaseCode phase X D B

def stats (X Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :=
  (code X "W" "Bias" "Y" Mean Rstd S D B t).take 22

def meanTail (Mean : RegionName) (_S D B : Nat) (_t : ReductionTree B) := meanCode Mean D

def inverseTail (Rstd : RegionName) (_S D B : Nat) (_t : ReductionTree B) := inverseCode Rstd D

def normalizeBody (reuse : Bool) (X W Bias Y : RegionName) (S D B : Nat)
    (_t : ReductionTree B) : List ComputeStmt :=
  if reuse then
    source_fragment (fun (t : SourceCode.ReductionTree B) =>
      match (SourceCode.rowOffsetReusedKernel X W Bias Y "Mean" "Rstd" S D B t).surfaceBody.drop 22 with
      | .forRange _ _ _ _ body :: _ => body
      | _ => [])
  else
    source_fragment (fun (t : SourceCode.ReductionTree B) =>
      match (SourceCode.rowKernel X W Bias Y "Mean" "Rstd" S D B t).surfaceBody.drop 22 with
      | .forRange _ _ _ _ body :: _ => body
      | _ => [])

theorem sum_expression {B : Nat} (phase : Bool) (t : ReductionTree B) :
    DSL.WelfordSource.sumExpression (.ref .real [1,B] (if phase then "_tmp12" else "_tmp3"))
      (tree t)=sumExpr phase t := by
  induction t with
  | leaf i => rfl
  | node l r hl hr => simp only [tree,SourceBinding.tree,DSL.WelfordSource.sumExpression,sumExpr,hl,hr]

theorem stats_split (X Mean Rstd : RegionName) (S D B : Nat) (t : ReductionTree B) :
    stats X Mean Rstd S D B t =
      init S B ++ [.forRange "roffset" 0 D B (phaseBody Bool.false X S D B t)] ++
      sumCode Bool.false t ++ meanTail Mean S D B t ++ reset B ++
      [.forRange "roffset" 0 D B (phaseBody Bool.true X S D B t)] ++
      sumCode Bool.true t ++ inverseTail Rstd S D B t := by
  rfl

theorem source_split (reuse : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (t : ReductionTree B) :
    (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (tree t)).surfaceBody =
      stats X Mean Rstd S D B t ++
      [.forRange "roffset" 0 D B (normalizeBody reuse X W Bias Y S D B t)] := by
  cases reuse <;> rfl

#axiomsClean stats_split
#axiomsClean source_split
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
