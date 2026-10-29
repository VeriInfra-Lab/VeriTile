import bench.repairs.tritonbench_g.layer_norm_welfold.Kernels
import VeriTile.Meta.StatementAudit

/-! Proof-only decomposition of the exact Python quotations. The authoritative
programs remain the complete bodies in Kernels.lean. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Raw
open VeriTile.Triton
noncomputable section

def code (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) : List ComputeStmt :=
  (SourceCode.rowKernel X W Bias Y Mean Rstd S D B tree).surfaceBody

def prefix (S D B : Nat) (tree : DSL.WelfordSource.Tree B) :=
  (code "X" "W" "Bias" "Y" "Mean" "Rstd" S D B tree).take 6

def phaseBody (variance : Bool) (X : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) : List ComputeStmt :=
  match (code X "W" "Bias" "Y" "Mean" "Rstd" S D B tree)[if variance then 13 else 6]? with
  | some (.forRange _ _ _ _ body) => body
  | _ => []

def sumCode (variance : Bool) (S D B : Nat) (tree : DSL.WelfordSource.Tree B) :=
  ((code "X" "W" "Bias" "Y" "Mean" "Rstd" S D B tree).drop
    (if variance then 14 else 7)).take 1

def meanTail (Mean : RegionName) (S D B : Nat) (tree : DSL.WelfordSource.Tree B) :=
  ((code "X" "W" "Bias" "Y" Mean "Rstd" S D B tree).drop 8).take 4

def reset (S D B : Nat) (tree : DSL.WelfordSource.Tree B) :=
  ((code "X" "W" "Bias" "Y" "Mean" "Rstd" S D B tree).drop 12).take 1

def inverseTail (Rstd : RegionName) (S D B : Nat) (tree : DSL.WelfordSource.Tree B) :=
  ((code "X" "W" "Bias" "Y" "Mean" Rstd S D B tree).drop 15).take 7

def stats (X Mean Rstd : RegionName) (S D B : Nat) (tree : DSL.WelfordSource.Tree B) :=
  (code X "W" "Bias" "Y" Mean Rstd S D B tree).take 22

def normalizeBody (reuse : Bool) (X W Bias Y : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) : List ComputeStmt :=
  match (SourceCode.selected reuse X W Bias Y "Mean" "Rstd" S D B tree).surfaceBody[22]? with
  | some (.forRange _ _ _ _ body) => body
  | _ => []

theorem stats_split (X Mean Rstd : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) :
    stats X Mean Rstd S D B tree =
      prefix S D B tree ++ [.forRange "roffset" 0 D B (phaseBody Bool.false X S D B tree)] ++
      sumCode Bool.false S D B tree ++ meanTail Mean S D B tree ++ reset S D B tree ++
      [.forRange "roffset" 0 D B (phaseBody Bool.true X S D B tree)] ++
      sumCode Bool.true S D B tree ++ inverseTail Rstd S D B tree := by rfl

theorem source_split (reuse : Bool) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : DSL.WelfordSource.Tree B) :
    (SourceCode.selected reuse X W Bias Y Mean Rstd S D B tree).surfaceBody =
      stats X Mean Rstd S D B tree ++
      [.forRange "roffset" 0 D B (normalizeBody reuse X W Bias Y S D B tree)] := by
  cases reuse <;> rfl

#axiomsClean stats_split
#axiomsClean source_split
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Raw
