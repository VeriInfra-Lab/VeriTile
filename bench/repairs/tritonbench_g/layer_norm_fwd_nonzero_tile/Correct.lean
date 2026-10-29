import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

open RealInterface
open scoped VeriTile.Triton.LogicalIO

/- Pure logical formulas for row scaling, canonical Philox dropout, residuals,
population/RMS normalization and both affine outputs. -/
namespace Mathematics

/-- Exact unsigned cast of the loaded int64 seed. This explicitly retains
the source narrowing cast; replacing the pointer ABI by uint32 is invalid. -/
def seedU32 (seed : Int) : Int := seed % (2^32 : Int)

def keep (seed : Int) (p : ℝ) (col : Nat) : Bool :=
  decide (Random.randReal (seedU32 seed) (col : Int) .bits32 7 > p)

def dropout (seed : Int) (p value : ℝ) (col : Nat) : ℝ :=
  if keep seed p col then value / (1-p) else 0

def streamValue (f : Flags) (a : Inputs) (M : Nat) (p : ℝ)
    (second : Bool) (row col : Nat) : ℝ :=
  let input := if second then a.x1 row col else a.x row col
  let streamRow := if second then M + row else row
  let scaled := if f.rowscale then input * a.rowscale streamRow else input
  if f.dropout then dropout (a.seeds streamRow) p scaled col else scaled

/-- Complete residual-input value after row scaling, dropout, X1 and residual.
The order of the additions matches the source. -/
def combined (f : Flags) (a : Inputs) (M : Nat) (p : ℝ) (row col : Nat) : ℝ :=
  let main := streamValue f a M p Bool.false row col
  let twoInputs := if f.x1 then main + streamValue f a M p Bool.true row col else main
  if f.residual then twoInputs + a.residual row col else twoInputs

def mean (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ) (row : Nat) : ℝ :=
  (∑ col ∈ Finset.range N, combined f a M p row col) / N

def centered (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ) (row col : Nat) : ℝ :=
  if f.rms then combined f a M p row col
  else combined f a M p row col - mean f a M N p row

def variance (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ) (row : Nat) : ℝ :=
  (∑ col ∈ Finset.range N, (centered f a M N p row col)^2) / N

def rstd (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row : Nat) : ℝ :=
  1 / Real.sqrt (variance f a M N p row + eps)

def normalized (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ)
    (row col : Nat) : ℝ :=
  centered f a M N p row col * rstd f a M N eps p row

def y (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row col : Nat) : ℝ :=
  let weighted := normalized f a M N eps p row col * a.weight col
  if f.bias then weighted + a.bias col else weighted

def y1 (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row col : Nat) : ℝ :=
  let weighted := normalized f a M N eps p row col * a.weight1 col
  if f.b1 then weighted + a.bias1 col else weighted

end Mathematics

def formula (f : Flags) (M N : Nat) (eps p : ℝ) (a : LogicalIO.Inputs M N) :
    LogicalIO.Outputs M N :=
  let x := LogicalIO.mathInputs M N a
  ⟨fun row col => Mathematics.y f x M N eps p row.val col.val,
   fun row col => Mathematics.y1 f x M N eps p row.val col.val,
   fun row => Mathematics.mean f x M N p row.val,
   fun row => Mathematics.rstd f x M N eps p row.val,
   fun row col => Mathematics.combined f x M p row.val col.val,
   fun row col => Mathematics.keep (x.seeds row.val) p col.val,
   fun row col => Mathematics.keep (x.seeds (M+row.val)) p col.val⟩

/-- The complete ordinary grid implements the independent layer-normalization
formula. The shared IO judgment includes successful source execution, actual
access safety, all enabled outputs and the exact exterior memory frame. -/
specification correct (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (width : widthCheck N=Bool.true)
    (sep : Domain.ReadWriteSeparated f r) (geometry : NativeGeometry.Geometry f r stride M N) :
    Spec.Real (LogicalIO.io f r stride M N eps p ⊨ formula f M N eps p) :=
  LogicalIO.implements f r stride M N eps p width sep geometry

#axiomsClean correct
#specNonCircular formula avoiding [VeriTile.Triton.exec,quoted]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
