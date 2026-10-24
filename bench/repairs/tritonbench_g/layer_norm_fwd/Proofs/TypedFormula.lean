import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedValues
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedStreams

/-! Input-only ordered FP source target. Arithmetic remains opaque, and
padding is evaluated exactly: no zero identities or finite-value laws are
used. This is the source reduction callback, not an invented native tree. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section

def lit {α : Type} (A : Algebra α) (x : ℝ) : α :=
  (engine A).literal (some .fp32) .real x
def op {α : Type} (A : Algebra α) (b : Binary) (x y : α) : α :=
  (engine A).binary (some .fp32) .real b x y
def defaultOp {α : Type} (A : Algebra α) (b : Binary) (x y : α) : α :=
  (engine A).binary none .real b x y
def cast {α : Type} (A : Algebra α) (x : α) : α :=
  (engine A).cast (some .fp32) .real .real x
def scalarLoad {α : Type} [Inhabited α] (A : Algebra α) (s : State α)
    (region : RegionName) (offset : Nat) : α := cast A ((s.mem region offset).read .real)
def randomValue {α : Type} (A : Algebra α) (seed : Int) (col : Nat) : α :=
  (engine A).fp32Bits ⟨(Random.rand ((seed % (2^32 : Int)).toNat : Int)
    (col : Int) .bits32 7).bits⟩
def keepValue {α : Type} (A : Algebra α) (cmp : Comparisons A)
    (seed : Int) (p : ℝ) (col : Nat) : Bool :=
  cmp.lt none .real ((engine A).literal none .real p) (randomValue A seed col)
def streamValues {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (s : State α) (r : Regions) (index : Nat) (p : ℝ)
    (v : Values α .real [block]) : Values α .real [block] := fun j =>
  let scaled := if f.rowscale then op A .mul (v j) (scalarLoad A s r.rowscale index) else v j
  if f.dropout then
    if keepValue A cmp ((s.mem r.seeds index).read .int) p j.1.val
    then op A .div scaled (op A .sub (lit A 1) (lit A p)) else lit A 0
  else scaled

def combinedValues {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (s : State α) (r : Regions) (stride : Strides) (M N : Nat)
    (p : ℝ) (row block : Nat) : Values α .real [block] :=
  let first := streamValues A cmp f s r row p (fun j => raw A s r.x (row*stride.x) N j.1.val)
  let second := streamValues A cmp f s r (M+row) p (fun j => raw A s r.x1 (row*stride.x1) N j.1.val)
  let both := fun j => if f.x1 then op A .add (first j) (second j) else first j
  fun j => if f.residual then op A .add (both j) (raw A s r.residual (row*stride.residual) N j.1.val) else both j

def meanValue {α : Type} (A : Algebra α) (N : Nat) (v : Values α .real [block]) : α :=
  op A .div ((engine A).reduceSum (some .fp32) 0 Bool.false v PUnit.unit) (lit A N)
def centeredValues {α : Type} (A : Algebra α) (rms : Bool) (N : Nat)
    (v : Values α .real [block]) : Values α .real [block] := fun j =>
  if j.1.val<N then if rms then v j else op A .sub (v j) (meanValue A N v) else lit A 0
def varianceValue {α : Type} (A : Algebra α) (rms : Bool) (N : Nat)
    (v : Values α .real [block]) : α :=
  meanValue A N (fun j => op A .mul (centeredValues A rms N v j) (centeredValues A rms N v j))
def rstdValue {α : Type} (A : Algebra α) (rms : Bool) (N : Nat) (eps : ℝ)
    (v : Values α .real [block]) : α :=
  op A .div (lit A 1) ((engine A).unary (some .fp32) .sqrt
    (op A .add (varianceValue A rms N v) (lit A eps)))
def normalizedValues {α : Type} (A : Algebra α) (rms : Bool) (N : Nat) (eps : ℝ)
    (v : Values α .real [block]) : Values α .real [block] := fun j =>
  defaultOp A .mul (if rms then v j else defaultOp A .sub (v j) (meanValue A N v)) (rstdValue A rms N eps v)
def affineValue {α : Type} (A : Algebra α) (bias : Bool) (x w b : α) : α :=
  cast A (if bias then defaultOp A .add (defaultOp A .mul x w) b else defaultOp A .mul x w)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
