import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardModel
import VeriTile.Meta.StatementAudit

/-! Independent source-ordered scalar references. All reads are from the
initial input snapshot. Partial gradients are increasing-row scalar folds;
these definitions contain no execution or output-memory construction. -/
namespace VeriTile.Bench.Repairs.LayerNormOps.KernelReference
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]

def forwardY (A : Algebra α) (p : ForwardParameters) (s : State α)
    (i : TileIndex [p.block]) : α :=
  let base := if p.rms then Forward.z A p s i else
    Forward.unhintedBinary A .sub (Forward.z A p s i) (Forward.meanValue A p s)
  let h := Forward.unhintedBinary A .mul base (Forward.rstdValue A p s)
  let y := Forward.unhintedBinary A .mul h (Forward.unspecified A p s p.weight i)
  if p.hasBias then Forward.unhintedBinary A .add y (Forward.unspecified A p s p.bias i) else y

def rowDX (A : Algebra α) (p : BackwardParameters) (s : State α)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  let base := Backward.dxBase A p (Backward.initial A p s) row i
  if p.hasDres then Forward.binary A .add base (Backward.loaded A p s p.dres (row*p.dresStride) i)
  else base

def rowY (A : Algebra α) (p : BackwardParameters) (s : State α)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  Backward.yValue A p (Backward.initial A p s) row i

def rowDW (A : Algebra α) (p : BackwardParameters) (s : State α)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  Forward.binary A .mul (Backward.dy A p (Backward.initial A p s) row i)
    (Backward.xhat A p (Backward.initial A p s) row i)

def rowDB (A : Algebra α) (p : BackwardParameters) (s : State α)
    (row : Nat) (i : TileIndex [p.block]) : α :=
  Backward.dy A p (Backward.initial A p s) row i

abbrev Accum (α : Type) (block : Nat) :=
  (TileIndex [block] → α) × (TileIndex [block] → α)

def addRow (A : Algebra α) (p : BackwardParameters) (s : State α)
    (row : Nat) (a : Accum α p.block) : Accum α p.block :=
  (fun i => Forward.binary A .add (a.1 i) (rowDW A p s row i),
    if p.hasBias then fun i => Forward.binary A .add (a.2 i) (rowDB A p s row i) else a.2)

def partials (A : Algebra α) (p : BackwardParameters) (s : State α) : Accum α p.block :=
  foldRange (addRow A p s) (Backward.start p s) (Backward.stop p s) 1
    (fun _ => Forward.literal A 0,fun _ => Forward.literal A 0)

#specNonCircular forwardY avoiding [forwardKernel,FP.Structural.exec,Forward.reference]
#specNonCircular rowDX avoiding [backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular rowY avoiding [backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular partials avoiding [backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelReference
