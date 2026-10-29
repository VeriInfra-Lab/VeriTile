import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalResult
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalHost
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.PublicFrame

namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

/-- Pure input-snapshot functions. The forward producer and every backward
formula below refer to this same pre-call state, never to a result buffer. -/
def z (fp : ForwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  (s.mem fp.x (row*fp.xStride+col)).read .fp32 +
    if fp.hasResidual then (s.mem fp.residual (row*fp.residualStride+col)).read .fp32 else 0

def weight (fp : ForwardParameters) (s : State ℝ) (col : Nat) : ℝ := (s.mem fp.weight col).read .fp32
def bias (fp : ForwardParameters) (s : State ℝ) (col : Nat) : ℝ := (s.mem fp.bias col).read .fp32

def mu (fp : ForwardParameters) (s : State ℝ) (row : Nat) : ℝ :=
  Logical.mean fp.columns fp.rms (z fp s row)
def inv (fp : ForwardParameters) (s : State ℝ) (row : Nat) : ℝ :=
  Logical.rstd fp.columns fp.rms fp.epsilon (z fp s row)
def y (fp : ForwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  Logical.y fp.columns fp.rms fp.hasBias fp.epsilon (z fp s row) (weight fp s) (bias fp s) col

def dy (bp : BackwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  (s.mem bp.dy (row*bp.dyStride+col)).read .fp32
def dres (bp : BackwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  (s.mem bp.dres (row*bp.dresStride+col)).read .fp32

def dx (fp : ForwardParameters) (bp : BackwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  Logical.dx fp.columns fp.rms bp.hasDres (z fp s row) (weight fp s)
    (dy bp s row) (dres bp s row) (mu fp s row) (inv fp s row) col

def dw (fp : ForwardParameters) (bp : BackwardParameters) (s : State ℝ) (rows col : Nat) : ℝ :=
  ∑ row ∈ Finset.range rows, dy bp s row col * Logical.normalized fp.columns fp.rms fp.epsilon (z fp s row) col

def db (bp : BackwardParameters) (s : State ℝ) (rows col : Nat) : ℝ :=
  ∑ row ∈ Finset.range rows, dy bp s row col

/-- Names for absent optional pointers denote distinct inert regions. This is
an ordinary symbolic Tensor/allocator binding: inputs may alias each other,
while fresh outputs and inert names do not alias any live read. No cell values
or execution equations are assumptions. -/
structure RegionBinding (fp : ForwardParameters) (bp : BackwardParameters)
    (dw db : RegionName) : Prop where
  forwardDistinct : [fp.y,fp.mean,fp.rstd,fp.residualOut].Nodup
  backwardDistinct : [bp.dx,bp.dw,bp.db,bp.y,bp.dresIn,dw,db].Nodup
  forwardFresh : ∀ r ∈ [fp.x,fp.residual,fp.weight,fp.bias,bp.dy,bp.dres],
    Host.ForwardRegionOutside fp r
  backwardFresh : ∀ r ∈ [bp.x,bp.weight,bp.bias,bp.dy,bp.dres,bp.mean,bp.rstd,
      fp.y,fp.mean,fp.rstd,fp.residualOut,fp.x,fp.residual],
    Backward.RegionOutside bp r ∧ r≠dw ∧ r≠db

#specNonCircular y avoiding [forwardKernel, backwardKernel, Host.composed]
#specNonCircular dx avoiding [forwardKernel, backwardKernel, Host.composed]
#specNonCircular dw avoiding [forwardKernel, backwardKernel, Host.composed]
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
