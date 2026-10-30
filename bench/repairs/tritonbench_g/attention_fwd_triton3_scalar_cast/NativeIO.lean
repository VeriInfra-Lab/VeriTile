import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.RealIO

/-! Native IO keeps the original six arbitrary decoded input arrays and a
finite, source-free access footprint. Every FP32 storage/operation/cast stays
in the selected quotation. The raw FP32 scale slot is bound by the existing
ScaleBinding interpretation, including signed zero and NaN payloads.
M/L/O inputs alias the corresponding outputs. M/O include padding rows;
L is written only for continuation. There is no private scratch buffer.
-/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership
noncomputable section
attribute [local instance] Classical.propDecidable

def arrayRead (v : Inputs) (rg : RegionName) (off : Nat) : Native.Expr :=
  if rg="Q" then arrayValues v 0 off else
  if rg="K" then arrayValues v 1 off else
  if rg="V" then arrayValues v 2 off else
  if rg="M" then arrayValues v 3 off else
  if rg="L" then arrayValues v 4 off else
  if rg="O" then arrayValues v 5 off else default

/-- Pure array packaging for the inherited source-independent online
recurrence. No executor or caller state is a mathematical input. -/
def arrayState (v : Inputs) : State Native.Expr where
  mem := fun rg off => .mk .real (arrayRead v rg off)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def expressionAt (c : Call) (v : Inputs) (o : Fin 3) (j : RealInterface.OutputPoint c o) : Native.Expr :=
  ((Reference.gridResult (ScaleBinding.template c) Native.backend (arrayState v)).mem
    (RealInterface.outputRegion o) (RealInterface.outputAddress c o j)).read .real

def expressionFormula (c : Call) (v : Inputs) : ChannelValues Native.Expr (outputPorts c) := by
  intro o j
  exact cast (congrArg (Value Native.Expr) (congrArg IOChannel.dtype (output_port c o))).symm
    (expressionAt c v o (outputPoint c o j))

def outputLane (c : Call) : (o : Fin 3) → TileIndex (RealInterface.outputShape c o) → Lane c.BM c.D
  | 0 => fun j => .center j.1
  | 1 => fun j => .denominator j.1
  | 2 => fun j => .output j.1 j.2.1

/-- Owner-specific word observation, including bytewise M/L publication. -/
def observeExpression (c : Call) (shape : Native.Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (scale : Native.Word) (o : Fin 3) (j : RealInterface.OutputPoint c o)
    (value : Native.Expr) : Native.Word :=
  Native.projectLane c shape schedule ops scale (fun _ _ => .mk .real value)
    j.1 (outputLane c o j.2)

def observer (c : Call) (shape : Native.Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (scale : Native.Word) (o : Fin (outputPorts c).length)
    (j : TileIndex ((outputPorts c)[o]).shape) : Native.Expr → Native.Word :=
  observeExpression c shape schedule ops scale o (outputPoint c o j)

/-- Independent online attention recurrence under the existing scalar and
fixed-K8 instruction ports. This is native observation, not equality to Real. -/
def formula (c : Call) (shape : Native.Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (scale : Native.Word) (v : Inputs) :
    ChannelValues Native.Word (outputPorts c) :=
  fun o j => cast (congrArg (Value Native.Word) (congrArg IOChannel.dtype (output_port c o))).symm
    (observeExpression c shape schedule ops scale o (outputPoint c o j)
      (expressionAt c v o (outputPoint c o j)))

#specNonCircular formula avoiding
  [Kernels.source,FP.Structural.exec,FP.Structural.run,runSchedule,Reference.complete_grid]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface
