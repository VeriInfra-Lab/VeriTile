import bench.repairs.tritonbench_g.decay_cumsum.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.DecayCumsum
open VeriTile.Triton VeriTile.Torch

def argumentRoles : Kind → List RegionName
  | .forward => ["g","go"]
  | .prepare => ["q","k","g","qg","kg"]
  | .backward => ["dqi","dqo","dki","dko","q","k","g","dg"]

def firstRole : Kind → RegionName
  | .forward => "g" | .prepare => "q" | .backward => "dqi"

structure TensorContract (h : Call) (device : Device) (x : TensorMetadata) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  shape : x.shape=[h.B,h.H,h.T,h.DK]
  cuda : x.device.kind="cuda"
  device : x.device=device
  contiguous : x.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false
  alignment : 0<x.numel → x.dataPtr.toNat%4=0

def tensorCheck (h : Call) (device : Device) (x : TensorMetadata) : Bool :=
  x.layout==.strided && x.dtype==.floating .fp32 && x.shape==[h.B,h.H,h.T,h.DK] &&
  x.isCuda && x.device==device && x.isContiguous && !x.isNeg &&
  (!(x.numel>0) || x.dataPtr.toNat%4==0)

structure InputContract (h : Call) (kind : Kind) (observed : RegionName → TensorMetadata)
    (scaleIsFloat : Bool) : Prop where
  blockPower : h.BK &&& (h.BK-1)=0
  blockLimit : h.BK≤1048576
  scaleType : scaleIsFloat=Bool.true
  tensors : ∀ r∈argumentRoles kind, TensorContract h (observed (firstRole kind)).device (observed r)
  grid : 0<h.B*h.H*h.T*h.DK → h.features<2^31 ∧ h.T/h.BT≤65535 ∧ h.B*h.H≤65535

inductive InputError where
  | tensor | integer | dimensions | block | scale | metadata | grid
  deriving DecidableEq, Repr

def checkInput (h : Call) (kind : Kind) (observed : RegionName → TensorMetadata)
    (scaleIsFloat : Bool) : Except InputError Unit :=
  if h.BK &&& (h.BK-1) != 0 || h.BK>1048576 then .error .block
  else if !scaleIsFloat then .error .scale
  else if !(argumentRoles kind).all (fun r=>tensorCheck h (observed (firstRole kind)).device (observed r)) then .error .metadata
  else if h.B*h.H*h.T*h.DK>0 && (h.features>=2^31 || h.T/h.BT>65535 || h.B*h.H>65535) then .error .grid
  else .ok ()

/-- Type classification happens before querying Tensor metadata. Dimensions are
validated by the source's signed-integer adapter before constructing Call. -/
def classifyTensor : ArgumentMetadata → Except InputError TensorMetadata
  | .nonTensor => .error .tensor
  | .tensor t => .ok t
end VeriTile.Bench.Repairs.DecayCumsum
