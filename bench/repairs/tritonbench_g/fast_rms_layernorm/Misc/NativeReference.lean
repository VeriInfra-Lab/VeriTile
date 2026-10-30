import bench.repairs.tritonbench_g.fast_rms_layernorm.IO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Reference
open VeriTile Triton FP.Structural NativeGraph
noncomputable section

/-- Pure word observation on the named finite output ports. -/
def observer (h : Call) (backward : Bool) (I : Instructions) (S : Selection) :
    (o : Fin (Ports.outputs h backward).length) → TileIndex (Ports.outputs h backward)[o].shape → Term → Word :=
  fun _ _=>eval I S

/-- Native words are a pure observation of the independent expression formula;
all byte coordinates follow through the existing little-endian codec. -/
def nativeWords (R : Reductions) (S : Selection) (I : Instructions) (h : Call)
    (backward : Bool) (epsilon : Term) (vals : ChannelValues Term (Ports.inputs h)) :
    ChannelValues Word (Ports.outputs h backward) :=
  fun o j=>mapValue (eval I S) (Ports.outputs h backward)[o].dtype (native R h backward epsilon vals o j)

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Reference
