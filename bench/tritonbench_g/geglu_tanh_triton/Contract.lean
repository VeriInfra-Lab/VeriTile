import bench.tritonbench_g.geglu_tanh_triton.Proofs.Physical
import bench.tritonbench_g.swiglu_triton.Contract

namespace VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost
open Physical
abbrev Tensor := VeriTile.Bench.Repairs.SwigluTriton.DirectHost.Tensor
abbrev Prepared := VeriTile.Bench.Repairs.SwigluTriton.DirectHost.Prepared
abbrev payload := VeriTile.Bench.Repairs.SwigluTriton.DirectHost.payload
abbrev spanSeparate := VeriTile.Bench.Repairs.SwigluTriton.DirectHost.spanSeparate

structure Admitted (p : Prepared backward) : Prop where
  payloads : ∀ r, payload (p.tensor r) = Bool.true
  counts : ∀ r, (p.tensor r).metadata.numel = p.rows*p.cols
  forwardShape : backward = Bool.false → (p.tensor .b).metadata.shape = p.originalShape
  sameDevice : ∀ r, (p.tensor r).metadata.device = (p.tensor .a).metadata.device
  positive : 0 < p.cols
  grid32 : (p.rows*p.cols+127)/128 < 2^31
  padded64 : p.rows*p.cols+127 < 2^63
  alignment : 0 < p.rows*p.cols → ∀ r, (p.tensor r).metadata.dataPtr.toNat % 4 = 0
  separate : 0 < p.rows*p.cols → ∀ r s, r ≠ s → spanSeparate (p.tensor r) (p.tensor s)

def check (p : Prepared backward) : Bool :=
  decide (∀ r, payload (p.tensor r) = Bool.true) &&
  decide (∀ r, (p.tensor r).metadata.numel = p.rows*p.cols) &&
  decide (backward = Bool.false → (p.tensor .b).metadata.shape = p.originalShape) &&
  decide (∀ r, (p.tensor r).metadata.device = (p.tensor .a).metadata.device) &&
  decide (0 < p.cols) && decide ((p.rows*p.cols+127)/128 < 2^31) && decide (p.rows*p.cols+127 < 2^63) &&
  decide (0 < p.rows*p.cols → ∀ r, (p.tensor r).metadata.dataPtr.toNat % 4 = 0) &&
  decide (0 < p.rows*p.cols → ∀ r s, r ≠ s → spanSeparate (p.tensor r) (p.tensor s))

end VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost
