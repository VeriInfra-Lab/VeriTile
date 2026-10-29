import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Tensor
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.RealInputs
import bench.repairs.tritonbench_g.lightning_attention.Proofs.Geometry
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def initialReal {phase} (r : Request phase) (read : Nat → ℝ) : Memory ℝ :=
  fun rg off => .mk .real (read (base r rg+4*off))

def tensorQ {phase} (r : Request phase) (read : Nat → ℝ) (head : Fin (r.dimensions.B*r.dimensions.H))
    (time : Fin r.dimensions.N) (feature : Fin r.dimensions.D) : ℝ :=
  logical read r.q (head.val*r.dimensions.N*r.dimensions.D+time.val*r.dimensions.D+feature.val)
def tensorK {phase} (r : Request phase) (read : Nat → ℝ) (head : Fin (r.dimensions.B*r.dimensions.H))
    (time : Fin r.dimensions.N) (feature : Fin r.dimensions.D) : ℝ :=
  logical read r.k (head.val*r.dimensions.N*r.dimensions.D+time.val*r.dimensions.D+feature.val)
def tensorV {phase} (r : Request phase) (read : Nat → ℝ) (head : Fin (r.dimensions.B*r.dimensions.H))
    (time : Fin r.dimensions.N) (feature : Fin r.dimensions.E) : ℝ :=
  logical read r.v (head.val*r.dimensions.N*r.dimensions.E+time.val*r.dimensions.E+feature.val)
def tensorG (r : Request .backward) (read : Nat → ℝ) (head : Fin (r.dimensions.B*r.dimensions.H))
    (time : Fin r.dimensions.N) (feature : Fin r.dimensions.E) : ℝ :=
  logical read r.incoming.original (head.val*r.dimensions.N*r.dimensions.E+time.val*r.dimensions.E+feature.val)

theorem logical_bound (d : Host.Dimensions) (w : Nat) (head : Fin (d.B*d.H))
    (time : Fin d.N) (feature : Fin w) :
    head.val*d.N*w+time.val*w+feature.val<d.B*d.H*d.N*w := by
  simpa [LightningAttention.Geometry.dense,Nat.add_mul,Nat.add_assoc]
    using LightningAttention.Geometry.dense_bound head.isLt time.isLt feature.isLt

theorem base_result {phase} (r : Request phase) (j : Fin (outputs phase)) :
    base r (resultRegion j)=(r.result j).tensor.dataPtr.toNat := by
  cases phase <;> fin_cases j <;> rfl

theorem input_Q {phase} (r : Request phase) (h : Contract r) (read : Nat → ℝ)
    (head : Fin (r.dimensions.B*r.dimensions.H)) :
    Real.logicalQ r.dimensions.parameters head.val (initialReal r read)=tensorQ r read head := by
  funext t a
  have bound : head.val*r.dimensions.N*r.dimensions.D+t.val*r.dimensions.D+a.val<r.q.tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ h.q]
    exact logical_bound _ _ head t a
  rw [tensorQ,logical_direct read r.q h.q.1 (h.q.2.2.2.1 rfl) _ bound]
  simp [Real.logicalQ,Real.query,initialReal,base,argument,Host.Dimensions.parameters]

theorem input_K {phase} (r : Request phase) (h : Contract r) (read : Nat → ℝ)
    (head : Fin (r.dimensions.B*r.dimensions.H)) :
    Real.logicalK r.dimensions.parameters head.val (initialReal r read)=tensorK r read head := by
  funext t a
  have bound : head.val*r.dimensions.N*r.dimensions.D+t.val*r.dimensions.D+a.val<r.k.tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ h.k]
    exact logical_bound _ _ head t a
  rw [tensorK,logical_direct read r.k h.k.1 (h.k.2.2.2.1 rfl) _ bound]
  simp [Real.logicalK,Real.key,initialReal,base,argument,Host.Dimensions.parameters]

theorem input_V {phase} (r : Request phase) (h : Contract r) (read : Nat → ℝ)
    (head : Fin (r.dimensions.B*r.dimensions.H)) :
    Real.logicalV r.dimensions.parameters head.val (initialReal r read)=tensorV r read head := by
  funext t a
  have bound : head.val*r.dimensions.N*r.dimensions.E+t.val*r.dimensions.E+a.val<r.v.tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ h.v]
    exact logical_bound _ _ head t a
  rw [tensorV,logical_direct read r.v h.v.1 (h.v.2.2.2.1 rfl) _ bound]
  simp [Real.logicalV,Real.value,initialReal,base,argument,Host.Dimensions.parameters]

theorem input_G (r : Request .backward) (h : Contract r) (read : Nat → ℝ) (prep : Prepared r read)
    (head : Fin (r.dimensions.B*r.dimensions.H)) :
    Real.logicalG r.dimensions.parameters head.val (initialReal r read)=tensorG r read head := by
  funext t a
  have hd := h.incoming.2
  have ho := h.incoming.1
  let i := head.val*r.dimensions.N*r.dimensions.E+t.val*r.dimensions.E+a.val
  have bi : i<r.incoming.dense.tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ hd]
    exact logical_bound _ _ head t a
  have bo : i<r.incoming.original.tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ ho]
    exact logical_bound _ _ head t a
  change _=logical read r.incoming.original i
  rw [←prep i bo,logical_direct read r.incoming.dense hd.1 (hd.2.2.2.1 rfl) _ bi]
  simp [Real.logicalG,Real.incoming,initialReal,base,argument,inputG,Host.Dimensions.parameters,i]

#axiomsClean input_Q
#axiomsClean input_K
#axiomsClean input_V
#axiomsClean input_G
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
