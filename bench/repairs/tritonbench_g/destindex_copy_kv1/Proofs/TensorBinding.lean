import bench.repairs.tritonbench_g.destindex_copy_kv1.Contract
import VeriTile.Torch.FP32

/-! The metadata contract is connected to the shared Torch storage laws.
These are ordinary API bindings, not assumptions about a kernel result. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Binding
open VeriTile.Triton VeriTile.Torch.FP32
noncomputable section

/-- Direct FP32 reads at the checked actual pointer are logical Tensor values,
including storage-offset views. The absent negative bit is used explicitly. -/
theorem checked_read (m : Inputs) (w : Nat → DestindexCopyKv1.Word)
    (h : InputContract m w) (r : Role) (t : Tensor) (laws : TensorLaws t)
    (metadata : t.metadata = m r) (read : Nat → VeriTile.Torch.FP32.Word) (i : Nat)
    (hi : i < (m r).numel) :
    read (addr (base m r) i) = logicalValue read t laws i := by
  have hc : t.metadata.isContiguous = Bool.true := by simpa only [metadata] using h.contiguous r
  have hn : t.metadata.isNeg = Bool.false := by simpa only [metadata] using h.noNeg r
  have hit : i < t.metadata.numel := by simpa only [metadata] using hi
  have ne : 0 < t.metadata.numel := by omega
  let v := dataView t laws hc
  have cap := v.spanFits
  have bound : v.base < 2^64 := by
    change v.base + 4 * t.metadata.numel ≤ 2^64 at cap
    omega
  have ptr := congrArg BitVec.toNat (actual_pointer t laws hc ne)
  have ptr' : v.base % 2^64 = base m r := by
    simpa only [DataView.address, Nat.mul_zero, Nat.add_zero,
      BitVec.toNat_ofNat, Tensor.dataPtr, metadata, base] using ptr
  have vb : v.base = base m r := by
    simpa only [Nat.mod_eq_of_lt bound] using ptr'
  have same : addr (base m r) i = v.address i := by
    simp only [addr, DataView.address, vb]
  rw [same]
  exact direct_read_eq_logical read t laws hc hn i hit

#axiomsClean checked_read
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Binding
