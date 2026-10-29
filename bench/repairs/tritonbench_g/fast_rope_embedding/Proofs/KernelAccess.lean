import bench.repairs.tritonbench_g.fast_rope_embedding.Contract

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
open VeriTile.Bench.Repairs.RopeEmbedding
set_option maxHeartbeats 3000000

/-- Source integer expression, with the actual observed row stride. The width
can be 32 or 64: singleton tensors may pass an i64 stride to Triton. -/
def qIndex (width : Nat) (c : Call) (row head lane : Nat) : BitVec width :=
  BitVec.ofNat width row * BitVec.ofNat width c.qStride +
    BitVec.ofNat width head * BitVec.ofNat width c.geometry.dimension +
    BitVec.ofNat width lane + BitVec.ofNat width c.geometry.half

theorem actual_offset (c : Call) (admit : Admitted c) (row head lane : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads) (half : 0<c.geometry.half) :
    row*c.qStride+head*c.geometry.dimension+lane+c.geometry.half=
      c.geometry.offset row head (lane+c.geometry.half) := by
  have dimension : 0<c.geometry.dimension := by unfold Request.half at half; omega
  rw [active_stride c admit row hr (Nat.mul_pos (by omega) dimension)]
  simp [Request.offset,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

/-- Modular machine multiplication handles even huge unused singleton strides;
only the reached product must fit, not the stride argument itself. -/
theorem actual_bits (width : Nat) (c : Call) (admit : Admitted c) (row head lane : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads) (half : 0<c.geometry.half) :
    qIndex width c row head lane=BitVec.ofNat width (c.geometry.offset row head (lane+c.geometry.half)) := by
  unfold qIndex
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_add,←BitVec.ofNat_add,←BitVec.ofNat_add,
    actual_offset c admit row head lane hr hh half]

structure Access (c : Call) (row group : Nat) : Prop where
  groupBound : group*4<c.geometry.heads ∧ group*4+4<2^31
  offsets : ∀ head lane,head<c.geometry.heads → lane<c.geometry.block →
    row*c.qStride+head*c.geometry.dimension+lane+c.geometry.half<2^31 ∧
    row%c.geometry.sequence*c.cosineStride+lane<2^31 ∧
    row%c.geometry.sequence*c.sineStride+lane<2^31 ∧
    c.inputs.q.dataPtr.toNat+4*(row*c.qStride+head*c.geometry.dimension+lane+c.geometry.half)+4≤2^64 ∧
    c.inputs.cosine.dataPtr.toNat+4*(row%c.geometry.sequence*c.cosineStride+lane)+4≤2^64 ∧
    c.inputs.sine.dataPtr.toNat+4*(row%c.geometry.sequence*c.sineStride+lane)+4≤2^64 ∧
    (qIndex 32 c row head lane).toInt=(c.geometry.offset row head (lane+c.geometry.half) : Int) ∧
    (qIndex 64 c row head lane).toNat=c.geometry.offset row head (lane+c.geometry.half)

theorem access (c : Call) (admit : Admitted c) (row group : Nat)
    (launch : c.geometry.Launched row group) : Access c row group := by
  refine ⟨KernelAccess.group_bound _ admit.metadata group launch.2.2,?_⟩
  intro head lane hh hl
  obtain ⟨qo,co,qa,ca,sa⟩ := KernelAccess.checked_offsets (KernelBinding.metadata c.inputs c.geometry)
    admit.metadata row head lane launch.2.1 hh hl
  change c.geometry.offset row head (lane+c.geometry.half)<2^31 at qo
  simp only [KernelBinding.metadata] at co qa ca sa
  rw [actual_offset c admit row head lane launch.2.1 hh launch.1,
    admit.cosineStride launch.1,admit.sineStride launch.1,
    actual_bits 32 c admit row head lane launch.2.1 hh launch.1,
    actual_bits 64 c admit row head lane launch.2.1 hh launch.1]
  exact ⟨qo,co,co,qa,ca,sa,KernelAccess.signed32 _ qo,
    by simp only [BitVec.toNat_ofNat]; exact Nat.mod_eq_of_lt (by omega)⟩

#axiomsClean actual_offset
#axiomsClean actual_bits
#axiomsClean access
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
