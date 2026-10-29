import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.NativeControl
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Snapshot

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.NativeGrid
open Storage NativeIndex

/-- Local ABI facts about an actually admitted compiled launch. These are not
new wrapper shape checks. Global launch-product admission remains shared. -/
structure ABI (h : Call) (schedule : Snapshot.Schedule h) : Prop where
  pids : ∀ p∈schedule.programs,p.batch<2147483648 ∧ p.row<2147483648
  lanes : h.BK≤2147483648

def program (h : Call) (schedule : Snapshot.Schedule h) (abi : ABI h schedule)
    (p : Geometry.Program h) : NativeIndex.Program h :=
  ⟨p,(abi.pids p (schedule.covers p)).1,(abi.pids p (schedule.covers p)).2⟩

def outputPointer (h : Call) (storage : Placement h) (schedule : Snapshot.Schedule h)
    (abi : ABI h schedule) (p : Geometry.Program h) (n : Fin 256) : BitVec 64 :=
  BitVec.ofNat 64 storage.output.base+
    outputWord h (program h schedule abi p) n.val*BitVec.ofNat 64 storage.output.bytes

theorem output_readback (h : Call) (storage : Placement h) (schedule : Snapshot.Schedule h)
    (abi : ABI h schedule) (p : Geometry.Program h) (n : Fin 256) :
    (outputPointer h storage schedule abi p n).toNat=
      storage.output.address (Geometry.outputOffset h p.batch p.row n.val) := by
  unfold outputPointer
  rw [(word_offsets h (program h schedule abi p) 0 n.val 0).2.2]
  apply pointer_word
  rw [storage.outputSize]
  exact Geometry.output_inside h p n

theorem lane_words (h : Call) (schedule : Snapshot.Schedule h) (abi : ABI h schedule)
    (p : Geometry.Program h) (n : Fin 256) (k : Fin h.BK) :
    BitVec.ofInt 64 (BitVec.ofNat 32 k.val).toInt=BitVec.ofNat 64 k.val ∧
    BitVec.ofInt 64 (BitVec.ofNat 32 n.val).toInt=BitVec.ofNat 64 n.val :=
  NativeControl.arange_widens h ⟨program h schedule abi p,abi.lanes⟩ k n

#axiomsClean output_readback
#axiomsClean lane_words
end VeriTile.Bench.Repairs.BgmvExpandSlice.NativeGrid
