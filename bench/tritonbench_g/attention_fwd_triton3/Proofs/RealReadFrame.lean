import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealFinitePins
import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeFrame

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

theorem written_input_owner (c : Call) (valid : Valid c) (idx jdx : GridIndex c.grid)
    (lane : Lane c.BM c.D) (i : Fin 6) (j : TileIndex (inputShape c i))
    (same : address c idx lane=(inputRegion i,inputAddress c i (jdx,j))) : idx=jdx := by
  fin_cases i <;> cases lane <;>
    simp [address,inputRegion,inputAddress,Call.io,RealContract.io,RealIO.IO.inputWindow,
      Call.params,Call.carryAddress,Call.row,Call.outAddress,Geometry.Strides.at,Nat.add_assoc] at same
  · exact (carry_injective c valid idx jdx _ j.1 same).1
  · exact (carry_injective c valid idx jdx _ j.1 same).1
  · rename_i r d
    exact (output_injective c valid idx jdx r j.1 d j.2.1
      (by simpa [Call.outAddress,Call.row,Geometry.Strides.at,Nat.add_assoc] using same)).1

theorem scatter_pinned_other (c : Call) (valid : Valid c) (v : Inputs c)
    (s : State Real.R) (idx jdx : GridIndex c.grid) (different : idx≠jdx)
    (pins : PinnedAt c v jdx s) (old : Reference.Carry c Real.R) :
    PinnedAt c v jdx
      (Reference.scatter c Real.ops (Reference.inputs (Reference.Composition.prepare c.grid idx s)) old
        (Reference.Composition.prepare c.grid idx s)) := by
  intro i j active
  rw [ArrayLocality.scatter_frame]
  · exact pins i j active
  · intro lane live same
    rw [ArrayLocality.prepared_address] at same
    exact different (written_input_owner c valid idx jdx lane i j same)

#axiomsClean scatter_pinned_other
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
