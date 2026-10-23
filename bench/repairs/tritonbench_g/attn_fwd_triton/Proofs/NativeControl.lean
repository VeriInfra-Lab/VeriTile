import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeOffsets
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeLaunch

/-! CUDA launch feasibility supplies control bounds. These limits describe
successful setup; oversized grids remain ordinary API failures, not excluded
logical tensors. The selected CUDA grid limits are x≤2^31−1, y,z≤65535;
Triton's arange/tensor limit is 1048576 elements (installed Triton 3.7.1). -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeControl
open VeriTile.Triton Memory GridCoordinates
open VeriTile.Bench.Repairs.LayerNormFwd.Address
noncomputable section

def shape (p : Parameters) : NativeLaunch.Shape := ⟨p.batches,p.heads,p.context⟩
abbrev CudaLaunchable (p : Parameters) (BD : Nat) := NativeLaunch.CudaLaunchable (shape p) BD

theorem context (p : Parameters) (BD : Nat) (h : CudaLaunchable p BD) :
    p.context ≤ 274877906816 := by
  exact NativeLaunch.context (shape p) BD h

theorem cdiv_inputs (p : Parameters) (BD : Nat) (h : CudaLaunchable p BD) :
    signed64 p.context=(p.context : Int) ∧
      signed64 (p.context+127)=(p.context+127 : Nat) ∧
      signed64 (p.context+63)=(p.context+63 : Nat) := by
  exact NativeLaunch.cdiv_inputs (shape p) BD h

theorem row_arithmetic (p : Parameters) (BD start row : Nat) (h : CudaLaunchable p BD)
    (hs : start*128<p.context) (hr : row<128) :
    signed64 (start*128+row)=(start*128+row : Nat) ∧
      signed64 ((start+1)*128)=((start+1)*128 : Nat) ∧
      signed64 (min ((start+1)*128) p.context)=(min ((start+1)*128) p.context : Nat) := by
  exact NativeLaunch.row_arithmetic (shape p) BD start row h hs hr

theorem key_arithmetic (p : Parameters) (BD cursor lane : Nat) (h : CudaLaunchable p BD)
    (hc : cursor<p.context) (hl : lane<64) :
    signed64 (p.context-cursor)=(p.context : Int)-(cursor : Int) ∧
      signed64 (cursor+lane)=(cursor+lane : Nat) ∧
      signed64 (cursor+64)=(cursor+64 : Nat) := by
  exact NativeLaunch.key_arithmetic (shape p) BD cursor lane h hc hl

theorem feature_lanes (p : Parameters) (BD lane : Nat) (h : CudaLaunchable p BD) (hl : lane<BD) :
    signed64 lane=(lane : Int) := by
  exact NativeLaunch.feature_lanes (shape p) BD lane h hl

theorem packed_offsets (p : Parameters) (BD head : Nat) (h : CudaLaunchable p BD)
    (hh : head<p.batches*p.heads) :
    signed64 (head*((p.context+127)/128))=(head*((p.context+127)/128) : Nat) ∧
      signed64 (head*((p.context+63)/64))=(head*((p.context+63)/64) : Nat) := by
  exact NativeLaunch.packed_offsets (shape p) BD head h hh

/-- Classification is ordered before device execution: failure to configure
the launch is an API outcome, never a numerical or memory-safety premise. -/
inductive Setup (p : Parameters) (BD : Nat) where
  | empty (h : p.batches*p.heads*p.context*p.headDim=0)
  | launch (h : CudaLaunchable p BD)
  | failure

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeControl
