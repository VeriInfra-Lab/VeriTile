import bench.tritonbench_g.layer_norm_triton.Proofs.ForwardIO
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardReal

namespace VeriTile.Bench.Repairs.LayerNormTriton.ForwardRealIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section

def formula (N : Nat) (eps : ℝ) (vals : Forward.Inputs ℝ N) : Forward.Outputs ℝ N :=
  let x := Forward.value vals 0
  let w := Forward.value vals 1
  let bias := Forward.value vals 2
  Forward.result (fun j => RealProof.Spec.output N x w bias eps j.val)
    (RealProof.Spec.mean N x) (RealProof.Spec.inverse N x eps)

theorem implements (m : Model ℝ) (laws : SourceForwardReal.Laws m)
    (stride N B row : Nat) (eps : ℝ) (positive : 0<N) (covers : N≤B) :
    (Forward.io m stride N B row eps (fun _ _ => id)).Implements (formula N eps) := by
  have he : ForwardIO.expected m stride N B row eps (fun _ _ => id)=formula N eps := by
    funext vals port lane
    fin_cases port
    · exact SourceForwardReal.output_real m laws N B covers eps _ _ _ lane.1.val lane.1.isLt
    · exact SourceForwardReal.mean_real m laws N B covers _
    · exact SourceForwardReal.inverse_real m laws N B covers eps _
  rw [←he]
  exact ForwardIO.implements m stride N B row eps (fun _ _ => id) positive covers
#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormTriton.ForwardRealIO
