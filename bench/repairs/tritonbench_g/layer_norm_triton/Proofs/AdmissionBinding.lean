import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Checked
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Saved
namespace VeriTile.Bench.Repairs.LayerNormTriton.Checked
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {caller : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {lib : APIHost api neg caller Metadata}

theorem materialize_identity (observations : Observations lib)
    (route : MaterializeRoute lib owner t shape initial)
    (checked : Contract.TensorContract (observations.tensor t)) :
    route.output.tensor.backing=t.backing := by
  have same := Host.dense_identity route.dense ((observations.contiguous t).trans checked.2.2.2.1)
  exact (lib.viewed route.dense.output shape route.dense.final route.output route.viewOK).backing.trans
    (congrArg (fun t : Storage.Tensor => t.backing) same)

/-- Guarded contiguous calls retain the original input storage. This is
derived from metadata and ordinary contiguous/view identities. -/
theorem forward_storage (observations : Observations lib)
    (service : Host.ForwardService platform api neg caller Metadata)
    (inputNormalized : Option (List Nat)) (inputEps : Option Float)
    (checked : Contract.InputContract (observations.tensor x) (observations.tensor w)
      (observations.tensor b) inputNormalized inputEps)
    (f : Host.Call lib service identity x w b (inputNormalized.getD []) (epsilon inputEps) initial) :
    f.body.preparation.data.x.tensor.backing=x.backing ∧
    f.body.preparation.data.w.tensor.backing=w.backing ∧
    f.body.context.weightDevice=f.body.context.x.tensor.view.device ∧
    f.body.context.biasDevice=some f.body.context.x.tensor.view.device := by
  have wx : w.view.device=x.view.device := by
    rw [←observations.device w,←observations.device x]
    exact congrArg (fun d : VeriTile.Torch.Device => d.index) checked.2.2.2.1
  have bx : b.view.device=x.view.device := by
    rw [←observations.device b,←observations.device x]
    exact congrArg (fun d : VeriTile.Torch.Device => d.index) checked.2.2.2.2.1
  have xd : f.body.context.x.tensor.view.device=x.view.device :=
    f.body.context.x.tensor.owner.trans (materialize_metadata f.body.preparation.xr).2.2.1
  exact ⟨materialize_identity observations f.body.preparation.xr checked.1,
    materialize_identity observations f.body.preparation.wr checked.2.1,
    wx.trans xd.symm,congrArg some (bx.trans xd.symm)⟩

#axiomsClean materialize_identity
#axiomsClean forward_storage
end
end VeriTile.Bench.Repairs.LayerNormTriton.Checked
