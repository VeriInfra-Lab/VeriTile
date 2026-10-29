import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SuccessfulForward

namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {metadata : MetadataAPI api neg inputs Metadata}
  {service : Runtime.Service platform api neg inputs Metadata} {identity : Nat}
  {x w : Storage.Tensor} {bias : Option Storage.Tensor} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

theorem forward_native (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (i : Fin r.body.preparation.data.y.count) :
    Storage.logical api neg r.final.memory r.body.output.tensor i.val=
      forwardReference (platform.scalar r.body.dispatch.code.object.primitives) r.body.dispatch.code.plan
        (Execution.rows r.body.preparation.data.x r.shape.columns r.body.dispatch.s2.memory api (i.val/r.shape.columns))
        (Execution.vector r.body.preparation.data.w r.body.dispatch.s2.memory api)
        (Runtime.biasValues (platform.scalar r.body.dispatch.code.object.primitives) api r.body.preparation.data r.body.dispatch.s2.memory)
        (platform.epsilon r.body.dispatch.code.object eps) (i.val%r.shape.columns) := by
  have hv := lib.viewed r.body.preparation.data.y x.view.shape r.body.sSave r.body.output r.body.viewed
  have hi : i.val<r.body.output.tensor.view.shape.prod := by rw [←r.body.output.countEq,hv.count]; exact i.isLt
  have h := (lib.leave r.sShape.owner).transition r.body.final |>.values _ hv.present i.val hi
  exact h.trans (body_native_readback r.body i)

#axiomsClean forward_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
