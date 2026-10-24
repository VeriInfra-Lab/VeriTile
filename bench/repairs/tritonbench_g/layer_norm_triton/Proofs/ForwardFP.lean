import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardCall
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Numerics

namespace VeriTile.Bench.Repairs.LayerNormTriton.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : ForwardService platform api neg inputs Metadata}
  {p : Preparation.Data} {layout : Prepared p M N inputs}
  {settings : Compiler.settings N=.ok (B,Geometry.warps B)} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

def LaunchRoute.conformance (r : LaunchRoute service p layout settings eps initial) (target : Numerics.Target) : Prop :=
  match r with | .skipped _ => True | .ran _ r => Numerics.Conformance platform target r.code neg

def LaunchRoute.targetValue (r : LaunchRoute service p layout settings eps initial) (target : Numerics.Target) (i : Nat) : UInt32 :=
  match r with
  | .skipped _ => 0
  | .ran _ r => forwardValue (target.scalar r.code.object.primitives) r.code.plan
      (Execution.E.rows p.x N r.s2.memory api (i/N)) (Execution.E.vector p.w r.s2.memory api)
      (Execution.E.vector (p.bias.getD p.w) r.s2.memory api)
      (target.epsilon r.code.object.primitives.epsilon r.code.object.signature.epsilon eps) (i%N)

theorem launch_fp (r : LaunchRoute service p layout settings eps initial) (target : Numerics.Target)
    (laws : r.conformance target) (i : Nat) : r.nativeValue i=r.targetValue target i := by
  cases r with
  | skipped _ => rfl
  | ran _ r =>
    change Numerics.Conformance platform target r.code neg at laws
    change forwardValue (platform.scalar r.code.object.primitives) _ _ _ _ _ _ = _
    rw [laws.primitives.operations,laws.epsilon]
    rfl

variable {lib : APIHost api neg inputs Metadata} {identity : Nat} {x w b : Storage.Tensor} {normalized : List Nat}

theorem returned_fp (r : Call lib service identity x w b normalized eps initial) (target : Numerics.Target)
    (laws : r.body.launch.conformance target) (i : Fin r.body.preparation.data.y.count) :
    Storage.logical api neg r.final.memory r.body.output.tensor i.val=r.body.launch.targetValue target i.val :=
  (returned_native r i).trans (launch_fp r.body.launch target laws i.val)

#axiomsClean returned_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Host
