import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.HostBridge
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Machine
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Profile

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

structure StructuralCertificate {M α β : Type} [Inhabited α] (ops : Ops M α β) (alg : Algebra α)
    (rt : Runtime M) (call : Call) (eps : α) (initial : World M) (output : View) (after : World M) where
  path : WholePath ops (sourceReduction alg) rt call eps initial output after
  source : path.context.bodyAfter.memory=Bridge.sourceRows ops alg id (fun _ => id)
    path.context.body.layout eps (rt.schedule path.context.body.layout path.context.body.launch)
    path.context.body.launch.memory
  machine : Machine.LaunchSafe path.context.body
  profile : path.context.body.layout.block∈Profile.blocks
  placement : Machine.Placement path.context.body

structure NativeCertificate {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (rt : Runtime M) (call : Call) (eps : Native.Word) (initial : World M) (output : View) (after : World M) where
  path : WholePath ops (Native.reduction instructions) rt call eps initial output after
  source : path.context.bodyAfter.memory=Bridge.sourceRows ops (Native.algebra instructions path.context.body.layout.block)
    Native.atom (fun c n => n.value (c%128)) path.context.body.layout (Native.atom eps)
    (rt.schedule path.context.body.layout path.context.body.launch) path.context.body.launch.memory
  machine : Machine.LaunchSafe path.context.body
  profile : path.context.body.layout.block∈Profile.blocks
  placement : Machine.Placement path.context.body

theorem structural_certificate {M α β : Type} [Inhabited α] (ops : Ops M α β) (neg : Negation α)
    (alg : Algebra α) (rt : Runtime M) (lib : Library ops neg rt) (grids : Machine.GridContract rt)
    (call : Call) (eps : α) (initial after : World M) (output : View) (domain : InputDomain call initial)
    (success : run ops (sourceReduction alg) rt call eps initial=.ok output after) :
    Nonempty (StructuralCertificate ops alg rt call eps initial output after) := by
  obtain ⟨path⟩ := successful_path ops neg (sourceReduction alg) rt lib call eps initial after output domain success
  exact ⟨⟨path,Bridge.structural_launch ops neg alg rt lib _ eps _ _ path.context.body,
    Machine.launch_safe grids path.context.body,
    Profile.path_profile ops neg (sourceReduction alg) rt lib call eps initial after output domain path,
    Machine.placement ops neg (sourceReduction alg) rt lib call eps initial after output domain path⟩⟩

theorem native_certificate {M β : Type} (ops : Ops M Native.Word β) (neg : Negation Native.Word)
    (instructions : Native.Instructions) (rt : Runtime M) (lib : Library ops neg rt)
    (grids : Machine.GridContract rt) (call : Call) (eps : Native.Word) (initial after : World M)
    (output : View) (domain : InputDomain call initial)
    (success : run ops (Native.reduction instructions) rt call eps initial=.ok output after) :
    Nonempty (NativeCertificate ops instructions rt call eps initial output after) := by
  obtain ⟨path⟩ := successful_path ops neg (Native.reduction instructions) rt lib call eps initial after output domain success
  exact ⟨⟨path,Bridge.native_launch ops neg instructions rt lib _ eps _ _ path.context.body,
    Machine.launch_safe grids path.context.body,
    Profile.path_profile ops neg (Native.reduction instructions) rt lib call eps initial after output domain path,
    Machine.placement ops neg (Native.reduction instructions) rt lib call eps initial after output domain path⟩⟩

#axiomsClean structural_certificate
#axiomsClean native_certificate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host
