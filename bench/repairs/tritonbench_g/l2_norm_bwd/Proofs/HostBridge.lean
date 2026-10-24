import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Bridge
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Execution

namespace VeriTile.Bench.Repairs.L2NormBwd.Bridge
open VeriTile.Triton FP.Structural Physical Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

/-- Direct connection to this call's actual resolved handles and launch state;
all rows are derived from full source execution. -/
theorem native_launch {M β : Type} (ops : Ops M Native.Word β) (neg : Negation Native.Word)
    (instructions : Native.Instructions) (rt : Runtime M) (lib : Library ops neg rt)
    (q : Request) (eps : Native.Word) (before after : World M)
    (path : BodyPath ops (Native.reduction instructions) rt q eps before after) :
    after.memory=sourceRows ops (Native.algebra instructions path.layout.block) Native.atom
      (fun c n => n.value (c%128)) path.layout (Native.atom eps)
      (rt.schedule path.layout path.launch) path.launch.memory := by
  calc
    after.memory = Physical.run ops (Native.reduction instructions) path.layout eps
      (rt.schedule path.layout path.launch) path.launch.memory := congrArg World.memory path.computed
    _ = _ := ?_
  symm
  apply native_rows ops instructions path.layout path.valid
  intro r hr
  exact List.mem_range.mp ((lib.schedule path.layout path.launch).mem_iff.mp hr)

theorem structural_launch {M α β : Type} [Inhabited α] (ops : Ops M α β) (neg : Negation α)
    (alg : Algebra α) (rt : Runtime M) (lib : Library ops neg rt)
    (q : Request) (eps : α) (before after : World M)
    (path : BodyPath ops (sourceReduction alg) rt q eps before after) :
    after.memory=sourceRows ops alg id (fun _ => id) path.layout eps
      (rt.schedule path.layout path.launch) path.launch.memory := by
  calc
    after.memory = Physical.run ops (sourceReduction alg) path.layout eps
      (rt.schedule path.layout path.launch) path.launch.memory := congrArg World.memory path.computed
    _ = _ := ?_
  symm
  apply structural_rows ops alg path.layout path.valid
  intro r hr
  exact List.mem_range.mp ((lib.schedule path.layout path.launch).mem_iff.mp hr)

#axiomsClean native_launch
#axiomsClean structural_launch
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Bridge
