import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Compiler
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Composition

/-! Default FP32 admits all UInt32 payloads. Primitive names are selected with
actual object bytes/options, including sqrt.approx.ftz.f32 and div.full.f32.
The scalar interface never assumes finite values, FP associativity, exact
sqrt, or FMA=mul+add. CPU/offline evidence is not target-bit conformance. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Numerics
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

structure Target where
  scalar : Compiler.Primitives → Ops UInt32
  epsilon : String → Option Compiler.Shared.Binding → Compiler.Shared.HostScalar → UInt32
  negate : UInt32 → UInt32

structure Conformance (platform : Compiler.Platform) (target : Target)
    (code : Compiler.Selected platform.decoder entry N B) (neg : UInt32 → UInt32) : Prop where
  primitives : LocalExact (platform.scalar code.object.primitives) (target.scalar code.object.primitives)
  epsilon : ∀ value,platform.epsilon code.object value=
    target.epsilon code.object.primitives.epsilon code.object.signature.epsilon value
  tensorNegation : ∀ word,neg word=target.negate word

theorem forward_scalar (a b : Ops UInt32) (laws : LocalExact a b) (plan : Plan N B)
    (x w bias : Nat → UInt32) (eps : UInt32) (column : Nat) :
    forwardValue a plan x w bias eps column=forwardValue b plan x w bias eps column := by
  rw [laws.operations]

theorem backward_scalar (a b : Ops UInt32) (laws : LocalExact a b) (plan : Plan N B)
    (stats : Reuse.Statistics UInt32) (x dy w : Nat → UInt32) (column : Nat) :
    backwardValue a plan stats x dy w column=backwardValue b plan stats x dy w column := by
  rw [laws.operations]

theorem twoLaunch_memory (a b : Ops UInt32) (laws : LocalExact a b) (plan : Plan N B)
    (last : FinalPlan) (api : Memory.API UInt32 Store UInt8)
    (p : Execution.BackwardBuffers M N G inputs) (q : Execution.FinalBuffers N inputs)
    (x w dy r mu : VeriTile.Bench.Repairs.FastLayernorm.Storage.Dense)
    (m : VeriTile.Bench.Repairs.FastLayernorm.Storage.World Store)
    (first : Execution.E.Schedule 3) (second : Execution.E.Schedule 2) :
    Execution.twoLaunch a plan last api p q x w dy r mu m first second=
      Execution.twoLaunch b plan last api p q x w dy r mu m first second := by
  rw [laws.operations]

#axiomsClean twoLaunch_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Numerics
