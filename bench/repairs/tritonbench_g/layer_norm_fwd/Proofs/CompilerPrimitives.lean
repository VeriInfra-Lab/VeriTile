import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperNativeArguments

/-! Named local numerical obligations for the selected emitted opcodes. They
are total scalar callbacks on opaque representations, not exact-Real laws.
No reduction, pointer, execution, output or finiteness premise occurs here. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Compiler
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section

structure FP32Primitives (α : Type) where
  fromHostBinary64 : UInt64 → α
  constant32 : ℝ → α
  addF32 : α → α → α
  subF32 : α → α → α
  mulF32 : α → α → α
  divFullF32 : α → α → α
  fmaRnF32 : α → α → α → α
  sqrtApproxFtzF32 : α → α
  greaterF32 : α → α → Bool
  philoxUniform32 : FP.Value .fp32 → α
  /-- Used only by the separate mathematical interpretation. Native execution
  and its graph correspondence never use or constrain this projection. -/
  decode : α → ℝ

def FP32Primitives.ops (primitives : FP32Primitives α) : NativeRow.Ops α where
  literal := primitives.constant32
  add := primitives.addF32
  sub := primitives.subF32
  mul := primitives.mulF32
  div := primitives.divFullF32
  fma := primitives.fmaRnF32
  sqrt := primitives.sqrtApproxFtzF32
  greater := primitives.greaterF32
  randomValue := primitives.philoxUniform32
  decode := primitives.decode
  cast32 := id

def FP32Primitives.encoding (primitives : FP32Primitives α) : WrapperNative.ScalarEncoding α :=
  ⟨primitives.fromHostBinary64⟩

theorem FP32Primitives.cast_identity (primitives : FP32Primitives α) (value : α) :
    primitives.ops.cast32 value=value := rfl

theorem FP32Primitives.host_word (primitives : FP32Primitives α) (value : Float) :
    primitives.encoding.encode value=primitives.fromHostBinary64 value.toBits := rfl

#axiomsClean FP32Primitives.cast_identity
#axiomsClean FP32Primitives.host_word
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Compiler
