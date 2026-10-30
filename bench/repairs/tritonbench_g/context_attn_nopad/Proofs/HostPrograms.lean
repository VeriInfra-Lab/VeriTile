import bench.repairs.tritonbench_g.context_attn_nopad.Kernels
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeLaunch
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideSource
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Wrapper

/-! Compatibility host-return wrappers, separate from the canonical typed source.
The int32 and wider metadata interfaces retain their existing names and bodies.
NativeLaunch and WideSource compose native programs with their complete host
results; Wrapper models the ordered ordinary API prefix. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad
open VeriTile.Triton
noncomputable section

abbrev synchronizeMetadataMaximum := HostPlan.metadataMaximum
abbrev nativePrograms := HostPlan.jobs
abbrev nativeProgram := NativeProgram.execute
abbrev nativeProjection := NativeProgram.Projects
abbrev hostLaunch := NativeLaunch.Returns
abbrev logicalWriteback := @Logical.finish

/-- Observable source-model host program, after coordinate-preserving tensor
setup: actual metadata synchronization and native launches followed by caller
writeback. The caller's original max_input_len remains an argument here. -/
def hostProgram (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (view : Logical.OutputView c) (before : Nat → ℝ)
    (result : Logical.CallerResult c) : Prop :=
  ∃ raw, hostLaunch c r A s raw ∧
    result = Logical.finish view before (Logical.Values c r A raw.1)

abbrev wideProjection := WideNative.Projects
abbrev wideHostLaunch := WideSource.Returns

def wideHostProgram (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (types : WideNative.MetadataTypes) (view : Logical.OutputView c) (before : Nat → ℝ)
    (result : Logical.CallerResult c) : Prop :=
  ∃ raw, WideSource.Returns c r A s types raw ∧
    result = Logical.finish view before (Logical.Values c r A raw.1)

end
end VeriTile.Bench.Repairs.ContextAttnNopad
