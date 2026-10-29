import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativePublicIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeSource

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FPEquivalence
open VeriTile Triton FP.Structural Source
noncomputable section

/-- Actual FP16/E5M2 runtime-scalar source, original finite strided windows,
full ordinary grid and lane-specific native output observation. -/
abbrev io (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) := NativeIO.io A C arg B c

/-- Signed-byte representation for selected E5M2 inputs. This is an input
representation condition, not an experimental floating-point law. -/
abbrev inputDomain {α : Type} (B : Nat) (c : Config) := IOLayout.nativeDomain (α:=α) B c

/-- Complete source-to-native attention observation, with actual access safety
and exact exterior frame. The original primitive instruction profile remains
the model boundary; no numerical rewrite or native-equals-Real claim is added. -/
theorem attention_native_grid (B : Nat) (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config)
    (_supported : supportedHead c.D) (hD : 0<c.D) (hN : 0<c.N)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) :
    (io A C arg B c).ImplementsOn (inputDomain B c) (Mathematical.native A C arg B c) :=
  NativePublicIO.implements A C arg B c hD hN hinj

#stmtSurfaceSubset attention_native_grid ⊆
  [Target.Ops, ScalarABI.Conversions, ScalarABI.Argument, Config,
    Config.D, Config.N, Config.NH, Config.H, supportedHead,
    Function.Injective, IOLayout.output, io, inputDomain, OrdinaryProgramIO.ImplementsOn,
    OrdinaryProgramIO.Inputs, TileIndex, Term, Source.instInhabitedTerm, Mathematical.native]
#specNonCircular Mathematical.native avoiding [FP.Structural.exec,FP.Structural.run,Source.accumulated]
#axiomsClean attention_native_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FPEquivalence
