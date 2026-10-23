import bench.repairs.tritonbench_g.fused_activation.Proofs.NativeInterface

/-! Auxiliary native pair equivalence. The source-level strict addition-commutation result is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.FusedActivation.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
open scoped VeriTile.Spec
noncomputable section

/-- Native IO has the same three reads and in-place output. IO.lean spells
out the windows/masks and both authoritative source invocations; the observer
retains native FMA and instruction-sized sigmoid ports. -/
def nativeOriginal := NativeIO.original
def nativeCommuted := NativeIO.commuted

/-- Complete native pair equivalence on the existing finite outer-add domain.
Both actual sources are access-safe, execute successfully, have equal typed
native outputs, and preserve the exact active-store exterior frame. -/
theorem fused_activation_native_equiv (rest : FP.Structural.Algebra Graph.Expr)
    (ops : Graph.Operations) (cfg : FP.Config) (binding : NativeRewrite.AddBinding ops cfg)
    (D : FP.Guarded.Domain Graph.Word)
    (p : Parameters) (W : Nat) (a : Bool) (launch : Nat → Nat) :
    (nativeOriginal rest ops p W a launch).EquivalentOn
      (nativeCommuted rest ops p W a launch) (NativeIO.domain ops D p launch) :=
  NativeInterface.equivalent rest ops cfg binding D p W a launch

#stmtSurfaceSubset fused_activation_native_equiv ⊆
  [nativeOriginal, nativeCommuted, Parameters, Graph.Expr, Graph.Word, Graph.Operations,
   FP.Structural.Algebra, FP.Guarded.Domain, FP.Config,
   NativeRewrite.AddBinding, NativeIO.domain, FP.Structural.ProgramIO.EquivalentOn,
   Graph.instInhabitedExpr]

#axiomsClean fused_activation_native_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedActivation.FPEquiv
