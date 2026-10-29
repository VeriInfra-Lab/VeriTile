import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativePipeline
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Spec
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.PublicCorrectness

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-- The complete shared DSL source computes the independent native word
target under the selected scalar and fixed-K8 instruction interpretation.
There are no execution/result premises or numerical deferrals. -/
theorem kernel_native (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ → Native.Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory Native.Word) (μ : RegionName → Nat → Native.Word)
    (hi : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,mem r o=.mk .fp32 (μ r o))
    (hz : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 0) :
    ∃ final,Native.Selected.nativeRun (Native.constexprScale scalar) c (scalar.project scalar.raw)
        owners I literal fw pw bw mem=some final ∧
      Native.Selected.WordValues (Native.constexprScale scalar) c (scalar.project scalar.raw) owners I literal μ final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) :=
  Native.Selected.whole_execution_words (Native.constexprScale scalar) c (scalar.project scalar.raw)
    owners I literal fw pw bw hf hp hb mem μ hi hz

#axiomsClean kernel_native
#axiomsClean Native.raw_integer_one
#axiomsClean Native.raw_float_runtime
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.FPEquiv

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.FPEquivBound
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-- Compatibility name for the independently defined scalar ABI input fact. -/
abbrev SourceScale := PublicIO.SourceScale

/-- The selected complete source implements its independent ordered native
formula. Primitive instructions, per-store replicas, internal FP16 casts and
the raw scalar ABI binding remain explicit. This is a native observation,
with no new kernel transformation or numerical assumption. -/
theorem kernel_native (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ → Native.Word) (binding : Native.ScalarBinding scalar literal)
    (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    SourceScale scalar I literal ∧
    (PublicIO.nativeIO scalar c owners I literal fw pw bw).ImplementsOnCells
      (PublicIO.ZeroDQ 0 c) (PublicIO.nativeFormula scalar c owners I literal) := by
  exact ⟨fun kind owner=>binding.source_scale_word scalar literal I kind owner,
    PublicIO.native_implements scalar c owners I literal fw pw bw hf hp hb⟩

#stmtSurfaceSubset kernel_native ⊆
  [Checked.ScalarABI,Shape,TritonAttention.Whole001.Native.Owners,Native.Prior.Instructions,Native.Word,
   Native.ScalarBinding,Grid.Program,Grid.FullSchedule,Grid.forwardGrid,Grid.preprocessGrid,Grid.backwardGrid,
   SourceScale,PublicIO.nativeIO,MaterializedProgramIO.ImplementsOnCells,PublicIO.ZeroDQ,PublicIO.nativeFormula,
   Native.Graph,Native.Expr,Native.instInhabitedExpr]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.FPEquivBound
