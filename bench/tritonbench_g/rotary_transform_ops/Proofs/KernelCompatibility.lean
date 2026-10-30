import bench.tritonbench_g.rotary_transform_ops.Proofs.KernelEndpoints

/-! Kernel-only public mathematical endpoints. The complete typed quotation
executes from arbitrary Real inputs and original signed metadata. Every
logical output coordinate satisfies the independent rotation formula, with a
complete tagged-memory frame. The ordinary suffix-copy state is an entry
invariant; allocator and returned-Tensor semantics remain historical. -/
namespace VeriTile.Bench.Repairs.RotaryTransformOps
open VeriTile.Triton FP.Structural
noncomputable section

theorem rotary_transform_ops_correct_real (c : Config) (valid : c.Valid) (initial : State ℝ)
    (md : ProgramModel.MetadataBinding c initial) (suffix : KernelEndpoints.TailReady c initial)
    (cumulative64 offsets64 : Bool) :
    ∃ final,TypedGrid.run realOps c cumulative64 offsets64 initial=some final ∧
      (∀ batch row head col,batch<c.shape.batches → row<c.length batch →
        head<c.shape.heads → col<c.shape.width →
        (final.mem (KernelAlgebra.regionName c.shape.output)
          (c.shape.address (c.start batch+row) head col)).read .real=
            realFormula c (KernelAlgebra.memory initial) batch row head col) ∧
      (∀ r o,TypedGrid.Outside c r o → final.mem r o=initial.mem r o) :=
  KernelEndpoints.real c valid initial md suffix cumulative64 offsets64
abbrev rotary_transform_ops_reference := realFormula
abbrev rotary_transform_ops_grid_order := @TypedGrid.complete_order
abbrev rotary_transform_ops_output_coverage := KernelEndpoints.every_token

#axiomsClean rotary_transform_ops_correct_real
#axiomsClean rotary_transform_ops_grid_order
#specNonCircular rotary_transform_ops_reference avoiding
  [FP.Structural.exec,TypedGrid.run,ProgramModel.result,source,sourcePair,pairValues]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryTransformOps


/-! Kernel-only exact raw-FP32 output under the explicitly selected primitive
and compiler graph. All bit patterns are retained. The four primitive
callbacks and fixed default FMA pattern are model boundaries, with no
whole-output equality, numerical experiment, or deferred scalar equation as
a premise of this public endpoint. -/
namespace VeriTile.Bench.Repairs.RotaryTransformOps
open VeriTile.Triton FP.Structural
noncomputable section

theorem rotary_transform_ops_kernel_fp32 (c : Config) (valid : c.Valid) (p : Native.Instructions) (initial : State Native.Node)
    (md : ProgramModel.MetadataBinding c initial) (suffix : KernelEndpoints.TailReady c initial)
    (cumulative64 offsets64 : Bool) :
    ∃ final,TypedGrid.run (Native.ops p) c cumulative64 offsets64 initial=some final ∧
      (∀ batch row head col,batch<c.shape.batches → row<c.length batch →
        head<c.shape.heads → col<c.shape.width →
        ((final.mem (KernelAlgebra.regionName c.shape.output)
          (c.shape.address (c.start batch+row) head col)).read .real).value=
            nativeFormula c p (Native.project (KernelAlgebra.memory initial)) batch row head col) ∧
      (∀ r o,TypedGrid.Outside c r o → final.mem r o=initial.mem r o) :=
  KernelEndpoints.native c valid p initial md suffix cumulative64 offsets64
abbrev rotary_transform_ops_native_reference := nativeFormula
abbrev rotary_transform_ops_native_graph := Native.source_graph_lowering
abbrev rotary_transform_ops_padded_integer := KernelIndex.padded_coefficient
abbrev rotary_transform_ops_native_coverage := KernelEndpoints.every_token

#axiomsClean rotary_transform_ops_kernel_fp32
#axiomsClean rotary_transform_ops_native_graph
#specNonCircular rotary_transform_ops_native_reference avoiding
  [FP.Structural.exec,TypedGrid.run,ProgramModel.result,Native.graph,Native.pairWord,Native.ops]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryTransformOps
