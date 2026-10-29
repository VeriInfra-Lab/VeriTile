import bench.repairs.tritonbench_g.int8_quantization.Proofs.RawKernel
import bench.repairs.tritonbench_g.int8_quantization.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.Int8Quantization.FloatingSpecification
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section

/-- Full raw source/software relation with the unchanged context ports and
fourteen reached primitive agreements. All words and both store channels are
covered; no Tensor admission or host scalar calculation is a kernel premise. -/
theorem per_block_int8_equiv_raw (p : Parameters) (positive : 0<p.width)
    (block : 0<p.blockRows) (M : Algebra Word) (ops : Model.Primitives Word)
    (entry : State Word) (software : Floating.Numerics Word)
    (contexts : ∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) →
      FPTrace.Context M (Grid.parameters p w) (Grid.atProgram entry w))
    (agreements : ∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) →
      Floating.Local (FPTrace.constants M (Grid.parameters p w)) (FPTrace.numerics M ops)
        software p.query (FPTrace.inputs M (Grid.parameters p w) (Grid.atProgram entry w))) :
    ∃ final,Grid.launch M ops p entry=some final ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) → ∀ i : Lane p,Grid.active p w i →
        final.mem outRegion (Grid.offset p w i)=.mk .int
          ((Floating.run (FPTrace.constants M p) software p.query
            (FPTrace.inputs M (Grid.parameters p w) (Grid.atProgram entry w))).1 i.1)) ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) →
        final.mem scaleRegion w.index=.mk .real
          ((Floating.run (FPTrace.constants M p) software p.query
            (FPTrace.inputs M (Grid.parameters p w) (Grid.atProgram entry w))).2)) ∧
      RawKernel.Frame p entry final := by
  obtain ⟨final,success,values,scales,frame⟩ := RawKernel.complete M ops p positive block entry
  have trace (w : Schedule.Program) (member : w∈Schedule.programs 0 (p.batches*p.blocks)) :=
    (FPTrace.arithmetic_trace M ops (Grid.parameters p w) positive (Grid.atProgram entry w)
      (contexts w member)).trans (Floating.local_equiv _ _ _ _ _ (agreements w member))
  refine ⟨final,success,?_,?_,frame⟩
  · intro w member i active
    rw [values w member i active]
    have value := congrFun (congrArg Prod.fst (trace w member)) i.1
    cases i with | mk j u => cases u;exact congrArg (Cell.mk .int) value
  · intro w member
    rw [scales w member]
    exact congrArg (Cell.mk .real) (congrArg Prod.snd (trace w member))

/-- Exact complete raw native launch against the independent ordered scalar
FP32 graph. The FP32 runtime factor, original primitive semantics and explicit
maximum reduction plan are retained, including exceptional words. -/
theorem per_block_int8_native_raw (p : Parameters) (positive : 0<p.width)
    (block : 0<p.blockRows) (rest : Algebra Word) (op : NativeGraph.Instructions)
    (plans : NativeGraph.Plans) (entry : State Word) :
    let M := NativeGraph.algebra rest op plans
    ∃ final,Grid.launch M (NativeGraph.comparisons op) p entry=some final ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) → ∀ i : Lane p,Grid.active p w i →
        final.mem outRegion (Grid.offset p w i)=.mk .int
          ((NativeGraph.block op plans p.query p.factor
            (FPTrace.inputs M (Grid.parameters p w) (Grid.atProgram entry w))).1 i.1)) ∧
      (∀ w,w∈Schedule.programs 0 (p.batches*p.blocks) →
        final.mem scaleRegion w.index=.mk .real
          ((NativeGraph.block op plans p.query p.factor
            (FPTrace.inputs M (Grid.parameters p w) (Grid.atProgram entry w))).2)) ∧
      RawKernel.Frame p entry final :=
  per_block_int8_equiv_raw p positive block (NativeGraph.algebra rest op plans)
    (NativeGraph.comparisons op) entry (NativeGraph.numerics op plans)
    (fun w _ => NativeGraph.context rest op plans _ _)
    (fun w _ => NativeGraph.local_agreement rest op plans _ _ _)

#axiomsClean per_block_int8_equiv_raw
#axiomsClean per_block_int8_native_raw
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8Quantization.FloatingSpecification
