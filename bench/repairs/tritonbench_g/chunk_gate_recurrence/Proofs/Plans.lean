import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Native

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked

/-- Concrete local compiler reduction trees are checked for complete input
coverage. Neither check evaluates a tensor or accepts a reduction answer. -/
def planCheck (c : Shape) (tile : Reduction.ProductTree 1024) (host : Reduction.Tree c.tiles) : Bool :=
  decide (tile.leaves.Perm (List.finRange 1024)) && decide (host.leaves.Perm (List.finRange c.tiles))

def trees (c : Shape) (tile : Reduction.ProductTree 1024) (host : Reduction.Tree c.tiles)
    (accepted : planCheck c tile host=Bool.true) : Trees c :=
  let checks : tile.leaves.Perm (List.finRange 1024) ∧ host.leaves.Perm (List.finRange c.tiles) := by
    simpa only [planCheck,Bool.and_eq_true,decide_eq_true_eq] using accepted
  ⟨fun _ _ => ⟨tile,checks.1⟩,fun _ _ => ⟨host,checks.2⟩⟩

/-- Local numerical agreement is sufficient for every input word. Operand
coverage of the complete source graph is proved here, never a caller premise. -/
theorem numerical_all (isa : ISA) (laws : LocalConformance isa allOperands)
    (c : Shape) (plans : Trees c) (seed : Bool) (m : Memory Word) :
    Numerical isa allOperands c plans seed m :=
  ⟨laws,fun _ _ _ _ => ⟨Expr.all_admitted _ _,Expr.all_admitted _ _⟩⟩

theorem native_complete_all (isa : ISA) (laws : LocalConformance isa allOperands)
    (c : Shape) (plans : Trees c) (seed : Bool) (m : Memory Word) :
    NativeGraph.evalRun isa c plans seed m=Source.completeImage (wordOps SourceFP.scalar) c plans seed m :=
  native_complete isa allOperands c plans seed m (numerical_all isa laws c plans seed m)

#axiomsClean numerical_all
#axiomsClean native_complete_all
#specNonCircular planCheck avoiding [Source.run,Source.completeImage,NativeGraph.evalRun]
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
