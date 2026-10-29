import bench.repairs.tritonbench_g.token_attn_reduceV.Bounds
import bench.repairs.tritonbench_g.token_attn_reduceV.HostSchedule
namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Host.Historical
open VeriTile.Triton FP.Structural Grid
noncomputable section

/-- Complete successful execution for the actual typed source graph. -/
theorem host_typed {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) (m : Memory α) (work : List PID)
    (layout : Layout c) (data : Data c m) (storage : Storage c)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ ∃ result, invokeActual M R c work m=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      result.memory=(outputEvents c).foldl (Grid.write (params c) (values M R (params c) m)) m ∧
      (∀ e∈outputEvents c,
        result.memory c.output.region (offset (params c) e.1 e.2)=
          .mk .fp32 (values M R (params c) m e)) ∧ Frame c m result.memory := by
  refine ⟨accesses c m layout data storage,?_⟩
  rw [invokeActual_eq _ _ _ _ _ layout legal]
  exact whole_typed M R c m layout

/-- Raw FP32 observation, preserving default fusion and the actual selected
scalar reduction tree. Only local instruction interpretation remains open. -/
theorem host_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (c : Call) (m : Memory Native.Word) (work : List PID)
    (layout : Layout c) (data : Data c m) (storage : Storage c)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ ∃ result,
      invokeActual Native.algebra (Native.reduction R) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        Native.observe I S result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)=
          Native.eval I S (values Native.algebra (Native.reduction R) (params c) (Native.entry m)
            ((b.val,h.val),(d,PUnit.unit)))) ∧ Frame c (Native.entry m) result.memory := by
  refine ⟨accesses c m layout data storage,?_⟩
  rw [invokeActual_eq _ _ _ _ _ layout legal]
  exact whole_native I R S c m layout

/-- Every full logical output cell is written, and all other memory is
preserved. This graph coverage theorem uses no scalar interpretation. -/
theorem host_native_coverage (R : Native.Reductions)
    (c : Call) (m : Memory Native.Word) (work : List PID)
    (layout : Layout c) (data : Data c m) (storage : Storage c)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ (∀ e : Event (params c),e∈outputEvents c ↔
      e.1.1<batches c ∧ e.1.2<heads c) ∧
    ∃ result, invokeActual Native.algebra (Native.reduction R) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      result.memory=(outputEvents c).foldl
        (Grid.write (params c) (values Native.algebra (Native.reduction R) (params c) (Native.entry m))) (Native.entry m) ∧
      (∀ e∈outputEvents c,result.memory c.output.region (offset (params c) e.1 e.2)=
        .mk .fp32 (values Native.algebra (Native.reduction R) (params c) (Native.entry m) e)) ∧
      Frame c (Native.entry m) result.memory := by
  refine ⟨accesses c m layout data storage,events_exact c,?_⟩
  rw [invokeActual_eq _ _ _ _ _ layout legal]
  exact whole_typed Native.algebra (Native.reduction R) c (Native.entry m) layout

#axiomsClean host_typed
#axiomsClean host_native
#axiomsClean host_native_coverage
#auditStmt host_typed
#auditStmt host_native
#specNonCircular values avoiding [FP.Structural.exec,Grid.run,invoke,invokeActual]
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Host.Historical
