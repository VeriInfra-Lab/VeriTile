import bench.repairs.tritonbench_g.token_attn_reduceV.HostCore
import bench.repairs.tritonbench_g.token_attn_reduceV.Native
namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000

structure Result (α : Type) where
  returned : Option Nat
  output : Cube
  memory : Memory α

def invoke {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) (m : Memory α) : Option (Result α) := do
  let final ← Grid.run M R (params c) (fullGrid (batches c) (heads c)) m
  pure ⟨none,c.output,final⟩

def outputEvents (c : Call) := events (params c) (fullGrid (batches c) (heads c))

theorem events_exact (c : Call) (e : Event (params c)) :
    e∈outputEvents c ↔ e.1.1<batches c ∧ e.1.2<heads c := by
  rw [outputEvents,mem_events,mem_fullGrid]

def Frame {α : Type} (c : Call) (before after : Memory α) :=
  ∀ r o, (r≠c.output.region ∨ ∀ e∈outputEvents c,o≠offset (params c) e.1 e.2) →
    after r o=before r o

theorem whole_typed {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) (m : Memory α) (layout : Layout c) :
    ∃ result, invoke M R c m=some result ∧ result.returned=none ∧ result.output=c.output ∧
      result.memory=(outputEvents c).foldl (Grid.write (params c) (values M R (params c) m)) m ∧
      (∀ e∈outputEvents c,
        result.memory c.output.region (offset (params c) e.1 e.2)=
          .mk .fp32 (values M R (params c) m e)) ∧ Frame c m result.memory := by
  obtain ⟨final,run,trace,readback,frame⟩ := Grid.whole M R (params c)
    (fullGrid (batches c) (heads c)) m layout.separated (unique c layout)
  refine ⟨⟨none,c.output,final⟩,?_,rfl,rfl,trace,readback,frame⟩
  simp [invoke,run]

/-- Actual source graph, under the same local scalar instructions and compiler
choices. Numerical words are not consulted by the compiler selectors. -/
theorem whole_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (c : Call) (m : Memory Native.Word) (layout : Layout c) :
    ∃ result, invoke Native.algebra (Native.reduction R) c (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        Native.observe I S result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)=
          Native.eval I S (values Native.algebra (Native.reduction R) (params c) (Native.entry m)
            ((b.val,h.val),(d,PUnit.unit)))) ∧ Frame c (Native.entry m) result.memory := by
  obtain ⟨result,run,ret,out,_,readback,frame⟩ := whole_typed Native.algebra (Native.reduction R) c (Native.entry m) layout
  refine ⟨result,run,ret,out,?_,frame⟩
  intro b h d
  let e : Event (params c) := ((b.val,h.val),(d,PUnit.unit))
  have live : e∈outputEvents c := (events_exact c e).mpr ⟨b.isLt,h.isLt⟩
  have cell := readback e live
  rw [outputOffset_dense c layout] at cell
  change Native.eval I S ((result.memory c.output.region ((e.1.1*heads c+e.1.2)*c.value.width+e.2.1.val)).read .fp32)=_
  rw [cell,Cell.read_same]

#axiomsClean events_exact
#axiomsClean whole_typed
#axiomsClean whole_native
#specNonCircular values avoiding [Grid.run,invoke]
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Host
