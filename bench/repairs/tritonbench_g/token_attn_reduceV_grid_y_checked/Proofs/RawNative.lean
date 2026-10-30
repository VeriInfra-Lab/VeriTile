import bench.repairs.tritonbench_g.token_attn_reduceV.Native

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.RawNative
open VeriTile.Triton FP.Structural TokenAttnReduceV
noncomputable section

/-- Complete source-grid success and native outputs over a raw typed word heap.
The source-ordered input recurrence, exact reduction trees and local FMA
selection retain the existing primitive/compiler boundaries. -/
theorem grid_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (batches heads : Nat) (work : List Grid.PID)
    (order : (Grid.fullGrid batches heads).Perm work) (memory : Grid.Memory Native.Word)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    ∃ after,Grid.run Native.algebra (Native.reduction R) p work (Native.entry memory)=some after ∧
      (∀ b<batches,∀ h<heads,∀ d : Fin p.width,
        Native.observe I S after p.output (Grid.offset p (b,h) (d,PUnit.unit))=
          Native.eval I S (Grid.values Native.algebra (Native.reduction R) p
            (Native.entry memory) ((b,h),(d,PUnit.unit)))) ∧
      (∀ r o,(r≠p.output ∨ ∀ e∈Grid.events p work,o≠Grid.offset p e.1 e.2) →
        after r o=Native.entry memory r o) := by
  obtain ⟨after,run,values,frame⟩ := Native.complete I R S p work memory separate unique
  refine ⟨after,run,?_,frame⟩
  intro b hb h hh d
  let e : Grid.Event p := ((b,h),(d,PUnit.unit))
  exact values e ((Grid.mem_events _ _ _).mpr
    (order.mem_iff.mp ((Grid.mem_fullGrid _ _ _).mpr ⟨hb,hh⟩)))

#axiomsClean grid_native
#specNonCircular Grid.values avoiding [kernel,FP.Structural.exec,Grid.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.RawNative
