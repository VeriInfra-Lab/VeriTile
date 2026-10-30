import bench.tritonbench_g.token_attn_reduceV.Proofs.Schedule
import bench.tritonbench_g.token_attn_reduceV.Proofs.LogicalValues

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section
variable {A : Type} [Inhabited A]

theorem outside_old (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID)
    (valid : ∀ pid∈jobs, pid.1<B ∧ pid.2<H) (v : Inputs A p B H counts)
    (r : RegionName) (o : Nat) (outside : (programIO M R p B H counts jobs).Outside v r o) :
    r≠p.output ∨ ∀ e∈Grid.events p jobs, o≠Grid.offset p e.1 e.2 := by
  by_cases hr : r=p.output
  · subst r
    right
    intro e he same
    have dims := valid e.1 ((Grid.mem_events p jobs e).mp he)
    have miss := outside.1 ⟨0,by simp [programIO,memory,outputs]⟩
      (⟨e.1.1,dims.1⟩,⟨e.1.2,dims.2⟩,e.2.1,PUnit.unit) trivial
    exact miss.elim (fun h => h rfl) (fun h => h same)
  · exact Or.inl hr

/-- The full existing grid theorem supplies output cells and frame; actual
source success/safety follow the same concrete invocation list. -/
theorem implements (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID)
    (order : (Grid.fullGrid B H).Perm jobs) (separate : Grid.Separate p) (unique : Grid.Unique p jobs) :
    (programIO M R p B H counts jobs).ImplementsOn (LengthsMatch p B H counts)
      (orderedTarget M R p B H counts) := by
  intro v domain bounds bounded initial inputs
  have valid : ∀ pid∈jobs, pid.1<B ∧ pid.2<H := by
    intro pid member
    exact (Grid.mem_fullGrid B H pid).mp (order.mem_iff.mpr member)
  refine ⟨schedule_safe M R p B H counts jobs valid separate v domain bounds bounded initial inputs,?_⟩
  obtain ⟨final,execution,mem⟩ := schedule_complete M R p jobs initial
  obtain ⟨after,oldRun,_,values,frame⟩ := Grid.whole M R p jobs initial.mem separate unique
  have old : after=Grid.graph M R p jobs initial.mem := Option.some.inj (oldRun.symm.trans (Grid.complete M R p jobs initial.mem))
  have finalMem : final.mem=after := mem.trans old.symm
  refine ⟨final,execution,?_,?_⟩
  · intro q j active
    have hq : q.val<1 := q.isLt
    have eq : q=⟨0,by simp [programIO,memory,outputs]⟩ := Fin.ext (show q.val=0 by omega)
    subst q
    let e : Grid.Event p := ((j.1.val,j.2.1.val),(j.2.2.1,PUnit.unit))
    have live : e∈Grid.events p jobs := (Grid.mem_events p jobs e).mpr
      (order.mem_iff.mp ((Grid.mem_fullGrid B H e.1).mpr ⟨j.1.isLt,j.2.1.isLt⟩))
    change mapCell id (final.mem p.output (Grid.offset p e.1 e.2)) =
      Cell.mk .fp32 (orderedRow M R p (counts j.1) (probAt v j.1 j.2.1) (valAt v j.1 j.2.1) j.2.2.1)
    rw [finalMem,values e live]
    change Cell.mk .fp32 (outputValues M R p (Grid.state initial.mem e.1) e.2) = _
    apply congrArg (Cell.mk .fp32)
    apply source_values M R p B H counts v _ j.1 j.2.1
    exact call_inputs M R p B H counts jobs v domain (Grid.state initial.mem e.1) inputs j.1 j.2.1 rfl rfl
  · intro r o outside
    rw [finalMem]
    exact frame r o (outside_old M R p B H counts jobs valid v r o outside)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
