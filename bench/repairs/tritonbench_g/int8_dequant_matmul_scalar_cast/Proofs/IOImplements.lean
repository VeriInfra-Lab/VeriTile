import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOTrace

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOImplements
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

/-- Compose proved complete-CTA safety/success with actual source locality,
coverage and cell transitions. The safety argument is discharged by IOSafety
in the public adapters; it is not a numerical or public theorem assumption. -/
theorem assemble (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (selected : Selection s c) (plans : q.plans=Quote.selectedPlan s c selected)
    (allPrograms : ∀ vals base,
      (IO.programIO q s c hasBias divisor schedule).InputCells vals base →
      (IO.programIO q s c hasBias divisor schedule).Programs vals base) :
    (IO.programIO q s c hasBias divisor schedule).ImplementsOnCells (IOLayout.domain q.ops s c)
      (IOTrace.result q s c hasBias divisor schedule selected) := by
  intro vals zero base cells initial output
  let io:=IO.programIO q s c hasBias divisor schedule
  have programs:=allPrograms vals base cells
  let words : ChannelValues (Sum F H) io.outputs:=
    Fin.cases (fun j=>Sum.inr (IOTrace.result q s c hasBias divisor schedule selected vals 0 j))
      (fun i=>Fin.elim0 i)
  let finals : io.RawOutputs:=fun key=>words key.1 key.2
  refine ⟨IOEvents.schedule_complete q s c hasBias divisor schedule vals,programs,
    IOEvents.locality q s c hasBias divisor schedule selected plans vals base cells,
    IOEvents.coverage q s c hasBias divisor schedule selected plans vals base cells,
    finals,?_,?_,?_⟩
  · intro key _
    rcases key with ⟨o,j⟩
    have ho : o=0 := Fin.eq_zero o
    subst o
    obtain ⟨r,n,u⟩:=j
    cases u
    let x : MatrixCell s:=(r,n)
    have trace:=IOTrace.trace q s c hasBias divisor schedule selected plans vals base cells programs
      initial x (projection x schedule.events) (initial ⟨0,(r,n,PUnit.unit)⟩)
    have nonempty : projection x schedule.events≠[] := by
      intro empty
      have length:=(schedule.complete x).length_eq
      rw [empty] at length
      simp only [List.length_nil,List.length_finRange] at length
      have positive:=c.positiveS
      omega
    rw [IOTrace.fold_nonempty _ _ _ _ nonempty] at trace
    have seed : cellFold q.ops c
        (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x)
        (projection x schedule.events) (Quote.asH q.ops (initial ⟨0,(r,n,PUnit.unit)⟩))=
      cellFold q.ops c
        (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x)
        (projection x schedule.events) q.ops.hzero := by
      by_cases single : c.S=1
      · exact cellFold_seed _ _ _ _ (schedule.complete x) single _ _
      · rw [IOTrace.initial_zero q s c hasBias divisor schedule vals base cells initial output zero single x]
        rfl
    rw [seed] at trace
    exact trace
  · intro o j _
    have ho : o=0 := Fin.eq_zero o
    subst o
    obtain ⟨r,n,u⟩:=j
    cases u
    change mapCell (Quote.asH q.ops)
      (((IOLayout.memory F H s c hasBias).patchOutputs vals finals base).mem "C"
        (TileShape.linearIndex [s.M,s.N] (r,n,PUnit.unit))) = _
    simp only [TileShape.linearIndex,TileShape.product,Nat.mul_one,Nat.add_zero]
    rw [IOEvents.patch_cell s c hasBias vals base finals (r,n)]
    rfl
  · exact io.toMemoryIO.patchOutputs_frame vals finals base

#axiomsClean assemble
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOImplements
