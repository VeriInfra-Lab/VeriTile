import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOInvocation

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOEvents
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

theorem schedule_complete (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (vals : (IO.programIO q s c hasBias divisor schedule).Inputs) :
    (IO.programIO q s c hasBias divisor schedule).ScheduleComplete vals := by
  refine ⟨rfl,IOGeometry.output_unique F H s c hasBias vals,?_⟩
  intro key _
  rcases key with ⟨o,j⟩
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨r,n,u⟩:=j
  cases u
  constructor
  · exact IOGeometry.order_complete s c schedule (r,n)
  · convert IOGeometry.events_partition s c (r,n) using 1
    congr 1
    apply List.filter_congr
    intro event _
    simp
    rfl

theorem patch_cell {F H : Type} (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base : State (Sum F H))
    (current : (IOLayout.memory F H s c hasBias).RawOutputs (Sum F H)) (x : MatrixCell s) :
    ((IOLayout.memory F H s c hasBias).patchOutputs vals current base).mem "C" (x.1.val*s.N+x.2.val)=
      Cell.mk .fp16 (current ⟨0,(x.1,x.2,PUnit.unit)⟩) := by
  have result:=(IOLayout.memory F H s c hasBias).patchOutputs_read vals
    (IOGeometry.output_unique F H s c hasBias vals) current base ⟨0,(x.1,x.2,PUnit.unit)⟩ trivial
  change ((IOLayout.memory F H s c hasBias).patchOutputs vals current base).mem "C"
    (TileShape.linearIndex [s.M,s.N] (x.1,x.2,PUnit.unit))=
      Cell.mk .fp16 (current ⟨0,(x.1,x.2,PUnit.unit)⟩) at result
  simpa [TileShape.linearIndex] using result

theorem locality (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (selected : Selection s c) (plans : q.plans=Quote.selectedPlan s c selected)
    (vals : (IO.programIO q s c hasBias divisor schedule).Inputs) (base : State (Sum F H))
    (cells : (IO.programIO q s c hasBias divisor schedule).InputCells vals base) :
    (IO.programIO q s c hasBias divisor schedule).SourceLocality vals base := by
  intro event _ key _ left right same l r ranL ranR
  rcases key with ⟨o,j⟩
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨row,column,u⟩:=j
  cases u
  have bound:=IODataBinding.from_input_cells s c hasBias vals base cells
  have lv:=IOInvocation.cell q s c hasBias divisor event vals _ l
    (IODataBinding.patch s c hasBias vals base bound left) selected plans ranL (row,column)
  have rv:=IOInvocation.cell q s c hasBias divisor event vals _ r
    (IODataBinding.patch s c hasBias vals base bound right) selected plans ranR (row,column)
  change l.mem "C" (TileShape.linearIndex [s.M,s.N] (row,column,PUnit.unit))=
    r.mem "C" (TileShape.linearIndex [s.M,s.N] (row,column,PUnit.unit))
  simp only [TileShape.linearIndex,TileShape.product,Nat.mul_one,Nat.add_zero]
  rw [lv,rv,patch_cell,patch_cell,same]

theorem coverage (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (selected : Selection s c) (plans : q.plans=Quote.selectedPlan s c selected)
    (vals : (IO.programIO q s c hasBias divisor schedule).Inputs) (base : State (Sum F H))
    (cells : (IO.programIO q s c hasBias divisor schedule).InputCells vals base) :
    (IO.programIO q s c hasBias divisor schedule).SourceCoverage vals base := by
  intro key _ event _ absent current final ran
  rcases key with ⟨o,j⟩
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨row,column,u⟩:=j
  cases u
  have bound:=IODataBinding.from_input_cells s c hasBias vals base cells
  have value:=IOInvocation.cell q s c hasBias divisor event vals _ final
    (IODataBinding.patch s c hasBias vals base bound current) selected plans ran (row,column)
  have other : event.1.val≠(IOPrograms.owner s c (row,column)).val := by
    intro same
    exact absent ((IOGeometry.events_mem s c (row,column) event).mpr (Fin.ext same))
  rw [if_neg other,patch_cell] at value
  change final.mem "C" (TileShape.linearIndex [s.M,s.N] (row,column,PUnit.unit))=
    Cell.mk .fp16 (current ⟨0,(row,column,PUnit.unit)⟩)
  simpa [TileShape.linearIndex] using value

#axiomsClean schedule_complete
#axiomsClean patch_cell
#axiomsClean locality
#axiomsClean coverage
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOEvents
