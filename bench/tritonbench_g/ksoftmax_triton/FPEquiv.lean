import bench.tritonbench_g.ksoftmax_triton.Proofs.ScheduleEquivalence

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped VeriTile.Spec
noncomputable section

abbrev original := @ScheduleEquivalence.original
abbrev reversed := @ScheduleEquivalence.reversed
abbrev assumptions := @ScheduleEquivalence.assumptions

/-- Two complete actual source schedules differ only in the order of
independent rows. Source FP32 precision, masks, full grids, access safety,
success, observations and exterior frames are retained, even when K=0. -/
specification reversed_schedule_equivalent {A : Type} [Inhabited A]
    (model : Algebra A) (op : Operation) (B Q K D : Nat) (hD : 0<D) (covered : K≤D) :
    original model op B Q K D ≡[assumptions model op B Q K D] reversed model op B Q K D :=
  ScheduleEquivalence.equivalent model op B Q K D hD covered

#print_fp_assumptions reversed_schedule_equivalent
#axiomsClean reversed_schedule_equivalent
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.FPEquiv
