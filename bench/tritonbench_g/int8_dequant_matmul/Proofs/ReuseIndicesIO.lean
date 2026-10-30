import bench.tritonbench_g.int8_dequant_matmul.ReuseIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.ReuseIO
open VeriTile Triton FP.Structural
noncomputable section

theorem invocation_run (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c)
    (initial : State (Sum F H)) :
    (IO.invocation q s c hasBias divisor p).run initial =
      (invocation q s c hasBias divisor p).run initial := by
  change IntegerDot.exec (Quote.floating q) (Quote.SharedInteger.arithmetic q)
      (Quote.SharedInteger.context q s.M s.N) (Quote.source s c hasBias) _ =
    IntegerDot.exec (Quote.floating q) (Quote.SharedInteger.arithmetic q)
      (Quote.SharedInteger.context q s.M s.N) (ReuseIndices.source s c hasBias) _
  rw [Quote.SharedInteger.exec_eq,Quote.SharedInteger.exec_eq]
  exact (ReuseIndices.execute_eq (Quote.floating q) q s c hasBias _).symm

theorem invocation_safe (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c)
    (bounds : RegionName → Nat) (initial : State (Sum F H)) :
    (IO.invocation q s c hasBias divisor p).Safe bounds initial ↔
      (invocation q s c hasBias divisor p).Safe bounds initial :=
  (ReuseIndices.safe_iff (Quote.floating q) q s c hasBias
    ((IO.invocation q s c hasBias divisor p).initial initial) bounds).symm

/-- Both complete grids, the original independent split order at each output
address and every full typed cell/frame obligation are retained. -/
theorem equivalent (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (selected : Selection s c) (plans : q.plans=Quote.selectedPlan s c selected) :
    PairIO.EquivalentOnCells (IO.programIO q s c hasBias divisor schedule)
      (variant q s c hasBias divisor schedule) (IOLayout.domain q.ops s c) :=
  PairIO.EquivalentOnCells.of_run_safe (variant q s c hasBias divisor schedule) rfl
    (fun _ p initial=>invocation_run q s c hasBias divisor p initial)
    (fun _ p bounds initial=>invocation_safe q s c hasBias divisor p bounds initial)
    (PublicIO.generic q s c hasBias divisor schedule selected plans)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.ReuseIO
