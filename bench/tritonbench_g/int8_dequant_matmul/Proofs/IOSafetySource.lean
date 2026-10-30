import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSafetyFactors
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSafetyRange
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSafetyEpilogue

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000

/-- Every actual source access is bounded, for arbitrary payloads in every
region, including arbitrary and per-address-patched output C. Missing unrelated
registers cannot create accesses: safety follows only successful source steps. -/
theorem source_safe_with_algebra {α : Type} [Inhabited α]
    (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c))
    (hz : initial.pids 1<c.S) (bounds : RegionName → Nat)
    (hb : Bounds s hasBias bounds) :
    IntegerDot.TraceSafe M (SharedInteger.arithmetic q) (SharedInteger.context q s.M s.N)
      bounds (Quote.source s c hasBias) initial := by
  unfold IntegerDot.TraceSafe
  rw [source_decomposition,List.append_assoc,IntegerDot.traceSafeList_append]
  refine ⟨prelude_safe M q s c hasBias initial valid bounds hb,?_⟩
  intro base executed
  rw [SharedInteger.run_eq,(preludeExecution M q s c hasBias initial).property] at executed
  cases Option.some.inj executed
  have init := preludeContext M q s c hasBias initial
  rw [List.singleton_append,IntegerDot.traceSafeList_cons]
  constructor
  · rw [IntegerDot.stepSafe_unfold]
    exact range_safe M q s c hasBias _ bounds
      (a_bound M q s c hasBias initial valid hz bounds hb)
      (b_bound M q s c hasBias initial valid hz bounds hb)
      0 _ (fun _=>0) init.1 init.2
  · intro current executed
    simp only [IntegerDot.step_unfold] at executed
    rw [SharedInteger.range_eq] at executed
    obtain ⟨final,ran,ctx,_⟩ := rangeExecution M q s c hasBias _ _ 0 (steps s c)
      (Nat.zero_le _) (fun _=>0) init.1 init.2
    rw [ran] at executed
    cases Option.some.inj executed
    exact epilogue_safe M q s c hasBias _ _ bounds ctx
      (bias_bound M q s c hasBias initial valid hz bounds hb)
      (output_bound M q s c hasBias initial valid hz bounds hb)

theorem source_safe (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State (Sum F H))
    (valid : initial.pids 0<Geometry.count (geometry s c))
    (hz : initial.pids 1<c.S) (bounds : RegionName → Nat)
    (hb : Bounds s hasBias bounds) :
    IntegerDot.TraceSafe (Quote.floating q) (SharedInteger.arithmetic q)
      (SharedInteger.context q s.M s.N) bounds (Quote.source s c hasBias) initial :=
  source_safe_with_algebra (Quote.floating q) q s c hasBias initial valid hz bounds hb

theorem source_safe_of_io (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State (Sum F H))
    (valid : initial.pids 0<Geometry.count (geometry s c))
    (hz : initial.pids 1<c.S) (bounds : RegionName → Nat)
    (vals : (IOLayout.memory F H s c hasBias).Inputs)
    (hb : (IOLayout.memory F H s c hasBias).InBounds bounds vals) :
    IntegerDot.TraceSafe (Quote.floating q) (SharedInteger.arithmetic q)
      (SharedInteger.context q s.M s.N) bounds (Quote.source s c hasBias) initial :=
  source_safe q s c hasBias initial valid hz bounds (bounds_of_io F H s c hasBias vals bounds hb)

#axiomsClean source_safe_with_algebra
#axiomsClean source_safe
#axiomsClean source_safe_of_io
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
