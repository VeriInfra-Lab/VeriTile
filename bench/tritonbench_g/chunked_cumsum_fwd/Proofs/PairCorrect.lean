import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.PairIO
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.IOCorrect
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.PairSafety

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairCorrect
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar Kernel.Math Interface PairSource PairSemantics PairIO
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

 theorem candidate_initial (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias useSoftplus : Bool) (lower upper : α) (l : Launch) (s : State (Term α)) :
    (candidateInvocation M c plan hasBias useSoftplus lower upper l).initial s =
      tileState l (bindBounds s (.input lower) (.input upper)) := rfl

 theorem candidate_run (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias useSoftplus : Bool) (lower upper : α) (l : Launch) (s : State (Term α)) :
    (candidateInvocation M c plan hasBias useSoftplus lower upper l).run s =
      exec (engine (reverseMul (graphAlgebra M c plan))) (kernel c hasBias useSoftplus)
        (tileState l (bindBounds s (.input lower) (.input upper))) := by
  change exec (engine (graphAlgebra M c plan)) (candidateKernel c hasBias useSoftplus)
    ((candidateInvocation M c plan hasBias useSoftplus lower upper l).initial s) = _
  rw [candidate_initial]
  exact (PairSource.complete_run (graphAlgebra M c plan) c hasBias useSoftplus _).symm

 theorem candidate_schedule (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias useSoftplus : Bool) (lower upper : α) (ls : List Launch) (s : State (Term α)) :
    runSchedule (ls.map (candidateInvocation M c plan hasBias useSoftplus lower upper)) s =
      runTiles (reverseMul (graphAlgebra M c plan)) c hasBias useSoftplus
        (.input lower) (.input upper) ls s := by
  induction ls generalizing s with
  | nil => rfl
  | cons l ls ih =>
    simp only [List.map_cons, runSchedule, candidate_run, runTiles]
    exact congrArg
      (fun next => (exec (engine (reverseMul (graphAlgebra M c plan)))
        (kernel c hasBias useSoftplus)
        (tileState l (bindBounds s (.input lower) (.input upper)))).bind next)
      (funext ih)

 theorem candidate_safe {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) (values : Inputs α c) (bounds : RegionName → Nat)
    (hb : (programIO M c plan hasBias useSoftplus lower upper).InBounds bounds values)
    (l : Launch) (hl : c.valid l) (s : State (Term α)) :
    (candidateInvocation M c plan hasBias useSoftplus lower upper l).Safe bounds s := by
  change TraceSafe (engine (graphAlgebra M c plan)) bounds
    (candidateKernel c hasBias useSoftplus)
    ((candidateInvocation M c plan hasBias useSoftplus lower upper l).initial s)
  rw [candidate_initial]
  apply (PairSource.complete_safe (graphAlgebra M c plan) c hasBias useSoftplus bounds _).mp
  apply source_traceSafe c (reversePrimitives c (primitives M c plan)) sizes hasBias useSoftplus
    (.input lower) (.input upper) l hl s bounds
  · intro i j active
    have ht : l.chunk*c.chunk+j.val < c.sequence := by
      have hs := chunk_start_lt c l hl
      change head c.layout l i.val < c.heads ∧
        j.val < min c.chunk (c.sequence-l.chunk*c.chunk) at active
      omega
    have eq : dtAddress c.layout l i.val j.val =
        l.batch*c.dtBatch+(l.chunk*c.chunk+j.val)*c.dtTime+head c.layout l i.val*c.dtHead := by
      simp [dtAddress, Geometry.layout]
      ring
    rw [eq]
    exact hb.2.1 ⟨0, by simp [programIO, memory, Interface.inputs]⟩
      (⟨l.batch,hl.1⟩,⟨l.chunk*c.chunk+j.val,ht⟩,⟨head c.layout l i.val,active.1⟩,PUnit.unit) trivial
  · intro enabled i hi
    exact hb.2.1 ⟨2, by simp [programIO, memory, Interface.inputs]⟩
      (⟨head c.layout l i.val,hi⟩,PUnit.unit) enabled
  · intro i hi
    exact hb.2.1 ⟨1, by simp [programIO, memory, Interface.inputs]⟩
      (⟨head c.layout l i.val,hi⟩,PUnit.unit) trivial
  · intro i j active
    rw [out_packed]
    exact hb.2.2.1 ⟨0, by simp [programIO, memory, Interface.outputs]⟩
      (⟨l.batch,hl.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,hl.2.1⟩,⟨j.val,active.2⟩,PUnit.unit) trivial
  · intro i j active
    rw [out_packed]
    exact hb.2.2.1 ⟨1, by simp [programIO, memory, Interface.outputs]⟩
      (⟨l.batch,hl.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,hl.2.1⟩,⟨j.val,active.2⟩,PUnit.unit) trivial

 theorem candidate_schedule_safe {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) (values : Inputs α c) (bounds : RegionName → Nat)
    (hb : (programIO M c plan hasBias useSoftplus lower upper).InBounds bounds values)
    (s : State (Term α)) :
    ScheduleSafe bounds
      (c.grid.map (candidateInvocation M c plan hasBias useSoftplus lower upper)) s := by
  have go : ∀ ls : List Launch, (∀ l ∈ ls, c.valid l) → ∀ state,
      ScheduleSafe bounds (ls.map (candidateInvocation M c plan hasBias useSoftplus lower upper)) state := by
    intro ls valid
    induction ls with
    | nil => intro state; trivial
    | cons l ls ih =>
      intro state
      refine ⟨candidate_safe M c sizes plan hasBias useSoftplus lower upper values bounds hb
        l (valid l (by simp)) state, ?_⟩
      intro next _
      exact ih (fun q hq => valid q (by simp [hq])) next
  exact go c.grid (fun l hl => (c.mem_grid l).mp hl) s

/-- Complete natural candidate execution, safety, both output arrays and full
exterior frame on every input word and independent runtime clamp bounds. -/
theorem candidate_implements (cfg : FP.Config) (base : NativeModel (FP.Value .fp32))
    (c : Geometry) (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC)
    (hasBias useSoftplus : Bool) (lower upper : FP.Value .fp32) :
    (scaleFirstIO cfg base c plan hasBias useSoftplus lower upper).toProgramIO.Implements
      (nativeTarget (concreteModel cfg base) c plan hasBias useSoftplus lower upper) := by
  let M := concreteModel cfg base
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    exact candidate_schedule_safe M c sizes plan hasBias useSoftplus lower upper values bounds hb s
  · intro values s hin
    obtain ⟨v, ran, hvalues, frame⟩ := Kernel.complete c
      (reversePrimitives c (primitives M c plan)) sizes
      hasBias useSoftplus (.input lower) (.input upper) s
    refine ⟨v, ?_, ?_, ?_⟩
    · change runSchedule
        (c.grid.map (candidateInvocation M c plan hasBias useSoftplus lower upper)) s = some v
      rw [candidate_schedule]
      exact ran
    · intro o j _
      fin_cases o
      · change mapCell (interpret M) (v.mem "out"
          (packed c j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)) = _
        rw [(hvalues _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).1]
        change Cell.mk .real (interpret M (logicalPrepared c
          (reversePrimitives c (primitives M c plan)) hasBias useSoftplus
          (.input lower) (.input upper) s j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)) = _
        rw [reverse_prepared]
        exact congrArg (Cell.mk .real)
          (preparation M c plan hasBias useSoftplus lower upper values s hin
            _ _ _ _ j.1.isLt)
      · change mapCell (interpret M) (v.mem "sum"
          (packed c j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)) = _
        rw [(hvalues _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).2]
        have hj : j.2.2.2.1.val < c.BC := lt_of_lt_of_le j.2.2.2.1.isLt c.chunk_le_BC
        change Cell.mk .real (interpret M (logicalCumulative c
          (reversePrimitives c (primitives M c plan)) hasBias useSoftplus
          (.input lower) (.input upper) s j.1.val j.2.1.val j.2.2.1.val
            ⟨j.2.2.2.1.val,hj⟩)) = Cell.mk .real
          (if h : j.2.2.2.1.val < c.BC then cumulativeValue M c plan (Interface.logicalInputs c values)
            hasBias useSoftplus lower upper j.1.val j.2.1.val j.2.2.1.val ⟨j.2.2.2.1.val,h⟩ else default)
        rw [dif_pos hj]
        exact congrArg (Cell.mk .real)
          ((reverse_cumulative cfg base c plan hasBias useSoftplus (.input lower) (.input upper)
            s j.1.val j.2.1.val j.2.2.1.val ⟨j.2.2.2.1.val,hj⟩).trans
            (cumulative M c plan hasBias useSoftplus lower upper values s hin
              j.1.val j.2.1.val j.2.2.1.val ⟨j.2.2.2.1.val,hj⟩ j.1.isLt j.2.1.isLt))
    · intro r offset outside
      apply frame r offset
      rintro ⟨l, hl, touched⟩
      have valid := (c.mem_grid l).mp hl
      rcases touched with ⟨hr,i,j,active,he⟩ | ⟨hr,i,j,active,he⟩
      · have ho := outside.1 ⟨0, by change 0 < 2; decide⟩
          (⟨l.batch,valid.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,valid.2.1⟩,
            ⟨j.val,active.2⟩,PUnit.unit) trivial
        exact ho.elim (fun h => h hr) (fun h => h (he.trans (out_packed c l i.val j.val)))
      · have ho := outside.1 ⟨1, by change 1 < 2; decide⟩
          (⟨l.batch,valid.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,valid.2.1⟩,
            ⟨j.val,active.2⟩,PUnit.unit) trivial
        exact ho.elim (fun h => h hr) (fun h => h (he.trans (out_packed c l i.val j.val)))


theorem equivalent (cfg : FP.Config) (M : NativeModel (FP.Value .fp32)) (c : Geometry)
    (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : FP.Value .fp32) :
    (originalIO cfg M c plan hasBias useSoftplus lower upper).Equivalent
      (scaleFirstIO cfg M c plan hasBias useSoftplus lower upper) :=
  ProgramIO.EquivalentOn.of_implementsOn
    (Interface.implements (concreteModel cfg M) c sizes plan hasBias useSoftplus lower upper)
    (candidate_implements cfg M c sizes plan hasBias useSoftplus lower upper)

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairCorrect
