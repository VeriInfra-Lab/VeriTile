import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Interface
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar Kernel.Math
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem invocation_safe {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) (values : Inputs α c) (bounds : RegionName → Nat)
    (hb : (programIO M c plan hasBias useSoftplus lower upper).InBounds bounds values)
    (l : Launch) (hl : c.valid l) (s : State (Term α)) :
    (invocation M c plan hasBias useSoftplus lower upper l).Safe bounds s := by
  change FP.Structural.TraceSafe (engine (graphAlgebra M c plan)) bounds
    (kernel c hasBias useSoftplus) ((invocation M c plan hasBias useSoftplus lower upper l).initial s)
  rw [initial_eq]
  apply source_traceSafe c (primitives M c plan) sizes hasBias useSoftplus
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
    exact hb.2.1 ⟨0, by simp [programIO, memory, inputs]⟩
      (⟨l.batch,hl.1⟩,⟨l.chunk*c.chunk+j.val,ht⟩,⟨head c.layout l i.val,active.1⟩,PUnit.unit) trivial
  · intro enabled i hi
    exact hb.2.1 ⟨2, by simp [programIO, memory, inputs]⟩
      (⟨head c.layout l i.val,hi⟩,PUnit.unit) enabled
  · intro i hi
    exact hb.2.1 ⟨1, by simp [programIO, memory, inputs]⟩
      (⟨head c.layout l i.val,hi⟩,PUnit.unit) trivial
  · intro i j active
    rw [out_packed]
    exact hb.2.2.1 ⟨0, by simp [programIO, memory, outputs]⟩
      (⟨l.batch,hl.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,hl.2.1⟩,⟨j.val,active.2⟩,PUnit.unit) trivial
  · intro i j active
    rw [out_packed]
    exact hb.2.2.1 ⟨1, by simp [programIO, memory, outputs]⟩
      (⟨l.batch,hl.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,hl.2.1⟩,⟨j.val,active.2⟩,PUnit.unit) trivial

theorem schedule_safe {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) (values : Inputs α c) (bounds : RegionName → Nat)
    (hb : (programIO M c plan hasBias useSoftplus lower upper).InBounds bounds values)
    (s : State (Term α)) :
    ScheduleSafe bounds ((programIO M c plan hasBias useSoftplus lower upper).schedule values) s := by
  have go : ∀ ls : List Launch, (∀ l ∈ ls, c.valid l) → ∀ state,
      ScheduleSafe bounds (ls.map (invocation M c plan hasBias useSoftplus lower upper)) state := by
    intro ls valid
    induction ls with
    | nil => intro state; trivial
    | cons l ls ih =>
      intro state
      refine ⟨invocation_safe M c sizes plan hasBias useSoftplus lower upper values bounds hb
        l (valid l (by simp)) state, ?_⟩
      intro next _
      exact ih (fun q hq => valid q (by simp [hq])) next
  exact go c.grid (fun l hl => (c.mem_grid l).mp hl) s

theorem implements {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) :
    (programIO M c plan hasBias useSoftplus lower upper).Implements
      (nativeTarget M c plan hasBias useSoftplus lower upper) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    exact schedule_safe M c sizes plan hasBias useSoftplus lower upper values bounds hb s
  · intro values s hin
    obtain ⟨v, ran, hvalues, frame⟩ := Kernel.complete c (primitives M c plan) sizes
      hasBias useSoftplus (.input lower) (.input upper) s
    refine ⟨v, ?_, ?_, ?_⟩
    · change runSchedule (c.grid.map (invocation M c plan hasBias useSoftplus lower upper)) s = some v
      rw [schedule_eq]
      exact ran
    · intro o j _
      fin_cases o
      · change mapCell (interpret M) (v.mem "out"
          (packed c j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)) = _
        rw [(hvalues _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).1]
        exact congrArg (Cell.mk .real)
          (preparation M c plan hasBias useSoftplus lower upper values s hin
            _ _ _ _ j.1.isLt)
      · change mapCell (interpret M) (v.mem "sum"
          (packed c j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)) = _
        rw [(hvalues _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).2]
        have hj : j.2.2.2.1.val < c.BC := lt_of_lt_of_le j.2.2.2.1.isLt c.chunk_le_BC
        change Cell.mk .real (interpret M (logicalCumulative c (primitives M c plan)
          hasBias useSoftplus (.input lower) (.input upper) s j.1.val j.2.1.val j.2.2.1.val
            ⟨j.2.2.2.1.val,hj⟩)) = Cell.mk .real
          (if h : j.2.2.2.1.val < c.BC then cumulativeValue M c plan (logicalInputs c values)
            hasBias useSoftplus lower upper j.1.val j.2.1.val j.2.2.1.val ⟨j.2.2.2.1.val,h⟩ else default)
        rw [dif_pos hj]
        exact congrArg (Cell.mk .real) (cumulative M c plan hasBias useSoftplus lower upper values s hin
          j.1.val j.2.1.val j.2.2.1.val ⟨j.2.2.2.1.val,hj⟩ j.1.isLt j.2.1.isLt)
    · intro r offset outside
      apply frame r offset
      rintro ⟨l, hl, touched⟩
      have valid := (c.mem_grid l).mp hl
      rcases touched with ⟨hr,i,j,active,he⟩ | ⟨hr,i,j,active,he⟩
      · have ho := outside.1 ⟨0, by simp [programIO, memory, outputs]⟩
          (⟨l.batch,valid.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,valid.2.1⟩,
            ⟨j.val,active.2⟩,PUnit.unit) trivial
        exact ho.elim (fun h => h hr) (fun h => h (he.trans (out_packed c l i.val j.val)))
      · have ho := outside.1 ⟨1, by simp [programIO, memory, outputs]⟩
          (⟨l.batch,valid.1⟩,⟨head c.layout l i.val,active.1⟩,⟨l.chunk,valid.2.1⟩,
            ⟨j.val,active.2⟩,PUnit.unit) trivial
        exact ho.elim (fun h => h hr) (fun h => h (he.trans (out_packed c l i.val j.val)))

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface
