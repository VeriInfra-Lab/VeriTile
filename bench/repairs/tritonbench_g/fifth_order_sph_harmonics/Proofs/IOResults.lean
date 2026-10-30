import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOSchedule

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOResults
open VeriTile.Triton FP.Structural HarmonicIO
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

/-- Assemble the old complete-grid result with logical channel bindings. The
scalar equation is proved separately for Real and the concrete native model. -/
theorem forward (M : Algebra α) (p : Parameters)
    (fit : p.columnOffset + 11 ≤ p.outputWidth)
    (f : (TileIndex [p.rows, 3] → α) → TileIndex [p.rows, 11] → α)
    (scalar : ∀ xs (i : TileIndex [p.rows, 11]),
      Reference.forwardValues M (M.fp32Load (xs (i.1, 0, ())))
        (M.fp32Load (xs (i.1, 1, ()))) (M.fp32Load (xs (i.1, 2, ()))) i.2.1 = f xs i)
    (vals : (forwardIO M p).Inputs) (s : State α)
    (binding : (forwardIO M p).InputBinding vals s) :
    ∃ final, runSchedule ((forwardIO M p).schedule vals) s = some final ∧
      (forwardIO M p).OutputBinding vals (forwardTarget M p f vals) final ∧
      (forwardIO M p).Frame vals s final := by
  have coords (r : Fin p.rows) (c : Fin 3) :
      (s.mem coordinateRegion (r.val*3+c.val)).read .real = vals (0 : Fin 1) (r, c, ()) :=
    binding (0 : Fin 1) (r, c, ()) trivial
  have x (r : Fin p.rows) : (s.mem coordinateRegion (r.val*3)).read .real = vals (0 : Fin 1) (r, 0, ()) := coords r 0
  have y (r : Fin p.rows) : (s.mem coordinateRegion (r.val*3+1)).read .real = vals (0 : Fin 1) (r, 1, ()) := coords r 1
  have z (r : Fin p.rows) : (s.mem coordinateRegion (r.val*3+2)).read .real = vals (0 : Fin 1) (r, 2, ()) := coords r 2
  obtain ⟨final, executed, values, frame⟩ := Grid.launch_correct M .forward p s fit
  simp only [Grid.output, Grid.offset, Grid.stride, Grid.first] at values
  refine ⟨final, ?_, ?_, ?_⟩
  · exact (IOSchedule.launch_run M .forward p s).trans executed
  · intro port i _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    change mapCell id (final.mem outputRegion (i.1.val*p.outputWidth+p.columnOffset+i.2.1.val)) =
      Cell.mk .real (f (vals (0 : Fin 1)) i)
    rw [values i.1.val i.1.isLt i.2.1]
    change Cell.mk .real (Grid.value M .forward p s i.1.val i.2.1) = _
    congr 1
    simpa only [Grid.value, x, y, z]
      using scalar (vals (0 : Fin 1)) i
  · intro region offset outside
    apply frame
    by_cases same : region = outputRegion
    · right
      intro r hr c
      have h := outside.1 (0 : Fin 1) (⟨r, hr⟩, c, ()) trivial
      exact h.resolve_left (by simpa using same)
    · exact Or.inl same

theorem backward (M : Algebra α) (p : Parameters)
    (fit : p.columnOffset + 11 ≤ p.outputWidth)
    (f : (TileIndex [p.rows, 3] → α) → (TileIndex [p.rows, 11] → α) →
      (TileIndex [p.rows, 3] → α) → TileIndex [p.rows, 3] → α)
    (domain : (backwardIO M p).Inputs → Prop)
    (scalar : ∀ vals, domain vals → ∀ i : TileIndex [p.rows, 3],
      Reference.backwardValues M
        (M.fp32Load (vals (0 : Fin 3) (i.1, 0, ())))
        (M.fp32Load (vals (0 : Fin 3) (i.1, 1, ())))
        (M.fp32Load (vals (0 : Fin 3) (i.1, 2, ())))
        (fun c => M.fp32Load (vals (1 : Fin 3) (i.1, c, ())))
        (fun c => M.fp32Load (vals (2 : Fin 3) (i.1, c, ()))) i.2.1 =
      f (vals (0 : Fin 3)) (vals (1 : Fin 3)) (vals (2 : Fin 3)) i)
    (vals : (backwardIO M p).Inputs) (admitted : domain vals) (s : State α)
    (binding : (backwardIO M p).InputBinding vals s) :
    ∃ final, runSchedule ((backwardIO M p).schedule vals) s = some final ∧
      (backwardIO M p).OutputBinding vals (backwardTarget M p f vals) final ∧
      (backwardIO M p).Frame vals s final := by
  have coords (r : Fin p.rows) (c : Fin 3) :
      (s.mem coordinateRegion (r.val*3+c.val)).read .real = vals (0 : Fin 3) (r, c, ()) :=
    binding (0 : Fin 3) (r, c, ()) trivial
  have x (r : Fin p.rows) : (s.mem coordinateRegion (r.val*3)).read .real = vals (0 : Fin 3) (r, 0, ()) := coords r 0
  have y (r : Fin p.rows) : (s.mem coordinateRegion (r.val*3+1)).read .real = vals (0 : Fin 3) (r, 1, ()) := coords r 1
  have z (r : Fin p.rows) : (s.mem coordinateRegion (r.val*3+2)).read .real = vals (0 : Fin 3) (r, 2, ()) := coords r 2
  have upstream (r : Fin p.rows) (c : Fin 11) :
      (s.mem upstreamRegion (r.val*p.upstreamRowStride+(p.columnOffset+c.val)*p.upstreamColStride)).read .real =
        vals (1 : Fin 3) (r, c, ()) := binding (1 : Fin 3) (r, c, ()) trivial
  have old (r : Fin p.rows) (c : Fin 3) :
      (s.mem gradientRegion (r.val*3+c.val)).read .real = vals (2 : Fin 3) (r, c, ()) :=
    binding (2 : Fin 3) (r, c, ()) trivial
  obtain ⟨final, executed, values, frame⟩ := Grid.launch_correct M .backward p s fit
  simp only [Grid.output, Grid.offset, Grid.stride, Grid.first, Nat.add_zero] at values
  refine ⟨final, ?_, ?_, ?_⟩
  · exact (IOSchedule.launch_run M .backward p s).trans executed
  · intro port i _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    change mapCell id (final.mem gradientRegion (i.1.val*3+i.2.1.val)) =
      Cell.mk .real (f (vals (0 : Fin 3)) (vals (1 : Fin 3)) (vals (2 : Fin 3)) i)
    rw [values i.1.val i.1.isLt i.2.1]
    change Cell.mk .real (Grid.value M .backward p s i.1.val i.2.1) = _
    congr 1
    simpa only [Grid.value, Readback.backwardExpected, x, y, z, upstream, old]
      using scalar vals admitted i
  · intro region offset outside
    apply frame
    by_cases same : region = gradientRegion
    · right
      intro r hr c
      have h := outside.1 (0 : Fin 1) (⟨r, hr⟩, c, ()) trivial
      exact h.resolve_left (by simpa using same)
    · exact Or.inl same

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOResults
