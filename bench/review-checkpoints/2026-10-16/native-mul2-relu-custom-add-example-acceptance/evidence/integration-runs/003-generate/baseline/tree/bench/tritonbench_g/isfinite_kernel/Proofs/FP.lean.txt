import bench.tritonbench_g.isfinite_kernel.Proofs.FPContext
import bench.tritonbench_g.isfinite_kernel.Proofs.BitCorrect

/-! Full typed source/candidate equivalence, plus actual successful pairs and
bit-classification contracts. The equality theorem itself has no numerical
assumptions or successful-run premise and preserves complete final states. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedBits BitClassification
noncomputable section

theorem loop32_equiv (p : Parameters) {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) :
    FP.Structural.run M (loopBody32 p) s = FP.Structural.run M (candidateLoop32 p) s := by
  rw [loop32_split, candidateLoop32_split]
  exact run_cons_congr M _ _ _ (tileId_step M) s

theorem loop64_equiv (p : Parameters) {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) :
    FP.Structural.run M (loopBody64 p) s = FP.Structural.run M (candidateLoop64 p) s := by
  rw [loop64_split, candidateLoop64_split]
  exact run_cons_congr M _ _ _ (tileId_step M) s

theorem whole32 (p : Parameters) {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) :
    FP.Structural.exec M (original32 p) s = FP.Structural.exec M (candidate32 p) s :=
  source_context_congr M p (original32 p) (candidate32 p) (oneBody32 p)
    (loopBody32 p) (candidateLoop32 p) (source32_split p) (candidate32_split p)
    (loop32_equiv p M) s

theorem whole64 (p : Parameters) {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) :
    FP.Structural.exec M (original64 p) s = FP.Structural.exec M (candidate64 p) s :=
  source_context_congr M p (original64 p) (candidate64 p) (oneBody64 p)
    (loopBody64 p) (candidateLoop64 p) (source64_split p) (candidate64_split p)
    (loop64_equiv p M) s

theorem pair32_run (rest : Algebra Float32Bits) (p : Parameters)
    (s : State Float32Bits) :
    ∃ t, FP.Structural.exec (engine (backend32 rest)) (original32 p) s = some t ∧
      FP.Structural.exec (engine (backend32 rest)) (candidate32 p) s = some t ∧
      t.mem = sourceMemory finite32 p (s.pids 0) (s.numPids 0) s.mem := by
  obtain ⟨t, he, hm⟩ := original32_run rest p s
  exact ⟨t, he, (whole32 p _ s).symm.trans he, hm⟩

theorem pair64_run (rest : Algebra Float64Bits) (p : Parameters)
    (s : State Float64Bits) :
    ∃ t, FP.Structural.exec (engine (backend64 rest)) (original64 p) s = some t ∧
      FP.Structural.exec (engine (backend64 rest)) (candidate64 p) s = some t ∧
      t.mem = sourceMemory finite64 p (s.pids 0) (s.numPids 0) s.mem := by
  obtain ⟨t, he, hm⟩ := original64_run rest p s
  exact ⟨t, he, (whole64 p _ s).symm.trans he, hm⟩

theorem candidate32_correct (rest : Algebra Float32Bits) (p : Parameters)
    (bits : Nat → Float32Bits) (initial : State Float32Bits)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (hinput : InputBits p bits initial) :
    ∃ final, FP.Structural.exec (engine (backend32 rest)) (candidate32 p) initial = some final ∧
      Post finite32 p bits initial final := by
  obtain ⟨final, he, hp⟩ := original32_correct rest p bits initial hne hstride hinput
  exact ⟨final, (whole32 p _ initial).symm.trans he, hp⟩

theorem candidate64_correct (rest : Algebra Float64Bits) (p : Parameters)
    (bits : Nat → Float64Bits) (initial : State Float64Bits)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (hinput : InputBits p bits initial) :
    ∃ final, FP.Structural.exec (engine (backend64 rest)) (candidate64 p) initial = some final ∧
      Post finite64 p bits initial final := by
  obtain ⟨final, he, hp⟩ := original64_correct rest p bits initial hne hstride hinput
  exact ⟨final, (whole64 p _ initial).symm.trans he, hp⟩

#axiomsClean loop32_equiv
#axiomsClean loop64_equiv
#axiomsClean whole32
#axiomsClean whole64
#axiomsClean pair32_run
#axiomsClean pair64_run
#axiomsClean candidate32_correct
#axiomsClean candidate64_correct
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
