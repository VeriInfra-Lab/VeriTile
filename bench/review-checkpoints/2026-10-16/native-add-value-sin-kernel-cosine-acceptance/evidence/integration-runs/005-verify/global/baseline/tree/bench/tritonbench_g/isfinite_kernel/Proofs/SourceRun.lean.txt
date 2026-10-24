import bench.tritonbench_g.isfinite_kernel.Proofs.BodyMemory
import bench.tritonbench_g.isfinite_kernel.Proofs.Composition

/-! Complete original writer execution from arbitrary initial states, for
every size, tile size, stride, pid, program count, iteration count and selected
branch in the audited source domain. The result records the exact sequence
of source stores; no alias or output-stride condition is needed for this
operational formula. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedBits BitClassification
noncomputable section

def sourceMemory {α : Type} [Inhabited α] (classify : α → Bool)
    (p : Parameters) (pid numCtas : Nat) (mem : RegionName → Nat → Cell α) :
    RegionName → Nat → Cell α :=
  if p.oneTile then tileMemory classify p (pid * p.block) mem
  else iterateMemory
    (fun j => tileMemory classify p ((pid + j * numCtas) * p.block))
    p.iterations 0 mem

theorem source_memory {α : Type} [Inhabited α] (M : Algebra α)
    (classify : α → Bool) (p : Parameters) (k : ComputeKernel)
    (one loopBody : List ComputeStmt)
    (hsplit : k.surfaceBody = before p ++ [.ifThenElse (.constBool p.oneTile) one
      [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
        (.constNat 1) loopBody]])
    (hone : ∀ pid s, OneReady pid s → ∃ t,
      FP.Structural.run M one s = some t ∧
      t.mem = tileMemory classify p (pid * p.block) s.mem)
    (hloop : ∀ pid numCtas j s, LoopReady pid numCtas s → ∃ t,
      FP.Structural.run M loopBody (s.setReg "j" .nat [] (fun _ => j)) = some t ∧
      LoopReady pid numCtas t ∧
      t.mem = tileMemory classify p ((pid + j * numCtas) * p.block) s.mem)
    (s : State α) :
    ∃ t, FP.Structural.exec M k s = some t ∧
      t.mem = sourceMemory classify p (s.pids 0) (s.numPids 0) s.mem := by
  cases hb : p.oneTile with
  | «true» =>
      obtain ⟨t, ht, hm⟩ := hone (s.pids 0) (beforeState p s) (before_ready p s)
      refine ⟨t, ?_, ?_⟩
      · exact exec_one M k (before p) one
          [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
            (.constNat 1) loopBody] s (beforeState p s) t
          (by simpa only [hb] using hsplit) (before_eval M p s) ht
      · simpa only [sourceMemory, hb, ↓reduceIte, before_memory] using hm
  | «false» =>
      obtain ⟨t, ht, _, hm⟩ := range_memory M "j" loopBody
        (LoopReady (s.pids 0) (s.numPids 0))
        (fun j => tileMemory classify p ((s.pids 0 + j * s.numPids 0) * p.block))
        (hloop (s.pids 0) (s.numPids 0)) p.iterations 0 (headerState p s)
        (header_ready p s)
      simp only [Nat.zero_add] at ht
      refine ⟨t, ?_, ?_⟩
      · exact exec_grid M k (before p) one numCtas loopBody p.iterations
          s (headerState p s) t (by simpa only [hb] using hsplit)
          (header_eval M p s) ht
      · simpa only [sourceMemory, hb, Bool.false_eq_true, if_false, header_memory] using hm

theorem original32_run (rest : Algebra Float32Bits) (p : Parameters)
    (s : State Float32Bits) :
    ∃ t, FP.Structural.exec (engine (backend32 rest)) (original32 p) s = some t ∧
      t.mem = sourceMemory finite32 p (s.pids 0) (s.numPids 0) s.mem := by
  exact source_memory (engine (backend32 rest)) finite32 p (original32 p)
    (oneBody32 p) (loopBody32 p) (source32_split p)
    (fun pid u h => ⟨(one32_run rest p pid u h).val,
      (one32_run rest p pid u h).property, one32_memory rest p pid u h⟩)
    (fun pid numCtas j u h => ⟨(loop32_run rest p pid numCtas j u h).val,
      (loop32_run rest p pid numCtas j u h).property,
      loop32_ready rest p pid numCtas j u h, loop32_memory rest p pid numCtas j u h⟩) s

theorem original64_run (rest : Algebra Float64Bits) (p : Parameters)
    (s : State Float64Bits) :
    ∃ t, FP.Structural.exec (engine (backend64 rest)) (original64 p) s = some t ∧
      t.mem = sourceMemory finite64 p (s.pids 0) (s.numPids 0) s.mem := by
  exact source_memory (engine (backend64 rest)) finite64 p (original64 p)
    (oneBody64 p) (loopBody64 p) (source64_split p)
    (fun pid u h => ⟨(one64_run rest p pid u h).val,
      (one64_run rest p pid u h).property, one64_memory rest p pid u h⟩)
    (fun pid numCtas j u h => ⟨(loop64_run rest p pid numCtas j u h).val,
      (loop64_run rest p pid numCtas j u h).property,
      loop64_ready rest p pid numCtas j u h, loop64_memory rest p pid numCtas j u h⟩) s

#axiomsClean source_memory
#axiomsClean original32_run
#axiomsClean original64_run
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
