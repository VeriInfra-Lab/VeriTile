import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Pair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.PairIO
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000
variable {α : Type} [Inhabited α]


/-- Source termination and exact exterior cells need no numerical formula law. -/
theorem schedule_frame (M : Algebra α) (ops : Operations α)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b)
    (ids : List Nat) (legal : ∀ pid∈ids,pid<g.problem.tiles) (s : State α) :
    ∃ final,runSchedule (ids.map (IO.invocation M ops g b)) s=some final ∧
      ∀ r o,ReadProjection.Outside g.problem (b.views g) r o → final.mem r o=s.mem r o := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons pid tail ih =>
    let start := (IO.invocation M ops g b pid).initial s
    have hp : start.pids 0=pid := by simp [start,Invocation.initial,IO.invocation,bindRegisters]
    have ht : start.pids 0<g.problem.tiles := hp ▸ legal pid (by simp)
    obtain ⟨next,ran,_,_,frame⟩ := programExecution M (selectedDot ops) g.problem (b.views g) start
      (active_injective g.problem (b.views g) L start ht)
    obtain ⟨final,hf,ff⟩ := ih (fun p hp => legal p (by simp [hp])) next
    refine ⟨final,?_,?_⟩
    · change (Invocation.run (IO.invocation M ops g b pid) s).bind
        (runSchedule (tail.map (IO.invocation M ops g b)))=some final
      rw [show Invocation.run (IO.invocation M ops g b pid) s=some next from ran]
      exact hf
    · intro r o ho
      exact (ff r o ho).trans (frame r o
        (ReadProjection.outside_program g.problem (b.views g) start ht r o ho))

/-- The headline judgment has the original input binding and logical windows,
both actual safe successful schedules, equal complete output cells and both
exact exterior frames, over the entire admitted word domain. -/
theorem equivalent (M : Algebra α) (ops : Operations α) (comm : Pair.MulCommutes M)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (L : KernelBuffers.Layout g b) :
    (IO.matmulIO M ops g b).Equivalent (variant M ops g b) := by
  intro vals _ bounds hb initial _
  have safe := IOImplements.safe M ops g b bounds
    (IOImplements.windows M ops g b vals bounds hb) (Source.tiles g.problem)
    (fun pid hp => (Source.tiles_exact g.problem pid).mp hp) initial
  obtain ⟨final,ran,frame⟩ := schedule_frame M ops g b L (Source.tiles g.problem)
    (fun pid hp => (Source.tiles_exact g.problem pid).mp hp) initial
  have ioFrame : (IO.matmulIO M ops g b).Frame vals initial final := by
    intro r o ho
    exact frame r o (IOImplements.outside M ops g b vals r o ho)
  refine ⟨safe,(Pair.schedule_safety_equal M ops comm g b _ bounds initial).mp safe,
    final,final,ran,(Pair.schedule_equal M ops comm g b _ initial).symm.trans ran,
    ?_,ioFrame,ioFrame⟩
  intro o j _
  rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.PairIO
