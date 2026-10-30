import bench.repairs.tritonbench_g.softmax_optimize.Proofs.SourceCandidate
import bench.tritonbench_g.softmax_optimize.Proofs.FP

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.KernelRewrite
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def program (k : ComputeKernel) : Program := ⟨k,[]⟩

/-- A genuine source candidate: exact integer commutation at each retained
loop. The complete floating program remains in its original context. -/
theorem equivalent (Y X : RegionName) (M N B : Nat) :
    program (kernel Y X M N B) ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      program (indexCommutedKernel Y X M N B) := by
  equiv_decompose <;> fp_prove

theorem execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (Y X : RegionName) (M N B : Nat) (s : FP.Structural.State A) :
    FP.Structural.exec (engine algebra) (kernel Y X M N B) s=
      FP.Structural.exec (engine algebra) (indexCommutedKernel Y X M N B) s := by
  rw [ReadableSource.candidate_preserved]
  have loop_step (t : FP.Structural.State A) (index : RegName)
      (start stop : Op .int []) (stride : Op .nat []) (init : ComputeStmt)
      (rest : List ComputeStmt) (a b : Op .int []) :
      FP.Structural.step (engine algebra)
        (.forRangeInt index start stop stride
          (init :: .assign .int [B] "offset" (.alg (.add .int .scalarL
            (.mul .int .nil a b) (.ref .int [B] "n_offsets"))) :: rest)) t =
      FP.Structural.step (engine algebra)
        (.forRangeInt index start stop stride
          (init :: .assign .int [B] "offset" (.alg (.add .int .scalarL
            (.mul .int .nil b a) (.ref .int [B] "n_offsets"))) :: rest)) t := by
    simpa [Contextual, FP.Structural.run] using
      (VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FPProof.contextual_index_loop
        algebra t [] [] index start stop stride init rest B a b)
  have run_map (body : List ComputeStmt) : ∀ t,
      (∀ statement ∈ body, ∀ u,
        FP.Structural.step (engine algebra) statement u =
          FP.Structural.step (engine algebra) (commuteLoop B statement) u) →
      FP.Structural.run (engine algebra) body t =
        FP.Structural.run (engine algebra) (body.map (commuteLoop B)) t := by
    induction body with
    | nil => intro t _; rfl
    | cons statement rest ih =>
      intro t same
      simp only [List.map_cons, FP.Structural.run]
      rw [same statement (by simp)]
      congr 1
      funext u
      exact ih u (fun st member => same st (by simp [member]))
  apply run_map
  intro statement member t
  simp only [ComputeKernel.surfaceBody, List.mem_cons,
    List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first | rfl | exact loop_step t _ _ _ _ _ _ _ _

#axiomsClean equivalent
#axiomsClean execution
end VeriTile.Bench.Repairs.SoftmaxOptimize.KernelRewrite
