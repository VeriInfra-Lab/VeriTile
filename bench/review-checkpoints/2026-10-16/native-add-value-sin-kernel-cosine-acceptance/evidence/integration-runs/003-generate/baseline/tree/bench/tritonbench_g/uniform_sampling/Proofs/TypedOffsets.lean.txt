import bench.tritonbench_g.uniform_sampling.Proofs.TypedBindings
import bench.tritonbench_g.uniform_sampling.Proofs.OffsetExecution

/-! All four actual output-offset assignments in the typed evaluator. -/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedOffsets
open VeriTile.Triton TypedSource FP.Structural TypedBindings
open OffsetExecution (outputFn firstOp bumpOp)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

def body (B : Nat) : List ComputeStmt := [
  .assign .nat [B] "off_0" (.alg (firstOp B)),
  .assign .nat [B] "off_1" (.alg (bumpOp B "off_0")),
  .assign .nat [B] "off_2" (.alg (bumpOp B "off_1")),
  .assign .nat [B] "off_3" (.alg (bumpOp B "off_2"))]

def result {α : Type} (pid B : Nat) (s : State α) : State α :=
  let s := s.setReg "off_0" .nat [B] (outputFn pid B 0)
  let s := s.setReg "off_1" .nat [B] (outputFn pid B 1)
  let s := s.setReg "off_2" .nat [B] (outputFn pid B 2)
  s.setReg "off_3" .nat [B] (outputFn pid B 3)

theorem first_eval {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (s : State α)
    (hb : s.regs .nat [] "BLOCK_arg" = some (fun _ => B)) :
    FP.Structural.evalOp M none (firstOp B) s = some (outputFn (s.pids 0) B 0) := by
  simp [firstOp, evalOp_unfold, hb, numeric]
  funext i
  simp only [outputFn, bop, Broadcast.rightIndex_scalarL]
  ring

theorem bump_eval {α : Type} [Inhabited α] (M : Algebra α)
    (pid B k : Nat) (name : RegName) (s : State α)
    (hb : s.regs .nat [] "BLOCK_arg" = some (fun _ => B))
    (ho : s.regs .nat [B] name = some (outputFn pid B k)) :
    FP.Structural.evalOp M none (bumpOp B name) s = some (outputFn pid B (k+1)) := by
  simp [bumpOp, evalOp_unfold, hb, ho, numeric]
  funext i
  simp only [outputFn, bop, Broadcast.leftIndex_scalarR]
  ring

private theorem four_steps {α : Type} [Inhabited α] (M : Algebra α)
    {a b c d : ComputeStmt} {s0 s1 s2 s3 s4 : State α}
    (ha : step M a s0 = some s1) (hb : step M b s1 = some s2)
    (hc : step M c s2 = some s3) (hd : step M d s3 = some s4) :
    run M [a, b, c, d] s0 = some s4 := by
  simp only [run, ha, hb, hc, hd, bind, Option.bind, pure]

theorem offsets_exec {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (s : State α)
    (hb : s.regs .nat [] "BLOCK_arg" = some (fun _ => B)) :
    run M (body B) s = some (result (s.pids 0) B s) := by
  let t0 := s.setReg "off_0" .nat [B] (outputFn (s.pids 0) B 0)
  let t1 := t0.setReg "off_1" .nat [B] (outputFn (s.pids 0) B 1)
  let t2 := t1.setReg "off_2" .nat [B] (outputFn (s.pids 0) B 2)
  have h0 := assign_step_of_eval M "off_0" _ _ _ (first_eval M B s hb)
  have h1 := assign_step_of_eval M "off_1" _ _ _
    (bump_eval M (s.pids 0) B 0 "off_0" t0
      (by simpa [t0] using hb) (by simp [t0]))
  have h2 := assign_step_of_eval M "off_2" _ _ _
    (bump_eval M (s.pids 0) B 1 "off_1" t1
      (by simpa [t1, t0] using hb) (by simp [t1]))
  have h3 := assign_step_of_eval M "off_3" _ _ _
    (bump_eval M (s.pids 0) B 2 "off_2" t2
      (by simpa [t2, t1, t0] using hb) (by simp [t2]))
  exact four_steps M h0 h1 h2 h3

theorem source_slice (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    ((source storage swapped y N seed offset lower upper B hB).surfaceBody.drop 15).take 4 =
      body B := by
  cases storage <;> cases swapped <;> rfl

#axiomsClean first_eval
#axiomsClean bump_eval
#axiomsClean offsets_exec
#axiomsClean source_slice

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedOffsets
