import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked_Source
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Host

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Actual PyTorch zero_ is represented by scalar half stores over the fresh
output, with no kernel launch and no input read. -/
def zeroCode (output : Tensor) : Code α Unit :=
  sequenceCode (fun i => do
    let zero ← emit .zero
    emit (.storeHalf (output.storage,output.offset+i) zero)) (List.range (output.rows*output.columns))

def zeroMemory (ops : Operations α) (output : Tensor) (memory : Memory α) : Memory α :=
  fun point => if point.1 = output.storage ∧ output.offset ≤ point.2 ∧ point.2-output.offset < output.rows*output.columns
    then ops.zero else memory point

theorem eval_zero (ops : Operations α) (output : Tensor) (memory : Memory α) :
    (zeroCode output).eval ops memory = ((),zeroMemory ops output memory) := by
  let writes (i : Nat) (point : Address) : Prop := point = (output.storage,output.offset+i)
  have atomic (i : Nat) (m : Memory α) :
      (do let zero ← emit (α:=α) .zero; emit (.storeHalf (output.storage,output.offset+i) zero)).eval ops m =
        ((),CrossEntropy1.Schedule.put writes (fun _ _ => ops.zero) i m) := by
    simp only [bind,eval_bind,eval_emit,Event.eval,Prod.fst,Prod.snd]
    rfl
  rw [zeroCode,eval_sequence ops _ _ atomic]
  congr 1
  funext point
  by_cases active : point.1 = output.storage ∧ output.offset ≤ point.2 ∧ point.2-output.offset < output.rows*output.columns
  · have same : point = (output.storage,output.offset+(point.2-output.offset)) := Prod.ext active.1 (by omega)
    have result := CrossEntropy1.Schedule.fixed_output writes (fun _ _ => ops.zero)
      (List.range (output.rows*output.columns)) memory (point.2-output.offset) point same
      (by intro i hi hit; have index := congrArg Prod.snd hit; dsimp only at index; omega)
    simpa [List.mem_range,active,active.1,active.2.1,active.2.2,zeroMemory,Schedule] using result
  · have result := CrossEntropy1.Schedule.fixed_frame writes (fun _ _ => ops.zero)
      (List.range (output.rows*output.columns)) memory point (by
        intro i hi hit
        have bound := List.mem_range.mp hi
        apply active
        change point = (output.storage,output.offset+i) at hit
        rw [hit]
        exact ⟨rfl,by omega,by omega⟩)
    simpa only [zeroMemory,if_neg active,Schedule] using result

def replayCode (p : Problem) (b : Buffers) : Nat → Code α Unit
  | 0 => pure ()
  | n+1 => do let _ ← gridCode p b; replayCode p b n

theorem eval_replay (ops : Operations α) (p : Problem) (b : Buffers) (layout : Layout p b)
    (count : Nat) (memory : Memory α) :
    (replayCode p b count).eval ops memory = ((),Source.replay (sourceArithmetic ops) p b count memory) := by
  induction count generalizing memory with
  | zero => rfl
  | succ count ih => simp only [replayCode,bind,eval_bind,eval_grid ops p b layout,Prod.fst,Prod.snd,ih,Source.replay]

def computationCode (args : Host.Arguments) (output : Tensor) (trials : Nat) : Code α Unit :=
  if args.A.rows = 0 ∨ args.B.columns = 0 then pure ()
  else if args.A.columns = 0 then zeroCode output
  else replayCode args.problem ⟨args.A,args.B,output⟩ (trials+1)

def resultMemory (ops : Operations α) (args : Host.Arguments) (output : Tensor) (memory : Memory α) : Memory α :=
  if args.A.rows = 0 ∨ args.B.columns = 0 then memory
  else if args.A.columns = 0 then zeroMemory ops output memory
  else Source.run (sourceArithmetic ops) args.problem ⟨args.A,args.B,output⟩ memory

theorem eval_computation (ops : Operations α) (args : Host.Arguments) (output : Tensor)
    (layout : Layout args.problem ⟨args.A,args.B,output⟩) (trials : Nat) (memory : Memory α) :
    (computationCode args output trials).eval ops memory = ((),resultMemory ops args output memory) := by
  unfold computationCode resultMemory
  split_ifs
  · rfl
  · exact eval_zero ops output memory
  · rw [eval_replay ops _ _ layout,Source.replay_positive _ _ _ layout]

#axiomsClean eval_zero
#axiomsClean eval_computation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
