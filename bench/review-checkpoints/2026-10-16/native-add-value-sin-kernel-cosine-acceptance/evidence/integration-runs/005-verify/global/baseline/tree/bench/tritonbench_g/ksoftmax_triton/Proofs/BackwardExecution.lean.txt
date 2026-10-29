import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardReal

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.RealExecution
open VeriTile.Triton
set_option maxHeartbeats 1600000
set_option maxRecDepth 16000
noncomputable section

/-- A genuine padded four-lane execution from empty registers, with two
nonzero program IDs and three distinct stride pairs. -/
def initial (log : Bool) : BlockState where
  mem := fun r _ => MemCell.real
    (if r = "G" then 2 else if r = "O" then (if log then 0 else 1 / 4) else 7)
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 2 else if ax = 1 then 1 else 0
  numPids := fun _ => 3

private theorem initial_gradient (log : Bool) (offset : Nat) :
    (initial log).readMem "G" offset = 2 := by
  simp [initial, BlockState.readMem, MemCell.readAs, MemCell.real, MemCell.of]

private theorem initial_saved (log : Bool) (offset : Nat) :
    (initial log).readMem "O" offset = (if log then 0 else 1 / 4) := by
  simp [initial, BlockState.readMem, MemCell.readAs, MemCell.real, MemCell.of]

/-- Both reduction branches run on nonempty, partially masked tiles. The
LOG+CAUSAL case actually writes -4 to the future lane, whereas the ordinary
causal branch writes zero there. The padded cell and another region retain
their original values. This is real-model evidence, not a native FP receipt. -/
theorem all_flags_successful (log causal fp16 : Bool) :
    ∃ t, exec (source "D" "G" "O" 11 7 13 5 17 3 3 4 log causal fp16).toAlgKernel
        (initial log) = some t ∧
      t.readMem "D" 29 = (if log then (if causal then -2 else -4)
        else (if causal then 1 / 4 else 1 / 8)) ∧
      t.readMem "D" 31 = (if log then -4 else (if causal then 0 else 1 / 8)) ∧
      t.mem "D" 32 = MemCell.real 7 ∧
      t.mem "untouched" 41 = MemCell.real 7 := by
  obtain ⟨t, hrun, hvalue, hframe⟩ :=
    source_region_run "D" "G" "O" 11 7 13 5 17 3 3 4 log causal fp16
      (initial log) (fun _ => 2) (fun _ => if log then 0 else 1 / 4)
      (fun j _ => initial_gradient log _) (fun j _ => initial_saved log _)
  refine ⟨t, hrun, ?_, ?_, ?_, ?_⟩
  · have h := hvalue (⟨0, by decide⟩ : Fin 4) (by decide)
    cases log <;> cases causal <;>
      norm_num [initial, formula, masked, readActive, Fin.sum_univ_succ] at h ⊢ <;>
      exact h
  · have h := hvalue (⟨2, by decide⟩ : Fin 4) (by decide)
    cases log <;> cases causal <;>
      norm_num [initial, formula, masked, readActive, Fin.sum_univ_succ] at h ⊢ <;>
      exact h
  · have h := hframe "D" 32 (Or.inr (by
      intro i hi
      simp only [initial, ↓reduceIte, Nat.reduceMul, Nat.reduceAdd]
      omega))
    simpa [initial] using h
  · have h := hframe "untouched" 41 (Or.inl (by decide))
    simpa [initial] using h

end
#axiomsClean all_flags_successful
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.RealExecution
