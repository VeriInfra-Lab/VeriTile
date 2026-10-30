import bench.tritonbench_g.lightning_attention.Proofs.NativeHeadLocality

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeLocality
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

/-- Exact floating cell tags, not merely equality after read-defaulting. -/
def AllReal (memory : Memory α) : Prop := ∀ r o, ∃ value, memory r o = .mk .real value

theorem scatter_real {ι : Type} (xs : List ι) (dest : RegionName)
    (active : ι → Prop) [DecidablePred active] (offset : ι → Nat)
    (value : ι → α) (memory : Memory α) (h : AllReal memory) :
    AllReal (scatter xs dest active offset (fun i => .mk .real (value i)) memory) := by
  induction xs generalizing memory with
  | nil => exact h
  | cons i rest ih =>
    apply ih
    intro r o
    by_cases ha : active i
    · simp only [if_pos ha]
      by_cases he : r=dest ∧ o=offset i
      · exact ⟨value i,by simp only [if_pos he]⟩
      · simpa only [if_neg he] using h r o
    · simpa only [if_neg ha] using h r o

theorem fold_real (f : Nat → Memory α → Memory α) (n : Nat) (memory : Memory α)
    (h : AllReal memory) (step : ∀ j m, AllReal m → AllReal (f j m)) :
    AllReal (foldRange f 0 n 1 memory) := by
  induction n with
  | zero => simpa only [foldRange_zero] using h
  | succ n ih => simpa only [foldRange_succ] using step n _ ih

theorem writeOutput_real (M : Algebra α) (p : Parameters) (pid : PID) (block : Nat)
    (values : Matrix α p.BT p.BM) (memory : Memory α) (h : AllReal memory) :
    AllReal (writeOutput M p pid block values memory) :=
  scatter_real _ _ _ _ _ _ h

theorem writeRows_real (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α rows width) (memory : Memory α) (h : AllReal memory) :
    AllReal (writeRows M r N width head start rows values memory) :=
  scatter_real _ _ _ _ _ _ h

theorem writeTrans_real (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α width rows) (memory : Memory α) (h : AllReal memory) :
    AllReal (writeTrans M r N width head start rows values memory) :=
  scatter_real _ _ _ _ _ _ h

theorem fwdMemory_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input memory : Memory α) (h : AllReal memory) :
    AllReal (fwdMemory M D p pid input memory) :=
  fold_real _ _ _ h (fun _ _ hm => writeOutput_real _ _ _ _ _ _ hm)

theorem intraMemory_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (memory : Memory α) (h : AllReal memory) :
    AllReal (intraMemory M D p head block memory) :=
  writeRows_real _ _ _ _ _ _ _ _ _
    (writeTrans_real _ _ _ _ _ _ _ _ _
      (writeRows_real _ _ _ _ _ _ _ _ _ h))

theorem queryUpdate_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (carry : Matrix α p.E p.D) (memory : Memory α) (h : AllReal memory) :
    AllReal (queryUpdate M D p head row carry memory) :=
  writeRows_real _ _ _ _ _ _ _ _ _ h

theorem keyValueUpdate_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (carry : Matrix α p.D p.E) (memory : Memory α) (h : AllReal memory) :
    AllReal (keyValueUpdate M D p head row carry memory) :=
  writeRows_real _ _ _ _ _ _ _ _ _
    (writeTrans_real _ _ _ _ _ _ _ _ _ h)

theorem queryBlockMemory_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (carry : Matrix α p.E p.D) (memory : Memory α) (h : AllReal memory) :
    AllReal (queryBlockMemory M D p head block carry memory) := by
  unfold queryBlockMemory
  split
  · exact h
  · exact fold_real _ _ _ h (fun _ _ hm => queryUpdate_real _ _ _ _ _ _ _ hm)

theorem keyValueBlockMemory_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (carry : Matrix α p.D p.E) (memory : Memory α) (h : AllReal memory) :
    AllReal (keyValueBlockMemory M D p head block carry memory) := by
  unfold keyValueBlockMemory
  split
  · exact h
  · exact fold_real _ _ _ h (fun _ _ hm => keyValueUpdate_real _ _ _ _ _ _ _ hm)

theorem queryScanMemory_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input memory : Memory α) (h : AllReal memory) :
    AllReal (queryScanMemory M D p head input memory) :=
  fold_real _ _ _ h (fun _ _ hm => queryBlockMemory_real _ _ _ _ _ _ _ hm)

theorem keyValueScanMemory_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input memory : Memory α) (h : AllReal memory) :
    AllReal (keyValueScanMemory M D p head input memory) :=
  fold_real _ _ _ h (fun _ _ hm => keyValueBlockMemory_real _ _ _ _ _ _ _ hm)

theorem interMemory_real (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input memory : Memory α) (h : AllReal memory) :
    AllReal (interMemory M D p head input memory) :=
  keyValueScanMemory_real _ _ _ _ _ _ (queryScanMemory_real _ _ _ _ _ _ h)

theorem stageMemory_real (M : Algebra α) (D : DotPrimitive α) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (memory : Memory α) (h : AllReal memory) :
    AllReal (stageMemory M D stage d pid memory) := by
  cases stage with
  | forward => exact fwdMemory_real _ _ _ _ _ _ h
  | intra => exact intraMemory_real _ _ _ _ _ _ h
  | inter => exact interMemory_real _ _ _ _ _ _ h

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeLocality
