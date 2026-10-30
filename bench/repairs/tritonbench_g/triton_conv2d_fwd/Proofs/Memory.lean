import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Grid
namespace VeriTile.Bench.Repairs.Conv2d
variable {α β : Type}

theorem sourceValue_inputs (s : Shape) (p : ISA α β) (flags : Flags) (a b : Memory α)
    (ha : a .input=b .input) (hb : a .weight=b .weight) (x : Cell) :
    sourceValue s p flags a x=sourceValue s p flags b x := by
  have step : ∀ g r f h w c z,sourceStep s p flags a g r f h w c z=
      sourceStep s p flags b g r f h w c z := by
    intros;simp only [sourceStep,aLoad,bLoad,ha,hb]
  have channel : ∀ g r f h w c z,channelLoop s p flags a g r f h w c z=
      channelLoop s p flags b g r f h w c z := by
    intro g r f h w c
    induction c with
    | zero => intro z;rfl
    | succ c ih => intro z;rw [channelLoop,channelLoop,ih,step]
  have width : ∀ g r f h w z,widthLoop s p flags a g r f h w z=
      widthLoop s p flags b g r f h w z := by
    intro g r f h w
    induction w with
    | zero => intro z;rfl
    | succ w ih => intro z;rw [widthLoop,widthLoop,ih,channel]
  have height : ∀ g r f h,heightLoop s p flags a g r f h=
      heightLoop s p flags b g r f h := by
    intro g r f h
    induction h with
    | zero => rfl
    | succ h ih => rw [heightLoop,heightLoop,ih,width]
  unfold sourceValue
  rw [height]

theorem store_frame (m : Memory α) (address : Nat) (value : α) (r : Region) (a : Nat)
    (outside : r≠.output ∨ a≠address) : store m address value r a=m r a := by
  rcases outside with h|h <;> simp [store,h]
theorem store_inputs (m : Memory α) (address : Nat) (value : α) :
    (store m address value .input=m .input) ∧ (store m address value .weight=m .weight) := by
  constructor <;> funext n <;> simp [store]

theorem commit_frame (writes : List (Nat×α)) (m : Memory α) (r : Region) (a : Nat)
    (outside : ∀ w∈writes,r≠.output ∨ a≠w.1) : commit writes m r a=m r a := by
  induction writes generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change commit ws (store m w.1 w.2) r a=m r a
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    exact store_frame m w.1 w.2 r a (outside w (by simp))
theorem commit_inputs (writes : List (Nat×α)) (m : Memory α) :
    (commit writes m .input=m .input) ∧ (commit writes m .weight=m .weight) := by
  constructor <;> funext a <;> apply commit_frame <;> intros <;> exact Or.inl (by decide)
theorem commit_append (a b : List (Nat×α)) (m : Memory α) :
    commit (a++b) m=commit b (commit a m) := by
  exact List.foldl_append

theorem programWrites_inputs (s : Shape) (p : ISA α β) (flags : Flags) (pid : Program)
    (a b : Memory α) (ha : a .input=b .input) (hb : a .weight=b .weight) :
    programWrites s p flags pid a=programWrites s p flags pid b := by
  unfold programWrites
  congr 1
  funext x
  split_ifs
  · rw [sourceValue_inputs s p flags a b ha hb]
  · rfl
theorem sourceProgram_inputs (s : Shape) (p : ISA α β) (flags : Flags) (pid : Program)
    (m : Memory α) : (sourceProgram s p flags pid m .input=m .input) ∧
      (sourceProgram s p flags pid m .weight=m .weight) :=
  commit_inputs _ m

theorem sourcePrograms_inputs (s : Shape) (p : ISA α β) (flags : Flags)
    (ps : List Program) (m : Memory α) :
    (sourcePrograms s p flags ps m .input=m .input) ∧ (sourcePrograms s p flags ps m .weight=m .weight) := by
  induction ps generalizing m with
  | nil => exact ⟨rfl,rfl⟩
  | cons pid ps ih =>
    have step := sourceProgram_inputs s p flags pid m
    have tail := ih (sourceProgram s p flags pid m)
    exact ⟨tail.1.trans step.1,tail.2.trans step.2⟩

/-- Flattening snapshots is a theorem of current-memory input preservation;
the executor itself computes each CTA from its current state. -/
theorem sourcePrograms_commit (s : Shape) (p : ISA α β) (flags : Flags)
    (ps : List Program) (m : Memory α) :
    sourcePrograms s p flags ps m =
      commit (ps.flatMap (fun pid => programWrites s p flags pid m)) m := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    rw [sourcePrograms,ih,List.flatMap_cons,commit_append]
    have inputs := sourceProgram_inputs s p flags pid m
    have same : ps.flatMap (fun q => programWrites s p flags q (sourceProgram s p flags pid m))=
        ps.flatMap (fun q => programWrites s p flags q m) := by
      congr 1
      funext q
      exact programWrites_inputs s p flags q _ m inputs.1 inputs.2
    rw [same]
    rfl


#axiomsClean sourceValue_inputs
#axiomsClean sourcePrograms_commit
end VeriTile.Bench.Repairs.Conv2d
