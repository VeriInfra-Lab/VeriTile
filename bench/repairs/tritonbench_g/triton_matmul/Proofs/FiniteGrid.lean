import bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteExecution
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FiniteGrid
open VeriTile.Triton FP.Structural FPValue
open FiniteExecution
noncomputable section
set_option maxHeartbeats 600000

theorem loop_mem {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (a b : State α) (same : a.mem=b.mem) (pid fuel : Nat) (k : BitVec 64) (carry : CIndex r.parameters → α) :
    loop M D r a pid fuel k carry=loop M D r b pid fuel k carry := by
  have ua : ∀ block carry,FiniteExecution.update M D r a pid block carry=FiniteExecution.update M D r b pid block carry := by
    intro block carry
    have ha : FiniteExecution.loadA M r a pid block=FiniteExecution.loadA M r b pid block := by
      funext i; simp only [FiniteExecution.loadA,same]
    have hb : FiniteExecution.loadB M r a pid block=FiniteExecution.loadB M r b pid block := by
      funext i; simp only [FiniteExecution.loadB,same]
    simp only [FiniteExecution.update,ha,hb]
  induction fuel generalizing k carry with
  | zero => rfl
  | succ fuel ih => simp only [FiniteExecution.loop,ua,ih]

#axiomsClean loop_mem

theorem store_fold_mem {α : Type} {I : Type} (region : RegionName) (mask : I → Bool)
    (offset : I → Nat) (value : I → Cell α) (items : List I) (a b : State α) (same : a.mem=b.mem) :
    (items.foldl (fun s i => if mask i then s.write region (offset i) (value i) else s) a).mem=
      (items.foldl (fun s i => if mask i then s.write region (offset i) (value i) else s) b).mem := by
  induction items generalizing a b with
  | nil => exact same
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases h : mask i=Bool.true
    · simp only [if_pos h]
      apply ih
      simp only [State.write,same]
    · simp only [if_neg h]
      exact ih a b same

#axiomsClean store_fold_mem

private theorem writeItems_mem {α : Type} (M : Algebra α) (r : Request) (pid : Nat)
    (carry : CIndex r.parameters → α) (items : List (CIndex r.parameters))
    (a b : State α) (same : a.mem = b.mem) :
    (writeItems M r pid carry items a).mem = (writeItems M r pid carry items b).mem :=
  store_fold_mem r.C _ _ _ items (instanceState r pid a) (instanceState r pid b) same

theorem writeResult_mem {α : Type} (M : Algebra α) (r : Request) (pid : Nat)
    (carry : CIndex r.parameters → α) (a b : State α) (same : a.mem = b.mem) :
    (writeResult M r pid carry a).mem = (writeResult M r pid carry b).mem :=
  writeItems_mem M r pid carry _ a b same

private theorem mapped_memory {α β : Type} (x y : Option β)
    (f g : β → State α) (same : x = y)
    (outputs : ∀ v, (f v).mem = (g v).mem) :
    (x.map f).map State.mem = (y.map g).map State.mem := by
  subst y
  cases x with
  | none => rfl
  | some v => exact congrArg some (outputs v)

private theorem mapped_loop_memory {α : Type} [Inhabited α] (M : Algebra α)
    (D : DotAcc α) (r : Request) (pid fuel : Nat) (k : BitVec 64)
    (carry : CIndex r.parameters → α) (a b : State α) (same : a.mem = b.mem) :
    ((loop M D r a pid fuel k carry).map (fun values => writeResult M r pid values a)).map State.mem =
    ((loop M D r b pid fuel k carry).map (fun values => writeResult M r pid values b)).map State.mem :=
  mapped_memory (loop M D r a pid fuel k carry) (loop M D r b pid fuel k carry)
    (fun values => writeResult M r pid values a) (fun values => writeResult M r pid values b)
    (loop_mem M D r a b same pid fuel k carry)
    (fun values => writeResult_mem M r pid values a b same)

theorem program_mem {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (a b : State α) (same : a.mem=b.mem) (pid : Nat) :
    (program M D r pid a).map State.mem=(program M D r pid b).map State.mem :=
  mapped_loop_memory M D r pid
    (Index.ceiling (Index.scalarABI r.inner) (SourceScope.blockK r.storage)).toNat
    (Index.word 0) (fun _ => M.literal (some .fp32) .real 0) a b same

#axiomsClean program_mem

def grid {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::tail,s => (program M D r pid s).bind (grid M D r tail)

/-- Each program executes finite signed controls/loads/stores. This composes
its full-memory refinement to the complete typed source, including any source
register state left by earlier programs. -/
theorem source {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (basis : State α) (t : TensorAPI r basis) (hl : programs r<2^31)
    (ids : List Nat) (valid : ∀ pid∈ids,pid<programs r)
    (a b : State α) (same : a.mem=b.mem) :
    ∃ actual typed,grid M D r ids a=some actual ∧ runPrograms M D r ids b=some typed ∧ actual.mem=typed.mem := by
  induction ids generalizing a b with
  | nil => exact ⟨a,b,rfl,rfl,same⟩
  | cons pid tail ih =>
    have hp := valid pid (by simp)
    obtain ⟨actual,typed,ha,ht,hm⟩ := program_source M D r basis t b pid hp hl
    have mapped := program_mem M D r a b same pid
    rw [ha,Option.map_some] at mapped
    obtain ⟨next,hn,hmem⟩ := Option.map_eq_some_iff.mp mapped
    have heq : next.mem=typed.mem := hmem.trans hm
    obtain ⟨final,typedFinal,hf,tf,mem⟩ := ih (fun q hq => valid q (by simp [hq])) next typed heq
    exact ⟨final,typedFinal,by simp only [grid,hn,Option.bind_some,hf],
      by simp only [runPrograms,ht,Option.bind_some,tf],mem⟩

#axiomsClean program_mem
#axiomsClean source
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FiniteGrid
