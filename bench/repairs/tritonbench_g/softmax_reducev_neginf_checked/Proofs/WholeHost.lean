import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Host
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Host
open VeriTile.Triton FP.Structural
open SoftmaxTriton1 (Result)
noncomputable section
set_option maxHeartbeats 2000000

abbrev Memory (α : Type) := RegionName → Nat → α

def readOutput {α} (base : Nat) (m : Memory α) : Nat → α := fun a => m "Out" (base+a)

/-- Memory interpretation for the supplied contiguous output pointer. Prefix
words before that pointer remain in their original storage, with no copyback. -/
def putOutput {α} (base : Nat) (m : Memory α) (out : Nat → α) : Memory α :=
  fun r a => if r="Out" ∧ base ≤ a then out (a-base) else m r a

def liftResult {α} (base : Nat) (m : Memory α) : Result (Nat → α) Unit → Result (Memory α) Unit
  | .ok () out => .ok () (putOutput base m out)
  | .error e out => .error e (putOutput base m out)

def invokeFull {α} (base : Nat) (invoke : Job → (Nat → α) → Result (Nat → α) Unit)
    (j : Job) (m : Memory α) : Result (Memory α) Unit :=
  liftResult base m (invoke j (readOutput base m))

def runFull {α} (base : Nat) (invoke : Job → (Nat → α) → Result (Nat → α) Unit) :
    List Job → Memory α → Result (Memory α) Unit
  | [],m => .ok () m
  | j::js,m => (invokeFull base invoke j m).bind (fun _ next => runFull base invoke js next)

@[simp] theorem put_same {α} (base : Nat) (m : Memory α) : putOutput base m (readOutput base m)=m := by
  funext r a
  by_cases h : r="Out" ∧ base ≤ a
  · simp [putOutput,readOutput,h,h.1,Nat.add_sub_of_le h.2]
  · simp [putOutput,h]
@[simp] theorem put_read {α} (base : Nat) (m : Memory α) (out : Nat → α) :
    readOutput base (putOutput base m out)=out := by
  funext a; simp [readOutput,putOutput]
@[simp] theorem put_twice {α} (base : Nat) (m : Memory α) (out next : Nat → α) :
    putOutput base (putOutput base m out) next=putOutput base m next := by
  funext r a
  by_cases h : r="Out" ∧ base ≤ a <;> simp [putOutput,h]

theorem runFull_projection {α} (base : Nat) (invoke : Job → (Nat → α) → Result (Nat → α) Unit)
    (jobs : List Job) (m : Memory α) :
    runFull base invoke jobs m=liftResult base m (runJobs invoke jobs (readOutput base m)) := by
  induction jobs generalizing m with
  | nil => simp [runFull,runJobs,liftResult]
  | cons j js ih =>
    cases call : invoke j (readOutput base m) with
    | error e out => simp [runFull,invokeFull,call,liftResult,runJobs,Result.bind]
    | ok v out =>
      cases v
      simp only [runFull,invokeFull,call,liftResult,runJobs,Result.bind]
      rw [ih]
      simp only [put_read]
      cases runJobs invoke js out <;> simp [liftResult]

def noWork (c : Call) : Prop := c.batch=0 ∨ c.heads=0
instance (c : Call) : Decidable (noWork c) := inferInstanceAs (Decidable (_∨_))

/-- The wrapper launches exactly (batch,head), returns Python None, and writes
through the supplied output pointer. The schedule is a complete grid order. -/
def wrapper {α} (c : Call) (base : Nat) (schedule : List Job)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (m : Memory α) : Result (Memory α) Unit :=
  if noWork c then .ok () m else runFull base invoke schedule m

def Outside (c : Call) (base : Nat) (r : RegionName) (a : Nat) : Prop :=
  r≠"Out" ∨ a < base ∨ Off c (workset c) (a-base)

/-- Successful complete host return: every batch/head/channel output and every
caller storage cell outside the actual grid's output footprint. -/
theorem whole {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α) (base : Nat)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : LocalRuntime M c x invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial final : Memory α)
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions → final "Out" (base+address c j d)=row M c x j d) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) := by
  by_cases empty : noWork c
  · have eq : initial=final := by simpa [wrapper,empty,Result.ok.injEq] using run
    subst final
    refine ⟨?_,fun _ _ _ => rfl⟩
    intro j hj d hd
    rcases empty with e | e <;> dsimp [Legal] at hj <;> omega
  · have run' : runFull base invoke schedule initial=.ok () final := by simpa [wrapper,empty] using run
    rw [runFull_projection] at run'
    cases call : runJobs invoke schedule (readOutput base initial) with
    | error e out => simp [call,liftResult] at run'
    | ok v out =>
      cases v
      have eq : putOutput base initial out=final := by simpa [call,liftResult] using run'
      subst final
      have result := jobs_return M c x invoke rt schedule
        (fun j hj => (mem_workset c j).mp (complete.mem_iff.mp hj)) (readOutput base initial) out call
      refine ⟨?_,?_⟩
      · intro j hj d hd
        simpa [putOutput] using result.1 j (complete.mem_iff.mpr ((mem_workset c j).mpr hj)) d hd
      · intro r a outside
        by_cases h : r="Out" ∧ base ≤ a
        · have off : Off c (workset c) (a-base) := by
            rcases outside with out | before | off
            · exact (out h.1).elim
            · omega
            · exact off
          have same := result.2 (a-base) (fun j hj => off j (complete.mem_iff.mp hj))
          simpa [putOutput,h,readOutput,Nat.add_sub_of_le h.2,h.1] using same
        · simp [putOutput,h]

theorem all_outcomes {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α) (base : Nat)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : LocalRuntime M c x invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial : Memory α)
    (r : RegionName) (a : Nat) (outside : Outside c base r a) :
    (wrapper c base schedule invoke initial).state r a=initial r a := by
  by_cases empty : noWork c
  · simp [wrapper,empty,Result.state]
  · have frame := jobs_all_outcomes M c x invoke rt schedule
      (fun j hj => (mem_workset c j).mp (complete.mem_iff.mp hj)) (readOutput base initial)
    have framed (out : Nat → α) (hframe : Frame c schedule (readOutput base initial) out) :
        putOutput base initial out r a=initial r a := by
      by_cases h : r="Out" ∧ base ≤ a
      · have off : Off c (workset c) (a-base) := by
          rcases outside with out | before | off
          · exact (out h.1).elim
          · omega
          · exact off
        have same := hframe (a-base) (fun j hj => off j (complete.mem_iff.mp hj))
        simpa [putOutput,h,readOutput,Nat.add_sub_of_le h.2,h.1] using same
      · simp [putOutput,h]
    simp only [wrapper,if_neg empty,runFull_projection]
    cases call : runJobs invoke schedule (readOutput base initial) with
    | error e out =>
      simp only [call,Result.state] at frame
      exact framed out frame
    | ok v out =>
      cases v
      simp only [call,Result.state] at frame
      exact framed out frame

/-- Exact byte frame for every success or failure, including output-prefix
and output-suffix canaries and every read-only input/metadata allocation. -/
theorem all_outcomes_bytes {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α) (base : Nat)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : LocalRuntime M c x invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial : Memory α)
    (encode : α → Fin 4 → BitVec 8) (r : RegionName) (word : Nat) (byte : Fin 4)
    (outside : Outside c base r word) :
    encode ((wrapper c base schedule invoke initial).state r word) byte=encode (initial r word) byte := by
  rw [all_outcomes M c x base invoke rt schedule complete initial r word outside]

#axiomsClean runFull_projection
#axiomsClean whole
#axiomsClean all_outcomes
#axiomsClean all_outcomes_bytes
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Host
