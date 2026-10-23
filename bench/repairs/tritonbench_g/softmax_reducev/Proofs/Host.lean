import bench.repairs.tritonbench_g.softmax_reducev.Proofs.FPContract
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Storage

namespace VeriTile.Bench.Repairs.SoftmaxReduceV.Host
open VeriTile.Triton FP.Structural FPValue
open SoftmaxTriton1 (Result)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Call where
  (batch heads dimensions pages tokens maximum : Nat)

structure Job where
  (batch head : Nat)
  deriving DecidableEq

def Legal (c : Call) (j : Job) : Prop := j.batch < c.batch ∧ j.head < c.heads

def workset (c : Call) : List Job :=
  (List.range c.batch).flatMap (fun b => (List.range c.heads).map (fun h => ⟨b,h⟩))

theorem mem_workset (c : Call) (j : Job) : j∈workset c ↔ Legal c j := by
  cases j; simp [workset,Legal]

structure Input (α : Type) where
  (logics values : Nat → α)
  (starts counts pages : Nat → Nat)

def parameters (c : Call) : Parameters :=
  ⟨"Logics","V","Out","Locations","Starts","Lengths",c.maximum,
    c.tokens,1,c.heads*c.dimensions,c.dimensions,1,
    c.heads*c.dimensions,c.dimensions,1,c.maximum,1,c.dimensions,64,0⟩

def launch {α} [Inhabited α] (c : Call) (x : Input α) (out : Nat → α) (j : Job) : State α :=
  { mem := fun r a =>
      if r="Logics" then .mk .fp32 (x.logics a)
      else if r="V" then .mk .fp32 (x.values a)
      else if r="Out" then .mk .fp32 (out a)
      else if r="Locations" then .mk .int (Int.ofNat (x.pages a))
      else if r="Starts" then .mk .nat (x.starts a)
      else if r="Lengths" then .mk .nat (x.counts a)
      else .mk .fp32 default
    regs := fun _ _ _ => none
    pids := fun ax => if ax=0 then j.batch else if ax=1 then j.head else 0
    numPids := fun ax => if ax=0 then c.batch else if ax=1 then c.heads else 1
    undef := fun d _ _ => defaultValue d }

def address (c : Call) (j : Job) (d : Nat) := (j.batch*c.heads+j.head)*c.dimensions+d

theorem source_address {α} [Inhabited α] (c : Call) (x : Input α) (out : Nat → α) (j : Job)
    (d : Index (parameters c)) : outputOffset (parameters c) (launch c x out j) d=address c j d.1.val := by
  simp [outputOffset,parameters,launch,address,Nat.add_mul,Nat.mul_assoc]

theorem source_injective {α} [Inhabited α] (c : Call) (x : Input α) (out : Nat → α) (j : Job) :
    Function.Injective (outputOffset (parameters c) (launch c x out j)) := by
  intro a b eq
  rw [source_address,source_address] at eq
  have first : a.1=b.1 := Fin.ext (Nat.add_left_cancel eq)
  cases a; cases b; simp_all

theorem accumulator_independent {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α)
    (out other : Nat → α) (j : Job) :
    finalAccumulator M (parameters c) (launch c x out j)=
      finalAccumulator M (parameters c) (launch c x other j) := by
  rfl

def row {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α) (j : Job) (d : Nat) : α :=
  if h : d < c.dimensions then
    outputValues M (parameters c) (finalAccumulator M (parameters c) (launch c x (fun _ => default) j))
      (⟨d,h⟩,())
  else default

def evaluate {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α) (j : Job) (out : Nat → α) :
    Option (Nat → α) :=
  (FP.Structural.exec M (source (parameters c)) (launch c x out j)).map
    (fun t a => (t.mem "Out" a).read .fp32)

def Active (c : Call) (j : Job) (a : Nat) : Prop := ∃ d, d < c.dimensions ∧ a=address c j d

def Off (c : Call) (jobs : List Job) (a : Nat) : Prop := ∀ j∈jobs, ¬Active c j a

def Frame {α} (c : Call) (jobs : List Job) (s t : Nat → α) : Prop := ∀ a, Off c jobs a → t a=s a

/-- Proven complete source result; the target is the explicit source recurrence. -/
theorem source_law {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α)
    (j : Job) (s t : Nat → α) (run : evaluate M c x j s=some t) :
    (∀ d, d < c.dimensions → t (address c j d)=row M c x j d) ∧ Frame c [j] s t := by
  obtain ⟨u,exec,post⟩ := correctness M (parameters c) (launch c x s j) (by change 0 < 64; decide) (source_injective c x s j)
  have result : (fun a => (u.mem "Out" a).read .fp32)=t := by simpa [evaluate,exec] using run
  subst t
  constructor
  · intro d hd
    have value := congrArg (Cell.read .fp32) (post.1 ((⟨d,hd⟩ : Fin c.dimensions),()))
    rw [source_address] at value
    simpa [row,hd,accumulator_independent M c x s (fun _ => default) j] using value
  · intro a outside
    have eq := post.2 "Out" a (Or.inr (by
      intro d eq
      apply outside j (by simp)
      exact ⟨d.1.val,d.1.isLt,eq.trans (source_address c x s j d)⟩))
    change (u.mem "Out" a).read .fp32=s a
    rw [eq]
    simp [launch]

theorem source_total {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α)
    (j : Job) (s : Nat → α) : ∃ t, evaluate M c x j s=some t := by
  obtain ⟨u,run,_⟩ := execution_memory M (parameters c) (launch c x s j) (by change 0 < 64; decide)
  exact ⟨(fun a => (u.mem "Out" a).read .fp32),by simp [evaluate,run]⟩

theorem flatten_injective (K h d h' d' : Nat) (hd : d < K) (hd' : d' < K)
    (same : h*K+d=h'*K+d') : h=h' ∧ d=d' := by
  have pos : 0 < K := by omega
  have divEq (h d : Nat) (hd : d < K) : (h*K+d)/K=h := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ pos,Nat.div_eq_of_lt hd,Nat.zero_add]
  have hs := congrArg (fun a => a/K) same
  dsimp only at hs
  rw [divEq h d hd,divEq h' d' hd'] at hs
  exact ⟨hs,by rw [hs] at same; omega⟩

theorem ownership (c : Call) (j k : Job) (hj : Legal c j) (hk : Legal c k)
    (d e : Nat) (hd : d < c.dimensions) (he : e < c.dimensions) (same : address c j d=address c k e) : j=k ∧ d=e := by
  have a := flatten_injective c.dimensions _ d _ e hd he same
  have b := flatten_injective c.heads j.batch j.head k.batch k.head hj.2 hk.2 a.1
  exact ⟨by cases j; cases k; simp_all,a.2⟩

/-- Local launch/runtime correspondence with the concrete full-source evaluator.
Success supplies execution only; it supplies no recurrence or output values.
Error states may contain partial local output stores. -/
structure LocalRuntime {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) : Prop where
  success : ∀ j s t, Legal c j → invoke j s=.ok () t → evaluate M c x j s=some t
  error : ∀ j s e t, Legal c j → invoke j s=.error e t → Frame c [j] s t

def runJobs {α} (invoke : Job → (Nat → α) → Result (Nat → α) Unit) : List Job → (Nat → α) → Result (Nat → α) Unit
  | [],s => .ok () s
  | j::js,s => (invoke j s).bind (fun _ t => runJobs invoke js t)

theorem jobs_return {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : LocalRuntime M c x invoke)
    (jobs : List Job) (legal : ∀ j∈jobs,Legal c j) (s t : Nat → α)
    (run : runJobs invoke jobs s=.ok () t) :
    (∀ j∈jobs, ∀ d, d < c.dimensions → t (address c j d)=row M c x j d) ∧ Frame c jobs s t := by
  induction jobs generalizing s with
  | nil =>
    simp only [runJobs,Result.ok.injEq] at run
    cases run.2
    exact ⟨by simp,fun _ _ => rfl⟩
  | cons j js ih =>
    cases call : invoke j s with
    | error e u => simp [runJobs,call,Result.bind] at run
    | ok v u =>
      cases v
      have hj := legal j (by simp)
      have first := source_law M c x j s u (rt.success j s u hj call)
      have tailRun : runJobs invoke js u=.ok () t := by simpa [runJobs,call,Result.bind] using run
      have rest := ih (fun k hk => legal k (by simp [hk])) u tailRun
      refine ⟨?_,?_⟩
      · intro k hk d hd
        by_cases tail : k∈js
        · exact rest.1 k tail d hd
        · have kj : k=j := by simpa [tail] using hk
          subst k
          have saved := rest.2 (address c j d) (by
            intro k hk hit
            obtain ⟨e,he,eq⟩ := hit
            have same := ownership c j k hj (legal k (by simp [hk])) d e hd he eq
            exact tail (same.1.symm ▸ hk))
          exact saved.trans (first.1 d hd)
      · intro a outside
        exact (rest.2 a (fun k hk => outside k (by simp [hk]))).trans
          (first.2 a (by intro k hk;exact outside k (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp hk)))))

theorem jobs_all_outcomes {α} [Inhabited α] (M : Algebra α) (c : Call) (x : Input α)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : LocalRuntime M c x invoke)
    (jobs : List Job) (legal : ∀ j∈jobs,Legal c j) (s : Nat → α) :
    Frame c jobs s (runJobs invoke jobs s).state := by
  induction jobs generalizing s with
  | nil => exact fun _ _ => rfl
  | cons j js ih =>
    have hj := legal j (by simp)
    cases call : invoke j s with
    | error e u =>
      have first := rt.error j s e u hj call
      intro a outside
      simpa [runJobs,call,Result.bind,Result.state] using
        first a (by intro k hk;exact outside k (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp hk))))
    | ok v u =>
      cases v
      have first := (source_law M c x j s u (rt.success j s u hj call)).2
      have rest := ih (fun k hk => legal k (by simp [hk])) u
      intro a outside
      simpa [runJobs,call,Result.bind] using
        (rest a (fun k hk => outside k (by simp [hk]))).trans
          (first a (by intro k hk;exact outside k (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp hk)))))

#axiomsClean source_law
#axiomsClean source_total
#axiomsClean ownership
#axiomsClean jobs_return
#axiomsClean jobs_all_outcomes
end
end VeriTile.Bench.Repairs.SoftmaxReduceV.Host
