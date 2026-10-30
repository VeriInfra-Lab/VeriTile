import bench.tritonbench_g.attention_forward_triton.Proofs.LogicalInputs

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite Source Whole
noncomputable section

theorem source_bounds {A : Type} (p : Parameters) (s : State A) (pid : Whole.Program)
    (legal : pid∈programs (grid p)) (bounds : RegionName → Nat)
    (input : ∀ i j, inputAddress p i j<bounds (inputChannels p)[i].buf)
    (output : ∀ j : DataLane p, dataAddress p.o j<bounds p.Out) :
    Access.Bounds p (start (grid p) pid s) bounds := by
  have occursBound (r : RegionName) (off : Nat) (occurs : Occurs p r off) : off<bounds r := by
    obtain ⟨⟨i,j⟩,hr,ho⟩ := occurs
    simpa only [hr,ho] using input i j
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro i active
    exact occursBound _ _ (occurs_query p s pid legal i active)
  · exact occursBound _ _ (occurs_queryScale p s pid legal)
  · intro row _ i active
    exact occursBound _ _ (occurs_key p s pid legal row i active)
  · intro row active
    exact occursBound _ _ (occurs_keyScale p s pid legal row active)
  · intro row _ i active
    exact occursBound _ _ (occurs_value p s pid legal row i active)
  · intro i active
    rcases pid with ⟨x,y,z⟩
    obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) x y z).mp legal
    obtain ⟨_,hb,hh⟩ := head_decode p y hy
    change x*128+i.1.val<p.length ∧ i.2.1.val<96 at active
    exact output (⟨y/p.heads,hb⟩,⟨y%p.heads,hh⟩,⟨x*128+i.1.val,active.1⟩,
      ⟨i.2.1.val,active.2⟩,())

theorem invocation_initial {A : Type} (M : Algebra A) (dot : Dot A)
    (p : Parameters) (pid : Whole.Program) (s : State A) :
    (invocation M dot p pid).initial s=start (grid p) pid s := rfl

theorem schedule_run {A : Type} [Inhabited A] (M : Algebra A) (dot : Dot A)
    (p : Parameters) (work : List Whole.Program) (s : State A) :
    runSchedule (work.map (invocation M dot p)) s=
      runPrograms (fun _ => M) (fun _ => dot) p (grid p) work s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,invocation_initial,runPrograms]
    exact congrArg (Option.bind (FP.Structural.exec (engine (withDot M dot)) (Source.kernel p)
      (start (grid p) pid s))) (funext ih)

theorem schedule_safe {A : Type} [Inhabited A] (M : Algebra A) (dot : Dot A)
    (p : Parameters) (work : List Whole.Program)
    (legal : ∀ pid∈work,pid∈programs (grid p)) (bounds : RegionName → Nat)
    (input : ∀ i j, inputAddress p i j<bounds (inputChannels p)[i].buf)
    (output : ∀ j : DataLane p, dataAddress p.o j<bounds p.Out) (s : State A) :
    ScheduleSafe bounds (work.map (invocation M dot p)) s := by
  induction work generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨?_,fun t _ => ih (fun pid member => legal pid (by simp [member])) t⟩
    change TraceSafe (engine (withDot M dot)) bounds (Source.kernel p) _
    rw [invocation_initial]
    exact Access.source_safe M dot p _ bounds
      (source_bounds p s pid (legal pid (by simp)) bounds input output)

#axiomsClean schedule_safe
#axiomsClean schedule_run
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.AttentionIO
