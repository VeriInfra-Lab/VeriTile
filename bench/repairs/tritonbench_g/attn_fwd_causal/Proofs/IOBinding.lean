import bench.repairs.tritonbench_g.attn_fwd_causal.IO
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.AccessSafety
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.FiniteReadback

namespace VeriTile.Bench.Repairs.AttnFwdCausal.IOBinding
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited α] [Inhabited β]

theorem launch_initial (M : Algebra β) (dot : Dot β) (p : Parameters) (pid : Program) (s : State β) :
    (IO.launch M dot p pid).initial s=start (grid p) pid s := by rfl

theorem run_eq (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program) (s : State β) :
    runSchedule (work.map (IO.launch M dot p)) s=
      runPrograms (fun _=>M) (fun _=>dot) p (grid p) work s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,runPrograms,Invocation.run,launch_initial,IO.launch]
    congr 1
    funext t
    exact ih t

theorem bounds_for (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : Fin 128 → β → α) (vals : IO.Inputs α p)
    (bounds : RegionName → Nat) (bounded : (IO.program M dot p work encode observe).InBounds bounds vals)
    (pid : Program) (legal : pid∈programs (grid p)) (s : State β) :
    AccessSafety.Bounds p (start (grid p) pid s) bounds := by
  obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) pid.1 pid.2.1 pid.2.2).mp legal
  obtain ⟨hh,hb,hd⟩ := head_decode p pid.2.1 hy
  have bx : pid.1*128<p.length := by change pid.1<(p.length+127)/128 at hx;omega
  have get (i : Fin (IO.inputs p).length) (j : TileIndex ((IO.inputs p)[i]).shape) := bounded.2.1 i j trivial
  constructor
  · intro i j hi
    simpa only [IO.program,IO.inputs,IO.inputAddress,Fin.cases_zero,start,ids,base,
      ↓reduceIte,Fin.val_zero,Fin.coe_ofNat_eq_mod,Fin.cases_succ] using
      get ⟨0,by simp [IO.inputs,IO.program]⟩ (⟨_,hb⟩,⟨_,hd⟩,⟨_,hi⟩,j,())
  · simpa only [IO.program,IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,↓reduceIte] using
      get ⟨3,by simp [IO.inputs,IO.program]⟩ (⟨_,hy⟩,⟨_,hx⟩,())
  · intro row active j
    simpa only [IO.program,IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,base,↓reduceIte] using
      get ⟨1,by simp [IO.inputs,IO.program]⟩ (⟨_,hb⟩,⟨_,hd⟩,⟨row,active⟩,j,())
  · intro row active j
    simpa only [IO.program,IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,base,↓reduceIte] using
      get ⟨2,by simp [IO.inputs,IO.program]⟩ (⟨_,hb⟩,⟨_,hd⟩,⟨row,active⟩,j,())
  · intro row active
    have slot : row/64<(p.length+63)/64 := by omega
    simpa only [IO.program,IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,↓reduceIte] using
      get ⟨4,by simp [IO.inputs,IO.program]⟩ (⟨_,hy⟩,⟨_,slot⟩,())
  · intro i j active
    simpa only [IO.program,IO.inputs,logicalAddress,Fin.cases_zero,start,ids,base,↓reduceIte] using
      bounded.2.2.1 ⟨0,by simp [IO.inputs,IO.program]⟩ (⟨_,hb⟩,⟨_,hd⟩,⟨_,active⟩,j,()) trivial

theorem schedule_safe (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : Fin 128 → β → α) (vals : IO.Inputs α p)
    (bounds : RegionName → Nat) (bounded : (IO.program M dot p work encode observe).InBounds bounds vals)
    (jobs : List Program) (legal : ∀ pid∈jobs,pid∈programs (grid p)) (s : State β) :
    ScheduleSafe bounds (jobs.map (IO.launch M dot p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨?_,fun next _=>ih (fun k hk=>legal k (by simp [hk])) next⟩
    change TraceSafe _ _ _ ((IO.launch M dot p pid).initial s)
    rw [launch_initial]
    apply AccessSafety.source_safe
    · exact bounds_for M dot p work encode observe vals bounds bounded pid (legal pid (by simp)) s
    · have hx := ((mem_programs (grid p) pid.1 pid.2.1 pid.2.2).mp (legal pid (by simp))).1
      change pid.1<(p.length+127)/128 at hx
      change pid.1*128<p.length
      omega

theorem bound_read (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : Fin 128 → β → α) (vals : IO.Inputs α p) (s : State β)
    (binding : (IO.program M dot p work encode observe).InputBinding vals s)
    (i : Fin (IO.inputs p).length) (j : TileIndex ((IO.inputs p)[i]).shape) :
    (s.mem (IO.inputs p)[i].buf (IO.inputAddress p i j)).read .real=encode (IO.inputValue p vals i j) := by
  fin_cases i
  all_goals exact binding _ j trivial

theorem lookup_read (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : Fin 128 → β → α) (vals : IO.Inputs α p) (s : State β)
    (binding : (IO.program M dot p work encode observe).InputBinding vals s)
    (i : Fin (IO.inputs p).length) (j : TileIndex ((IO.inputs p)[i]).shape) :
    (s.mem (IO.inputs p)[i].buf (IO.inputAddress p i j)).read .real=
      encode (Reference.inputAt p vals (IO.inputs p)[i].buf (IO.inputAddress p i j)) := by
  have present : ∃ x,Reference.HasValue p vals (IO.inputs p)[i].buf (IO.inputAddress p i j) x :=
    ⟨IO.inputValue p vals i j,i,j,rfl,rfl,rfl⟩
  rw [Reference.inputAt,dif_pos present]
  obtain ⟨k,q,region,offset,value⟩ := Classical.choose_spec present
  simpa only [region,offset,value] using bound_read M dot p work encode observe vals s binding k q

/-- The local source recurrence only needs equality of finite input ports. -/
theorem inputs_agree (p : Parameters) (pid : Program) (legal : pid∈programs (grid p))
    (s t : State β)
    (reads : ∀ (i : Fin (IO.inputs p).length) (j : TileIndex ((IO.inputs p)[i]).shape),
      (s.mem (IO.inputs p)[i].buf (IO.inputAddress p i j)).read .real=
      (t.mem (IO.inputs p)[i].buf (IO.inputAddress p i j)).read .real) :
    FiniteReadback.Agrees p (start (grid p) pid s) (start (grid p) pid t) := by
  obtain ⟨hx,hy,hz⟩ := (mem_programs (grid p) pid.1 pid.2.1 pid.2.2).mp legal
  obtain ⟨hh,hb,hd⟩ := head_decode p pid.2.1 hy
  constructor
  · rfl
  · intro i j hi
    simpa only [IO.inputs,IO.inputAddress,Fin.cases_zero,start,ids,base,↓reduceIte] using
      reads ⟨0,by simp [IO.inputs,IO.program]⟩ (⟨_,hb⟩,⟨_,hd⟩,⟨_,hi⟩,j,())
  · simpa only [IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,↓reduceIte] using
      reads ⟨3,by simp [IO.inputs,IO.program]⟩ (⟨_,hy⟩,⟨_,hx⟩,())
  · intro row active j
    simpa only [IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,base,↓reduceIte] using
      reads ⟨1,by simp [IO.inputs,IO.program]⟩ (⟨_,hb⟩,⟨_,hd⟩,⟨row,active⟩,j,())
  · intro row active j
    simpa only [IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,base,↓reduceIte] using
      reads ⟨2,by simp [IO.inputs,IO.program]⟩ (⟨_,hb⟩,⟨_,hd⟩,⟨row,active⟩,j,())
  · intro row active
    have slot : row/64<(p.length+63)/64 := by omega
    simpa only [IO.inputs,IO.inputAddress,Fin.cases_zero,Fin.cases_succ,start,ids,↓reduceIte] using
      reads ⟨4,by simp [IO.inputs,IO.program]⟩ (⟨_,hy⟩,⟨_,slot⟩,())

end
end VeriTile.Bench.Repairs.AttnFwdCausal.IOBinding
