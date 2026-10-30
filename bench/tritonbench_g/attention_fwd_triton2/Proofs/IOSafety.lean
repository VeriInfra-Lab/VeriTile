import bench.tritonbench_g.attention_fwd_triton2.Proofs.CompleteSafety
import bench.tritonbench_g.attention_fwd_triton2.Proofs.InputValues

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.IOSafety
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface InputValues
noncomputable section
variable {α β : Type} [Inhabited β]

@[simp] theorem invocation_initial (M : Algebra β) (dot : Dot β) (p : Parameters)
    (program : Program) (s : State β) : (invocation M dot p program).initial s=start (grid p) program s := rfl

theorem schedule_runs (M : Algebra β) (dot : Dot β) (p : Parameters)
    (work : List Program) (s : State β) :
    runSchedule (work.map (invocation M dot p)) s=runPrograms (fun _=>M) (fun _=>dot) p (grid p) work s := by
  induction work generalizing s with
  | nil => rfl
  | cons program rest ih =>
    simp only [List.map_cons,runSchedule,runPrograms,Invocation.run,invocation_initial]
    change (exec (FP.ProfiledRewrite.engine (withDot M dot)) (kernel p) (start (grid p) program s)).bind _ = _
    cases ran : exec (FP.ProfiledRewrite.engine (withDot M dot)) (kernel p) (start (grid p) program s) <;> simp [ran,ih]

theorem read_capacity (M : Algebra β) (dot : Dot β) (p : Parameters)
    (encode : α→β) (observe : TileIndex [p.batches,p.heads,p.length,96]→β→α)
    (work : List Program) (vals : Inputs α p) (bounds : RegionName→Nat)
    (backed : (io M dot p encode observe work).InBounds bounds vals)
    (r : RegionName) (off : Nat) (read : ReadAddress p r off) : off<bounds r := by
  rcases read with ⟨rfl,inside⟩|⟨rfl,inside⟩|⟨rfl,inside⟩|⟨rfl,inside⟩|⟨rfl,inside⟩
  · exact backed.2.1 (0 : Fin 5) (⟨off,inside⟩,()) trivial
  · exact backed.2.1 (1 : Fin 5) (⟨off,inside⟩,()) trivial
  · exact backed.2.1 (2 : Fin 5) (⟨off,inside⟩,()) trivial
  · exact backed.2.1 (3 : Fin 5) (⟨off,inside⟩,()) trivial
  · exact backed.2.1 (4 : Fin 5) (⟨off,inside⟩,()) trivial

theorem source_bounds (M : Algebra β) (dot : Dot β) (p : Parameters) (canon : Canonical p)
    (encode : α→β) (observe : TileIndex [p.batches,p.heads,p.length,96]→β→α)
    (work : List Program) (vals : Inputs α p) (bounds : RegionName→Nat)
    (backed : (io M dot p encode observe work).InBounds bounds vals)
    (s : State β) (hx : s.pids 0<(p.length+127)/128) (hy : s.pids 1<p.batches*p.heads) :
    SourceSafety.Bounds p s bounds := by
  have safe := Access.safe p canon
  have cap := read_capacity M dot p encode observe work vals bounds backed
  constructor
  · intro i d hi hd
    apply cap
    exact Or.inl ⟨rfl,safe.query _ hx _ hy _ i.isLt _ d.isLt hi hd⟩
  · apply cap
    exact Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl,safe.queryScale _ hx _ hy⟩)))
  · intro row inside d j hj hd
    have hk := (safe.keyValue _ hy row inside _ j.isLt _ d.isLt hj hd).1
    apply cap
    apply Or.inr ∘ Or.inl
    refine ⟨rfl,?_⟩
    simpa [base,offset,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hk
  · intro row inside
    apply cap
    exact Or.inr (Or.inr (Or.inr (Or.inr ⟨rfl,safe.keyScale _ hy row inside⟩)))
  · intro row inside j d hj hd
    apply cap
    exact Or.inr (Or.inr (Or.inl ⟨rfl,(safe.keyValue _ hy row inside _ j.isLt _ d.isLt hj hd).2⟩))
  · intro i d hi hd
    obtain ⟨_,hb,hh⟩ := head_decode p (s.pids 1) hy
    exact backed.2.2.1 (0 : Fin 1)
      (⟨s.pids 1/p.heads,hb⟩,⟨s.pids 1%p.heads,hh⟩,⟨s.pids 0*128+i.val,hi⟩,⟨d.val,hd⟩,()) trivial

theorem schedule_safe (M : Algebra β) (dot : Dot β) (p : Parameters) (canon : Canonical p)
    (encode : α→β) (observe : TileIndex [p.batches,p.heads,p.length,96]→β→α)
    (work : List Program) (order : (programs (grid p)).Perm work) (vals : Inputs α p) (bounds : RegionName→Nat)
    (backed : (io M dot p encode observe work).InBounds bounds vals) (initial : State β) :
    ScheduleSafe bounds (work.map (invocation M dot p)) initial := by
  have all : ∀ jobs : List Program,(∀ program∈jobs,program∈programs (grid p)) →
      ∀ s : State β,ScheduleSafe bounds (jobs.map (invocation M dot p)) s := by
    intro jobs
    induction jobs with
    | nil => intros; trivial
    | cons program jobs ih =>
      intro valid s
      refine ⟨?_,fun t _=>ih (fun q hq=>valid q (by simp [hq])) t⟩
      change TraceSafe (FP.ProfiledRewrite.engine (withDot M dot)) bounds (kernel p) (start (grid p) program s)
      have member := (mem_programs _ program.1 program.2.1 program.2.2).mp (valid program (by simp))
      apply SourceSafety.source_safe
      apply source_bounds M dot p canon encode observe work vals bounds backed
      · simpa [start,ids,grid] using member.1
      · simpa [start,ids,grid] using member.2.1
  exact all work (fun program h=>order.mem_iff.mpr h) initial

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.IOSafety
