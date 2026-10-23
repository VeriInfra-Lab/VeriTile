import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.IO
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.AccessSafety
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelSchedule
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section
set_option maxHeartbeats 3000000

theorem invocation_initial (M : Algebra A) (op : Operation) (Q K D b r : Nat) (s : State A) :
    (invocation M op Q K D b r).initial s = atProgram s b r := by
  have hp : (fun ax => if decide (ax = 0 ∨ ax = 1) then (if ax = 0 then b else r) else s.pids ax) =
      (fun ax => if ax = 0 then b else if ax = 1 then r else s.pids ax) := by
    funext ax
    by_cases h0 : ax = 0 <;> by_cases h1 : ax = 1 <;> simp [h0,h1]
  simpa only [invocation, Invocation.initial, bindRegisters, List.foldl_nil,
    Bool.false_eq_true, if_false, Option.getD_none, atProgram] using
    congrArg (fun p => { s with pids := p }) hp

theorem invocation_run [Inhabited A] (M : Algebra A) (op : Operation) (Q K D b r : Nat) (s : State A) :
    (invocation M op Q K D b r).run s = KernelExecution.row M op Q K D b r s := by
  rw [Invocation.run, invocation_initial]
  rfl

theorem schedule_run [Inhabited A] (M : Algebra A) (op : Operation) (B Q K D : Nat)
    (jobs : List (Fin B × Fin Q)) (s : State A) :
    runSchedule (jobs.map (fun p => invocation M op Q K D p.1 p.2)) s =
      KernelExecution.rows M op B Q K D jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons p ps ih => simp only [List.map_cons, runSchedule, invocation_run, KernelExecution.rows, ih]

theorem selectedRows_complete (B Q K : Nat) (b : Fin B) (r : Fin Q) (live : 0<K) :
    (b,r) ∈ selectedRows B Q K := by
  have nonempty : B*Q*K ≠ 0 := Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos (by have := b.isLt; omega) (by have := r.isLt; omega)) live)
  simp only [selectedRows,nonempty,if_false]
  exact programs_mem B Q b r

theorem selected_run [Inhabited A] (M : Algebra A) (op : Operation) (B Q K D : Nat) (s : State A) :
    runSchedule ((selectedRows B Q K).map (fun p => invocation M op Q K D p.1 p.2)) s =
      RawKernel.run M op B Q K D s := by
  rw [schedule_run]
  by_cases h : B*Q*K=0 <;> simp [selectedRows, RawKernel.run, h, KernelExecution.rows]

theorem value_cell (M : Algebra A) (op : Operation) (K r : Nat) {D : Nat}
    (x y : Fin D → A) (i : Fin D) :
    value M op K r x y i = Cell.mk .real (orderedRow M op K r x y i) := by
  cases op <;> rfl

theorem value_congr (M : Algebra A) (op : Operation) (K r : Nat) {D : Nat}
    (x x' y y' : Fin D → A)
    (hx : ∀ j, Forward.readActive K r op.causal j.val=Bool.true → x j=x' j)
    (hy : ∀ j, op.readsRight=Bool.true → Forward.readActive K r op.causal j.val=Bool.true → y j=y' j)
    (i : Fin D) : value M op K r x y i = value M op K r x' y' i := by
  cases op with
  | forward log causal mode => exact forwardValue_congr M K r log causal mode _ _ _ _ hx hy i
  | backward log causal => exact backwardValue_congr M K r log causal _ _ _ _ hx (fun j hj => hy j rfl hj) i

theorem schedule_safe {A V : Type} [Inhabited A] (M : Algebra A) (op : Operation)
    (B Q K D : Nat) (positive : 0<D) (jobs : List (Fin B × Fin Q))
    (encode : RegionName → Nat → V → A) (observe : A → V)
    (v : Inputs V B Q D) (bounds : RegionName → Nat)
    (bounded : (addressedIO M op B Q K D jobs encode observe).InBounds bounds v) (s : State A) :
    ScheduleSafe bounds (jobs.map (fun p => invocation M op Q K D p.1 p.2)) s := by
  have leftBound : ∀ (b : Fin B) (r : Fin Q) (i : Fin D),
      Forward.readActive K r op.causal i.val=Bool.true → Native.flat Q K b r i.val < bounds "left" := by
    intro b r i hi
    exact bounded.2.1 ⟨0,by simp [addressedIO,memory,inputs]⟩ (b,r,i,PUnit.unit) ⟨Or.inl rfl,hi⟩
  have rightBound : ∀ (b : Fin B) (r : Fin Q) (i : Fin D), op.readsRight=Bool.true →
      Forward.readActive K r op.causal i.val=Bool.true → rightAddress op Q K b r i.val < bounds "right" := by
    intro b r i hr hi
    exact bounded.2.1 ⟨1,by simp [addressedIO,memory,inputs]⟩ (b,r,i,PUnit.unit) ⟨Or.inr hr,hi⟩
  have outputBound : ∀ (b : Fin B) (r : Fin Q) (i : Fin D), i.val<K →
      Native.flat Q K b r i.val < bounds "dst" := by
    intro b r i hi
    exact bounded.2.2.1 ⟨0,by simp [addressedIO,memory,outputs]⟩ (b,r,i,PUnit.unit) hi
  clear bounded
  induction jobs generalizing s with
  | nil => trivial
  | cons p ps ih =>
    refine ⟨?_,fun t _ => ih t⟩
    change TraceSafe (FP.ProfiledRewrite.engine M) bounds (KernelExecution.source op Q K D)
      ((invocation M op Q K D p.1 p.2).initial s)
    rw [invocation_initial]
    apply source_safe M op Q K D positive (atProgram s p.1 p.2) bounds
    · simpa only [atProgram,if_pos rfl,show ¬(1:Nat)=0 by decide,if_false] using leftBound p.1 p.2
    · simpa only [atProgram,if_pos rfl,show ¬(1:Nat)=0 by decide,if_false] using rightBound p.1 p.2
    · simpa only [atProgram,if_pos rfl,show ¬(1:Nat)=0 by decide,if_false] using outputBound p.1 p.2

theorem outside_old (M : Algebra A) (op : Operation) (B Q K D : Nat) (covered : K≤D)
    (jobs : List (Fin B × Fin Q)) (encode : RegionName → Nat → V → A) (observe : A → V)
    (v : Inputs V B Q D) (r : RegionName) (offset : Nat)
    (outside : (addressedIO M op B Q K D jobs encode observe).Outside v r offset) :
    r≠"dst" ∨ B*Q*K≤offset := by
  by_cases same : r="dst"
  · subst r
    right
    by_contra h
    obtain ⟨b,q,i,hi,eq⟩ := KernelSchedule.coordinate B Q K D offset covered (by omega)
    have miss := outside.1 ⟨0,by simp [addressedIO,memory,outputs]⟩ (b,q,i,PUnit.unit) hi
    exact miss.elim (fun h => h rfl) (fun h => h eq.symm)
  · exact Or.inl same

/-- Reuses the complete existing row execution and value/frame proof. Safety
is discharged on the actual preceding source states for every invocation. -/
theorem addressed_implements {A V : Type} [Inhabited A] (M : Algebra A) (op : Operation)
    (B Q K D : Nat) (positive : 0<D) (covered : K≤D)
    (jobs : List (Fin B × Fin Q)) (complete : ∀ b r, 0<K → (b,r)∈jobs)
    (encode : RegionName → Nat → V → A) (observe : A → V) :
    (addressedIO M op B Q K D jobs encode observe).Implements
      (orderedTarget M op B Q K D encode observe) := by
  apply AddressedProgramIO.Implements.intro
  · intro v bounds bounded s _
    exact schedule_safe M op B Q K D positive jobs encode observe v bounds bounded s
  · intro v s bound
    refine ⟨runRows M op "dst" "left" "right" B Q K D positive jobs s, ?_, ?_, ?_⟩
    · exact (schedule_run M op B Q K D jobs s).trans
        (KernelExecution.rows_success M op B Q K D positive jobs s)
    · intro p j active
      have hp : p.val<1 := p.isLt
      have pz : p=⟨0,by simp [addressedIO,memory,outputs]⟩ := Fin.ext (show p.val=0 by omega)
      subst p
      have active' : j.2.2.1.val<K := active
      change mapCell observe ((runRows M op "dst" "left" "right" B Q K D positive jobs s).mem
        "dst" (Native.flat Q K j.1 j.2.1 j.2.2.1)) =
        Cell.mk .real (observe (orderedRow M op K j.2.1
          (fun i => encode "left" (Native.flat Q K j.1 j.2.1 i.val) (left v j.1 j.2.1 i))
          (fun i => encode "right" (rightAddress op Q K j.1 j.2.1 i.val) (right v j.1 j.2.1 i)) j.2.2.1))
      rw [rows_observe M op "dst" "left" "right" B Q K D positive jobs s j.1 j.2.1 j.2.2.1
        active' (by decide) (by decide), if_pos (complete j.1 j.2.1 (by omega))]
      have eq := value_congr M op K j.2.1
        (fun i => (s.mem "left" (Native.flat Q K j.1 j.2.1 i.val)).read .real)
        (fun i => encode "left" (Native.flat Q K j.1 j.2.1 i.val) (left v j.1 j.2.1 i))
        (fun i => (s.mem "right" (rightAddress op Q K j.1 j.2.1 i.val)).read .real)
        (fun i => encode "right" (rightAddress op Q K j.1 j.2.1 i.val) (right v j.1 j.2.1 i))
        (fun i hi => bound ⟨0,by simp [addressedIO,memory,inputs]⟩
          (j.1,j.2.1,i,PUnit.unit) ⟨Or.inl rfl,hi⟩)
        (fun i hr hi => bound ⟨1,by simp [addressedIO,memory,inputs]⟩
          (j.1,j.2.1,i,PUnit.unit) ⟨Or.inr hr,hi⟩) j.2.2.1
      rw [show gridValue M op Q K j.1 j.2.1 (fun i => (s.mem "left" i).read .real)
          (fun i => (s.mem "right" i).read .real) j.2.2.1 = _ from eq, value_cell]
      rfl
    · intro r offset outside
      exact rows_frame M op "dst" "left" "right" B Q K D positive jobs s r offset
        (outside_old M op B Q K D covered jobs encode observe v r offset outside)

theorem program_implements {A : Type} [Inhabited A] (M : Algebra A) (op : Operation)
    (B Q K D : Nat) (positive : 0<D) (covered : K≤D)
    (jobs : List (Fin B × Fin Q)) (complete : ∀ b r, 0<K → (b,r)∈jobs) :
    (programIO M op B Q K D jobs).Implements
      (orderedTarget M op B Q K D (fun _ _ => id) id) :=
  addressed_implements M op B Q K D positive covered jobs complete (fun _ _ => id) id

#axiomsClean addressed_implements
#axiomsClean program_implements
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
