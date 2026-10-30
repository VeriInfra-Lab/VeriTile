import bench.repairs.tritonbench_g.chunk_delta_fwd.Program
import bench.repairs.tritonbench_g.chunk_delta_fwd.ProgramReadback
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def grid (p : Parameters) : List PID :=
  (List.range (ceilDiv p.V p.BV)).product (List.range (p.B*p.H))
theorem mem_grid (p : Parameters) (pid : PID) : pid∈grid p ↔ Geometry.ValidPID p pid := by
  rcases pid with ⟨v,h⟩
  simp [grid,Geometry.ValidPID]

def launchState (p : Parameters) (pid : PID) (m : Memory α) : State α :=
  { memoryState m with
    pids := fun n => match n with | 0 => 0 | 1 => pid.1 | _ => pid.2
    numPids := fun n => match n with | 0 => 1 | 1 => ceilDiv p.V p.BV | _ => p.B*p.H }

/-- Each CTA executes against the memory produced by its predecessors. The
remaining generic backend obligation is realization of the declared grid. -/
def runGrid (M : Algebra α) (D : DotPrimitive α) (p : Parameters) :
    List PID → Memory α → Option (Memory α)
  | [],m => some m
  | pid::rest,m => do
      let t ← FP.Structural.exec (engine (withDot M D)) (kernel p) (launchState p pid m)
      runGrid M D p rest t.mem
def gridMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (schedule : List PID) (input current : Memory α) : Memory α :=
  schedule.foldl (fun mem pid => programMemory M D p pid input mem) current

theorem grid_execution (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (schedule : List PID) (input m : Memory α) (hi : SameInputs m input) :
    runGrid M D p schedule m=some (gridMemory M D p schedule input m) ∧
      SameInputs (gridMemory M D p schedule input m) input := by
  induction schedule generalizing m with
  | nil => exact ⟨rfl,hi⟩
  | cons pid rest ih =>
    obtain ⟨t,he,hm,ht⟩ := program M D p input (launchState p pid m) rfl hi
    have hr := ih t.mem ht
    change t.mem=programMemory M D p pid input m at hm
    constructor
    · simp only [runGrid,he,bind,Option.bind_some]
      simpa only [gridMemory,List.foldl_cons,hm] using hr.1
    · simpa only [gridMemory,List.foldl_cons,hm] using hr.2

theorem footprint_owner (p : Parameters) (a b : PID) (r : RegionName) (o : Nat)
    (ha : Footprint p a r o) (hb : Footprint p b r o) : a=b := by
  rcases ha with ⟨rfl,c,hc,i,hi,he⟩|⟨rfl,c,hc,k,hk,i,hi,he⟩|⟨rfl,hp,i,hi,he⟩
  · have hb' : ∃ d, d < p.NT ∧ ∃ j : HIndex p, hActive p b j ∧ o=hOffset p b d j := by
      simpa [Footprint] using hb
    obtain ⟨d,hd,j,hj,he'⟩ := hb'
    exact (Geometry.history_collision p a b c d i j hi hj hc hd (he.symm.trans he')).1
  · have hb' : ∃ d, d < p.NT ∧ ∃ l, l < innerCount p ∧ ∃ j : CIndex p,
        cActive p b d l j ∧ o=cOffset p b d l j := by simpa [Footprint] using hb
    obtain ⟨d,hd,l,hl,j,hj,he'⟩ := hb'
    exact (Geometry.corrected_collision p a b c d k l i j hi hj (he.symm.trans he')).1
  · have hb' : ∃ j : HIndex p, hActive p b j ∧ o=fOffset p b j := by
      simpa [Footprint,hp] using hb
    obtain ⟨j,hj,he'⟩ := hb'
    exact (Geometry.final_collision p a b i j hi hj (he.symm.trans he')).1

theorem list_observe {β γ ι : Type} (f : β → ι → β) (get : β → γ)
    (xs : List ι) (a : β) (h : ∀ i∈xs, ∀ b, get (f b i)=get b) :
    get (xs.foldl f a)=get a := by
  induction xs generalizing a with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih]
    · exact h i (by simp) a
    · intro j hj
      exact h j (by simp [hj])

theorem list_read {β γ ι : Type} [DecidableEq ι] (f : β → ι → β) (get : β → γ)
    (xs : List ι) (owner : ι) (a : β) (v : γ) (hm : owner∈xs)
    (hit : ∀ b, get (f b owner)=v)
    (other : ∀ i∈xs, i≠owner → ∀ b, get (f b i)=get b) : get (xs.foldl f a)=v := by
  have stable : ∀ ys : List ι, (∀ i∈ys, i≠owner → ∀ b, get (f b i)=get b) →
      ∀ b, get b=v → get (ys.foldl f b)=v := by
    intro ys hy b hb
    induction ys generalizing b with
    | nil => exact hb
    | cons i rest ih =>
      apply ih (fun j hj => hy j (by simp [hj])) (f b i)
      by_cases he : i=owner
      · subst i; exact hit b
      · rw [hy i (by simp) he, hb]
  induction xs generalizing a with
  | nil => simp at hm
  | cons i rest ih =>
    rw [List.foldl_cons]
    by_cases he : i=owner
    · subst i
      exact stable rest (fun j hj => other j (by simp [hj])) _ (hit a)
    · exact ih (f a i) (by simpa [Ne.symm he] using hm) (fun j hj => other j (by simp [hj]))

theorem grid_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (schedule : List PID) (input m : Memory α) (pid : PID) (r : RegionName) (o : Nat)
    (value : Cell α) (hm : pid∈schedule) (hw : Footprint p pid r o)
    (hit : ∀ mem, programMemory M D p pid input mem r o=value) :
    gridMemory M D p schedule input m r o=value := by
  apply list_read _ (fun mem : Memory α => mem r o) schedule pid m value hm hit
  intro other _ hne mem
  apply program_frame
  intro ho
  exact hne (footprint_owner p other pid r o ho hw)

theorem grid_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (schedule : List PID) (input m : Memory α) (r : RegionName) (o : Nat)
    (safe : ∀ pid∈schedule, ¬Footprint p pid r o) :
    gridMemory M D p schedule input m r o=m r o := by
  apply list_observe _ (fun mem : Memory α => mem r o) schedule m
  intro pid hp mem
  exact program_frame M D p pid input mem r o (safe pid hp)

#axiomsClean grid_execution
#axiomsClean footprint_owner
#axiomsClean grid_read
#axiomsClean grid_frame
#specNonCircular gridMemory avoiding [run,FP.Structural.exec,runGrid]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
