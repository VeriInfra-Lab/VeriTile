import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Ownership

namespace VeriTile.Bench.Repairs.RotaryEmbNopad
open Checked

variable {α : Type}

def component (value : α×α) (odd : Bool) := if odd then value.2 else value.1

def Writes (r : Request) (indices : Indices) (w : Work) (region : Region) (a : Nat) : Prop :=
  region=.other r.device ∧ (a=destination r indices w false ∨ a=destination r indices w true)

instance (r : Request) (indices : Indices) (w : Work) (region : Region) (a : Nat) :
    Decidable (Writes r indices w region a) := inferInstanceAs (Decidable (_∧(_∨_)))

theorem step_readback (p : Ops α) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) (m : Memory α) (odd : Bool) :
    sourceStep p c.request indices w m (.other c.request.device) (destination c.request indices w odd)=
      component (sourceValues p c.request w m) odd := by
  have different := different_destinations c indices accepted w active
  simp only [destination] at different ⊢
  cases odd <;> simp [sourceStep,component,different]

theorem step_frame (p : Ops α) (r : Request) (indices : Indices) (w : Work) (m : Memory α)
    (region : Region) (a : Nat) (outside : ¬Writes r indices w region a) :
    sourceStep p r indices w m region a=m region a := by
  simp only [Writes,destination,not_and,not_or] at outside
  unfold sourceStep
  by_cases same : region=.other r.device
  · simp [same,outside same]
  · simp [same]

theorem disjoint_cell (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w v : Work) (hw : c.request.Active w) (hv : c.request.Active v) (different : w≠v) (odd : Bool) :
    ¬Writes c.request indices w (.other c.request.device) (destination c.request indices v odd) := by
  rintro ⟨_,same|same⟩
  · exact different (destination_injective c indices accepted w v hw hv false odd same.symm).1
  · exact different (destination_injective c indices accepted w v hw hv true odd same.symm).1

theorem input_frame (p : Ops α) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w v : Work) (hw : c.request.Active w) (hv : c.request.Active v) (different : w≠v)
    (m : Memory α) (odd : Bool) :
    sourceStep p c.request indices w m (.other c.request.device)
      (c.request.data v.key v.token v.head (c.request.dim v.pair odd))=
    m (.other c.request.device) (c.request.data v.key v.token v.head (c.request.dim v.pair odd)) := by
  apply step_frame
  rintro ⟨_,same|same⟩
  · exact other_input_preserved c indices accepted w v hw hv different false odd same.symm
  · exact other_input_preserved c indices accepted w v hw hv different true odd same.symm

theorem coefficient_frame (p : Ops α) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w v : Work) (hw : c.request.Active w) (hv : c.request.Active v) (m : Memory α) (sine : Bool) :
    sourceStep p c.request indices w m (.other c.request.device) (c.request.coefficient sine v.token v.pair)=
      m (.other c.request.device) (c.request.coefficient sine v.token v.pair) := by
  apply step_frame
  rintro ⟨_,same|same⟩
  · exact coefficient_preserved c indices accepted w v hw hv false sine same.symm
  · exact coefficient_preserved c indices accepted w v hw hv true sine same.symm

theorem values_stable (p : Ops α) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w v : Work) (hw : c.request.Active w) (hv : c.request.Active v) (different : w≠v) (m : Memory α) :
    sourceValues p c.request v (sourceStep p c.request indices w m)=sourceValues p c.request v m := by
  unfold sourceValues
  rw [input_frame p c indices accepted w v hw hv different m false,
    input_frame p c indices accepted w v hw hv different m true,
    coefficient_frame p c indices accepted w v hw hv m false,
    coefficient_frame p c indices accepted w v hw hv m true]

def execute (p : Ops α) (r : Request) (indices : Indices) : List Work → Memory α → Memory α
  | [],m => m
  | w::rest,m => execute p r indices rest (sourceStep p r indices w m)

theorem execute_frame (p : Ops α) (r : Request) (indices : Indices) (jobs : List Work) (m : Memory α)
    (region : Region) (a : Nat) (outside : ∀ w∈jobs,¬Writes r indices w region a) :
    execute p r indices jobs m region a=m region a := by
  induction jobs generalizing m with
  | nil => rfl
  | cons w rest ih =>
    change execute p r indices rest (sourceStep p r indices w m) region a=m region a
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    exact step_frame p r indices w m region a (outside w (by simp))

theorem execute_readback (p : Ops α) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (jobs : List Work) (m : Memory α) (active : ∀ w∈jobs,c.request.Active w) (unique : jobs.Nodup)
    (v : Work) (member : v∈jobs) (odd : Bool) :
    execute p c.request indices jobs m (.other c.request.device) (destination c.request indices v odd)=
      component (sourceValues p c.request v m) odd := by
  induction jobs generalizing m with
  | nil => simp at member
  | cons w rest ih =>
    have nd := List.nodup_cons.mp unique
    have hw := active w (by simp)
    have hv := active v member
    rcases List.mem_cons.mp member with same|tail
    · subst v
      change execute p c.request indices rest (sourceStep p c.request indices w m) _ _=_
      rw [execute_frame p c.request indices rest _ _ _ (by
        intro v hv'
        exact disjoint_cell c indices accepted v w (active v (by simp [hv'])) hw
          (by intro eq; subst v; exact nd.1 hv') odd)]
      exact step_readback p c indices accepted w hw m odd
    · have different : w≠v := by intro eq; subst v; exact nd.1 tail
      change execute p c.request indices rest (sourceStep p c.request indices w m) _ _=_
      rw [ih _ (fun w hw => active w (by simp [hw])) nd.2 tail,
        values_stable p c indices accepted w v hw hv different m]

#axiomsClean step_readback
#axiomsClean execute_readback
#axiomsClean execute_frame
end VeriTile.Bench.Repairs.RotaryEmbNopad
