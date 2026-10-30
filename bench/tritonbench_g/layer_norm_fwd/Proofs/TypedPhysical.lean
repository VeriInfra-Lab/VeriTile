import bench.tritonbench_g.layer_norm_fwd.Proofs.AliasMemory
import VeriTile.Triton.Float.Structural

/-! Opaque tagged cells over the existing byte-start layout. Reuses the
checked Layout/FreshWrites address discipline; it is not a raw-byte codec. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedPhysical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open AliasMemory
noncomputable section
open scoped Classical
variable {α : Type}
set_option maxHeartbeats 2000000

theorem state_ext {s t : State α} (hm : s.mem=t.mem) (hr : s.regs=t.regs)
    (hp : s.pids=t.pids) (hn : s.numPids=t.numPids) (hu : s.undef=t.undef) : s=t := by
  cases s
  cases t
  simp_all

abbrev Physical (α : Type) (a : Layout) := Fin a.capacity → Cell α

def read (a : Layout) (p : Physical α a) (k : Key) : Cell α :=
  if h : a.live k then p ⟨a.address k, a.bounded k h⟩ else Cell.mk .nat 0

def pull (a : Layout) (p : Physical α a) (template : State α) : State α :=
  { template with mem := fun r o => read a p (r,o) }

/-- A physical read shared by overlapping views has one value. -/
theorem alias_read (a : Layout) (p : Physical α a) (k l : Key)
    (hk : a.live k) (hl : a.live l) (h : a.address k = a.address l) :
    read a p k = read a p l := by
  simp only [read, dif_pos hk, dif_pos hl]
  congr 1
  exact Fin.ext h

def write (a : Layout) (p : Physical α a) (k : Key) (v : Cell α) : Physical α a :=
  fun i => if i.val = a.address k then v else p i

/-- One actual store commutes with physical realization, including all
aliased live reads and the full complement. No read/read separation is used. -/
theorem store_commutes (a : Layout) (p : Physical α a) (s : State α)
    (writes : Key → Prop) (hf : FreshWrites a writes)
    (k : Key) (hk : writes k) (v : Cell α) :
    pull a (write a p k v) s = State.write (pull a p s) k.1 k.2 v := by
  apply state_ext
  · funext r o
    change read a (write a p k v) (r,o) =
      if r=k.1 ∧ o=k.2 then v else read a p (r,o)
    by_cases he : (r,o) = k
    · obtain ⟨rfl,rfl⟩ := Prod.mk.inj he
      simp [read,write,(hf k hk).1]
    · have hp : ¬(r=k.1 ∧ o=k.2) := by intro hc; exact he (Prod.ext hc.1 hc.2)
      rw [if_neg hp]
      by_cases hl : a.live (r,o)
      · have ha : a.address (r,o) ≠ a.address k := fun h => he ((hf k hk).2 _ hl h)
        simp [read,write,hl,ha]
      · simp [read,hl]
  · rfl
  · rfl
  · rfl
  · rfl

structure Event (α : Type) where
  key : Key
  value : Cell α

def runPhysical (a : Layout) (p : Physical α a) (events : List (Event α)) : Physical α a :=
  events.foldl (fun memory event => write a memory event.key event.value) p

def runTagged (s : State α) (events : List (Event α)) : State α :=
  events.foldl (fun state event => State.write state event.key.1 event.key.2 event.value) s

/-- Complete ordered store-trace simulation, not a final-output assumption. -/
theorem trace_commutes (a : Layout) (p : Physical α a) (s : State α)
    (writes : Key → Prop) (hf : FreshWrites a writes) (events : List (Event α))
    (he : ∀ e ∈ events, writes e.key) :
    pull a (runPhysical a p events) s = runTagged (pull a p s) events := by
  induction events generalizing p s with
  | nil => rfl
  | cons e es ih =>
    simp only [runPhysical,runTagged,List.foldl_cons]
    change pull a (runPhysical a (write a p e.key e.value) es) s = _
    rw [ih _ s (fun e' h => he e' (List.mem_cons_of_mem _ h))]
    rw [store_commutes a p s writes hf e.key (he e (List.mem_cons_self ..)) e.value]
    rfl

/-- Physical frame for every cell outside the actual emitted store addresses. -/
theorem trace_frame (a : Layout) (p : Physical α a) (events : List (Event α))
    (i : Fin a.capacity) (h : ∀ e ∈ events, i.val ≠ a.address e.key) :
    runPhysical a p events i = p i := by
  induction events generalizing p with
  | nil => rfl
  | cons e es ih =>
    simp only [runPhysical,List.foldl_cons]
    change runPhysical a (write a p e.key e.value) es i = p i
    rw [ih _ (fun e' h' => h e' (List.mem_cons_of_mem _ h'))]
    exact if_neg (h e (List.mem_cons_self ..))


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedPhysical
