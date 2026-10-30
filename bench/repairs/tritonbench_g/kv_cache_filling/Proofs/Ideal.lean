import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Source
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Real
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Execution
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling.Ideal

/-- Separate logical input/output roles for the mathematical source
interpretation. Values are arbitrary Reals, with no byte codec or finiteness
restriction. The checked disjoint physical roles justify this separation. -/
structure State where
  (key value : Nat → Nat → ℝ)
  cache : Address → Cell ℝ Byte

noncomputable def rowStores (r : Request) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (s : State) (j : Job) : List (Store (Cell ℝ Byte)) :=
  (rowAddresses r j).map fun a => ⟨a,sourceTaskValue r realOps kp vp kb vb s.key s.value j a⟩

theorem rowStores_eq (r : Request) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (s : State) (j : Job) : rowStores r kp vp kb vb s j=taskStores r realOps kp vp s.key s.value j := by
  apply List.map_congr_left
  intro a ha
  simp only [sourceTaskValue_eq r realOps kp vp kb vb s.key s.value j a ha]

/-- Each program executes the masked/reduction/scalar source graph and writes
its logical cache cells. Subsequent programs read the current input fields. -/
noncomputable def runJobs (r : Request) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv) :
    List Job → State → State
  | [],s => s
  | j::js,s => runJobs r kp vp kb vb js
      { s with cache := runStores (rowStores r kp vp kb vb s j) s.cache }

theorem execution (r : Request) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (js : List Job) (s : State) :
    (runJobs r kp vp kb vb js s).key=s.key ∧
    (runJobs r kp vp kb vb js s).value=s.value ∧
    (runJobs r kp vp kb vb js s).cache=
      runStores (js.flatMap (taskStores r realOps kp vp s.key s.value)) s.cache := by
  induction js generalizing s with
  | nil => exact ⟨rfl,rfl,rfl⟩
  | cons j js ih =>
    simp only [runJobs]
    have h := ih {s with cache := runStores (rowStores r kp vp kb vb s j) s.cache}
    refine ⟨h.1,h.2.1,?_⟩
    rw [h.2.2]
    simp only [List.flatMap_cons,runStores_append,rowStores_eq]

structure Result where
  state : State
  returned : Option Unit
  objects : List Nat
  device : VeriTile.Torch.Device

noncomputable def call (r : Request) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (s : State) (objects : List Nat) (incoming : VeriTile.Torch.Device) : Result :=
  ⟨runJobs r kp vp kb vb (jobs r) s,none,objects,incoming⟩

theorem call_execution (r : Request) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (s : State) (objects : List Nat) (incoming : VeriTile.Torch.Device) :
    (call r kp vp kb vb s objects incoming).state.key=s.key ∧
    (call r kp vp kb vb s objects incoming).state.value=s.value ∧
    (call r kp vp kb vb s objects incoming).state.cache=execute r realOps kp vp s.key s.value s.cache :=
  execution r kp vp kb vb (jobs r) s

#axiomsClean execution
#axiomsClean call_execution
end VeriTile.Bench.Repairs.KvCacheFilling.Ideal
