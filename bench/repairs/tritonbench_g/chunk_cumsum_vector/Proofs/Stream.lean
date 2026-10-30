import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Storage

/-! Generic execution composition, instantiated independently by the proved
Real and FP source executions. It retains complete cells in every frame. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Stream
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
abbrev Heap (β : Type) := Nat → β

def StableInput {β : Type} (h : Call) (p : Storage.Placement h) (a b : Heap β) : Prop :=
  ∀ off, off<p.backing.input.extent → b (Storage.address p.backing.input off)=a (Storage.address p.backing.input off)
def run {β : Type} {h : Call} (single : Config → Geometry.Program h → Heap β → Option (Heap β)) (c : Config) :
    List (Geometry.Program h) → Heap β → Option (Heap β)
  | [],s => some s
  | prog::rest,s => (single c prog s).bind (run single c rest)

theorem complete {β γ : Type} (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (read : β → γ) (single : Config → Geometry.Program h → Heap β → Option (Heap β))
    (expected : Config → Heap β → Geometry.Program h → Nat → Fin BS → γ)
    (works : ∀ c∈configurations, ∀ prog s, ∃ t, single c prog s=some t ∧ StableInput h p s t ∧
      (∀ time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected c s prog time j) ∧
      (∀ addr, ¬Storage.writes h p [prog] addr → t addr=s addr))
    (stable : ∀ c∈configurations, ∀ a b, StableInput h p a b → ∀ prog time j, Grid.active h prog time j →
      expected c b prog time j=expected c a prog time j)
    (c : Config) (chosen : c∈configurations) (order : List (Geometry.Program h)) (s : Heap β) :
    ∃ t, run single c order s=some t ∧ StableInput h p s t ∧
      (∀ prog∈order, ∀ time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected c s prog time j) ∧
      (∀ addr, ¬Storage.writes h p order addr → t addr=s addr) := by
  classical
  induction order generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,by simp,fun _ _ => rfl⟩
  | cons prog rest ih =>
    obtain ⟨mid,he,input,out,frame⟩ := works c chosen prog s
    obtain ⟨t,ht,input',out',frame'⟩ := ih mid
    refine ⟨t,by simp only [run,he,Option.bind_some,ht],
      fun off hi => (input' off hi).trans (input off hi),?_,?_⟩
    · intro current member time j ha
      have eq := stable c chosen s mid input current time j ha
      rcases List.mem_cons.mp member with same | member
      · subst current
        by_cases member : prog∈rest
        · exact (out' prog member time j ha).trans eq
        · have no : ¬Storage.writes h p rest (Storage.address p.backing.output (Grid.offset h prog time j)) := by
            rintro ⟨other,ho,t,k,hk,eq⟩
            have off := Storage.address_injective p.backing.output eq
            have same := (Grid.footprint_unique h layout prog other time t j k ha hk off).1
            exact member (same.symm ▸ ho)
          rw [frame' _ no]
          exact out time j ha
      · exact (out' current member time j ha).trans eq
    · intro addr outside
      have a : ¬Storage.writes h p [prog] addr := by
        rintro ⟨x,hx,time,j,ha,eq⟩
        simp only [List.mem_singleton] at hx
        subst x
        exact outside ⟨prog,by simp,time,j,ha,eq⟩
      have b : ¬Storage.writes h p rest addr := by
        rintro ⟨x,hx,time,j,ha,eq⟩
        exact outside ⟨x,List.mem_cons_of_mem prog hx,time,j,ha,eq⟩
      exact (frame' addr b).trans (frame addr a)

def trials {β : Type} {h : Call} (single : Config → Geometry.Program h → Heap β → Option (Heap β)) :
    List Config → Heap β → Option (Heap β)
  | [],s => some s
  | c::rest,s => (run single c (Grid.programs h) s).bind (trials single rest)

theorem trials_complete {β γ : Type} (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (read : β → γ) (single : Config → Geometry.Program h → Heap β → Option (Heap β))
    (expected : Config → Heap β → Geometry.Program h → Nat → Fin BS → γ)
    (works : ∀ c∈configurations, ∀ prog s, ∃ t, single c prog s=some t ∧ StableInput h p s t ∧
      (∀ time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected c s prog time j) ∧
      (∀ addr, ¬Storage.writes h p [prog] addr → t addr=s addr))
    (stable : ∀ c∈configurations, ∀ a b, StableInput h p a b → ∀ prog time j, Grid.active h prog time j →
      expected c b prog time j=expected c a prog time j)
    (cs : List Config) (valid : ∀ c∈cs, c∈configurations) (s : Heap β) :
    ∃ t, trials single cs s=some t ∧ StableInput h p s t ∧
      (∀ addr, ¬Storage.writes h p (Grid.programs h) addr → t addr=s addr) := by
  induction cs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,fun _ _ => rfl⟩
  | cons c rest ih =>
    obtain ⟨mid,he,input,_,frame⟩ := complete h layout p read single expected works stable c (valid c (by simp)) (Grid.programs h) s
    obtain ⟨t,ht,input',frame'⟩ := ih (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
    exact ⟨t,by simp only [trials,he,Option.bind_some,ht],
      fun off hi => (input' off hi).trans (input off hi),fun addr hn => (frame' addr hn).trans (frame addr hn)⟩

def tuned {β : Type} {h : Call} (single : Config → Geometry.Program h → Heap β → Option (Heap β))
    (tried : List Config) (selected : Config) (s : Heap β) : Option (Heap β) :=
  (trials single tried s).bind (run single selected (Grid.programs h))

theorem tuned_complete {β γ : Type} (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (read : β → γ) (single : Config → Geometry.Program h → Heap β → Option (Heap β))
    (expected : Config → Heap β → Geometry.Program h → Nat → Fin BS → γ)
    (works : ∀ c∈configurations, ∀ prog s, ∃ t, single c prog s=some t ∧ StableInput h p s t ∧
      (∀ time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected c s prog time j) ∧
      (∀ addr, ¬Storage.writes h p [prog] addr → t addr=s addr))
    (stable : ∀ c∈configurations, ∀ a b, StableInput h p a b → ∀ prog time j, Grid.active h prog time j →
      expected c b prog time j=expected c a prog time j)
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations) (s : Heap β) :
    ∃ t, tuned single tried selected s=some t ∧ StableInput h p s t ∧
      (∀ prog time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected selected s prog time j) ∧
      (∀ addr, ¬Storage.writes h p (Grid.programs h) addr → t addr=s addr) := by
  obtain ⟨mid,he,input,frame⟩ := trials_complete h layout p read single expected works stable tried valid s
  obtain ⟨t,ht,input',out,frame'⟩ := complete h layout p read single expected works stable selected chosen (Grid.programs h) mid
  exact ⟨t,by simp only [tuned,he,Option.bind_some,ht],fun off hi => (input' off hi).trans (input off hi),
    fun prog time j ha => (out prog (Grid.mem_programs h prog) time j ha).trans (stable selected chosen s mid input prog time j ha),
    fun addr hn => (frame' addr hn).trans (frame addr hn)⟩

#axiomsClean complete
#axiomsClean tuned_complete
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.Stream
