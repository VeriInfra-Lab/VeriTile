import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPReference
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Host

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPWrapper
open VeriTile.Triton FP.Structural FPValues FPStorage
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]
abbrev Oracle (α : Type) (h : Call) := Nat → Config → Geometry.Program h → Padding α

def grid (M : Config → Algebra α) (dot : Config → DotBackend α) (h : Call) (p : Storage.Placement h)
    (padding : Oracle α h) (n : Nat) (c : Config) (s : Heap α) : Option (Heap α) :=
  Stream.run (execute (M c) (dot c) h p (padding n)) c (Grid.programs h) s

theorem grid_complete (M : Config → Algebra α) (dot : Config → DotBackend α)
    (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (padding : Oracle α h) (n : Nat) (c : Config) (chosen : c∈configurations) (s : Heap α) :
    ∃ t, grid M dot h p padding n c s=some t ∧ Host.Good h p s t ∧
      (∀ prog time j, Grid.active h prog time j →
        t (Storage.address p.backing.output (Grid.offset h prog time j))=
          expected (M c) (dot c) h p (padding n) c s prog time j) := by
  obtain ⟨t,he,input,out,frame⟩ := Stream.complete h layout p id
    (execute (M c) (dot c) h p (padding n)) (expected (M c) (dot c) h p (padding n))
    (FPStorage.complete (M c) (dot c) h layout p (padding n))
    (fun cfg _ a b hs prog time j _ => expected_stable (M c) (dot c) h p (padding n) cfg a b hs prog time j)
    c chosen (Grid.programs h) s
  exact ⟨t,he,⟨input,frame⟩,fun prog time j ha => out prog (Grid.mem_programs h prog) time j ha⟩

/-- Full wrapper returns the exact allocation with the selected invocation's
structural operation graph on logical inputs and explicit zero padding. The
unused oracle may vary per invocation; trial results do not enter the reference. -/
theorem complete (M : Config → Algebra α) (dot : Config → DotBackend α) (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (padding : Oracle α h) (api : Host.API (Cell α)) (admitted : Bool)
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : Heap α) (live : Nat → Prop) (logical : Nat → Nat → Nat → α)
    (laws : Host.PreparationLaws api h p (FPReference.read h) logical caller live)
    (a : Host.Allocated (Cell α)) (prepared : Host.prepare api admitted caller=.ok a)
    (setups : a.shape.prod≠0 → Host.Setups api h 0 (tried++[selected])) :
    ∃ t, Host.run api (grid M dot h p padding) h admitted tried selected caller=.ok {a with heap:=t} ∧
      Host.metadata h p {a with heap:=t} ∧
      (∀ (row : Fin (h.B*h.H)) (time : Fin h.T) (channel : Fin h.S),
        t (Storage.address p.backing.output (h.outputStrides.address h.H row.val time.val channel.val))=
          FPReference.expected (M selected) (dot selected) h selected (Kernels.configurations_positive selected chosen).1
            (Grid.owner h row channel) logical (padding tried.length selected (Grid.owner h row channel))
            time.val (Grid.lane h channel)) ∧
      (∀ addr, live addr → t addr=caller addr) ∧
      (∀ addr, ¬Host.preparationWrites h p addr → t addr=caller addr) ∧
      (∀ addr, ¬Storage.writes h p (Grid.programs h) addr → t addr=a.heap addr) := by
  obtain ⟨metadata,inputs,callerFrame,prepareFrame⟩ := Host.prepare_correct api h p (FPReference.read h) logical caller live laws admitted a prepared
  by_cases empty : a.shape.prod=0
  · refine ⟨a.heap,?_,metadata,?_,callerFrame,prepareFrame,fun _ _ => rfl⟩
    · simpa using Host.empty_return api (grid M dot h p padding) h admitted tried selected caller a prepared empty
    · intro row time channel
      have product : h.B*h.H*h.T*h.S=0 := by
        rw [metadata.1] at empty
        simpa [Nat.mul_assoc] using empty
      have positive := Nat.mul_pos (Nat.mul_pos (Nat.zero_lt_of_lt row.isLt) (Nat.zero_lt_of_lt time.isLt)) (Nat.zero_lt_of_lt channel.isLt)
      omega
  · obtain ⟨t,he,good,out⟩ := Host.selected_complete h p id (grid M dot h p padding)
      (fun n c => expected (M c) (dot c) h p (padding n) c) (grid_complete M dot h layout p padding)
      (fun n c _ a b hs prog time j _ => expected_stable (M c) (dot c) h p (padding n) c a b hs prog time j)
      tried valid selected chosen a.heap
    refine ⟨t,Host.successful_return api (grid M dot h p padding) h admitted tried selected caller a prepared empty t he (setups empty),
      metadata,?_,?_,?_,good.2⟩
    · intro row time channel
      have ha : Grid.active h (Grid.owner h row channel) time.val (Grid.lane h channel) :=
        ⟨time.isLt,by rw [Grid.owner_coordinate]; exact channel.isLt⟩
      have output := out (Grid.owner h row channel) time.val (Grid.lane h channel) ha
      have eq := FPReference.expected_logical (M selected) (dot selected) h p (padding tried.length) selected
        (Kernels.configurations_positive selected chosen).1 (Grid.owner h row channel) a.heap logical inputs time.val (Grid.lane h channel)
      unfold Grid.offset at output
      rw [Grid.owner_coordinate] at output
      exact output.trans eq
    · intro addr hl
      exact (good.2 addr (laws.fresh addr hl)).trans (callerFrame addr hl)
    · intro addr hn
      have no : ¬Storage.writes h p (Grid.programs h) addr := fun hw => hn (Host.output_within_preparation h p _ addr hw)
      exact (good.2 addr no).trans (prepareFrame addr hn)

theorem launches_outcome (M : Config → Algebra α) (dot : Config → DotBackend α) (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (padding : Oracle α h) (api : Host.API (Cell α)) (n : Nat) (cs : List Config)
    (valid : ∀ c∈cs, c∈configurations) (s : Heap α) :
    match Host.launches api h (grid M dot h p padding) n cs s with
    | .ok t => Host.Good h p s t
    | .error e => e.stage=.setup ∧ Host.Good h p s e.heap := by
  have works : ∀ n c, c∈configurations → ∀ s, ∃ t, grid M dot h p padding n c s=some t ∧ Host.Good h p s t := by
    intro n c hc s
    obtain ⟨t,he,hg,_⟩ := grid_complete M dot h layout p padding n c hc s
    exact ⟨t,he,hg⟩
  have result := Host.launches_outcome api h (grid M dot h p padding) (Host.Good h p)
    (Host.good_refl h p) (Host.good_trans h p) works n cs valid s
  cases eq : Host.launches api h (grid M dot h p padding) n cs s <;> simpa only [eq] using result

#axiomsClean complete
#axiomsClean launches_outcome
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPWrapper
