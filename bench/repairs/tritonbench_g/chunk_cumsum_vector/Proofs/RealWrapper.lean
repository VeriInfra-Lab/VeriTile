import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealStorage
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Host

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealWrapper
open VeriTile.Triton RealStorage
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

abbrev Oracle (h : Call) := Nat → Config → Geometry.Program h → RegionName → Nat → ℝ

def grid (h : Call) (p : Storage.Placement h) (padding : Oracle h)
    (n : Nat) (c : Config) (s : Heap) : Option Heap :=
  Stream.run (execute h p (padding n)) c (Grid.programs h) s

theorem grid_complete (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (padding : Oracle h) (n : Nat) (c : Config) (chosen : c∈configurations) (s : Heap) :
    ∃ t, grid h p padding n c s=some t ∧ Host.Good h p s t ∧
      (∀ prog time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected h p c s prog time j) := by
  obtain ⟨t,he,input,out,frame⟩ := Stream.complete h layout p read (execute h p (padding n)) (expected h p)
    (RealStorage.complete h layout p (padding n))
    (fun c _ a b hs prog time j ha => expected_stable h p c a b hs prog time j ha)
    c chosen (Grid.programs h) s
  exact ⟨t,he,⟨input,frame⟩,fun prog time j ha => out prog (Grid.mem_programs h prog) time j ha⟩

def formula (h : Call) (logical : Nat → Nat → Nat → ℝ) (row time channel : Nat) : ℝ :=
  ∑ k ∈ (Finset.range h.T).filter (fun k => k ≤ time),logical row k channel

theorem expected_logical (h : Call) (p : Storage.Placement h) (c : Config) (heap : Heap)
    (logical : Nat → Nat → Nat → ℝ)
    (input : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
      read (heap (Storage.address p.backing.input (h.inputStrides.address h.H row time channel)))=logical row time channel)
    (row : Fin (h.B*h.H)) (time : Fin h.T) (channel : Fin h.S) :
    expected h p c heap (Grid.owner h row channel) time.val (Grid.lane h channel)=formula h logical row.val time.val channel.val := by
  unfold expected RealData.expected formula
  rw [Grid.owner_coordinate]
  apply Finset.sum_congr rfl
  intro k hk
  exact input row.val k channel.val row.isLt (Finset.mem_range.mp (Finset.mem_filter.mp hk).1) channel.isLt

/-- Actual wrapper return: the precise empty_like identity/metadata, every
logical prefix, caller live storage, and all storage outside preparation. -/
theorem complete (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (padding : Oracle h) (api : Host.API MemCell) (admitted : Bool)
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : Heap) (live : Nat → Prop) (logical : Nat → Nat → Nat → ℝ)
    (laws : Host.PreparationLaws api h p read logical caller live)
    (a : Host.Allocated MemCell) (prepared : Host.prepare api admitted caller=.ok a)
    (setups : a.shape.prod≠0 → Host.Setups api h 0 (tried++[selected])) :
    ∃ t, Host.run api (grid h p padding) h admitted tried selected caller=.ok {a with heap:=t} ∧
      Host.metadata h p {a with heap:=t} ∧
      (∀ (row : Fin (h.B*h.H)) (time : Fin h.T) (channel : Fin h.S),
        read (t (Storage.address p.backing.output (h.outputStrides.address h.H row.val time.val channel.val)))=
          formula h logical row.val time.val channel.val) ∧
      (∀ addr, live addr → t addr=caller addr) ∧
      (∀ addr, ¬Host.preparationWrites h p addr → t addr=caller addr) ∧
      (∀ addr, ¬Storage.writes h p (Grid.programs h) addr → t addr=a.heap addr) := by
  obtain ⟨metadata,inputs,callerFrame,prepareFrame⟩ := Host.prepare_correct api h p read logical caller live laws admitted a prepared
  by_cases empty : a.shape.prod=0
  · refine ⟨a.heap,?_,metadata,?_,callerFrame,prepareFrame,fun _ _ => rfl⟩
    · simpa using Host.empty_return api (grid h p padding) h admitted tried selected caller a prepared empty
    · intro row time channel
      have product : h.B*h.H*h.T*h.S=0 := by
        rw [metadata.1] at empty
        simpa [Nat.mul_assoc] using empty
      have positive := Nat.mul_pos (Nat.mul_pos (Nat.zero_lt_of_lt row.isLt) (Nat.zero_lt_of_lt time.isLt)) (Nat.zero_lt_of_lt channel.isLt)
      omega
  · obtain ⟨t,he,good,out⟩ := Host.selected_complete h p read (grid h p padding)
      (fun _ => expected h p) (grid_complete h layout p padding)
      (fun _ c _ a b hs prog time j ha => expected_stable h p c a b hs prog time j ha)
      tried valid selected chosen a.heap
    refine ⟨t,Host.successful_return api (grid h p padding) h admitted tried selected caller a prepared empty t he (setups empty),
      metadata,?_,?_,?_,good.2⟩
    · intro row time channel
      have ha : Grid.active h (Grid.owner h row channel) time.val (Grid.lane h channel) :=
        ⟨time.isLt,by rw [Grid.owner_coordinate]; exact channel.isLt⟩
      have output := out (Grid.owner h row channel) time.val (Grid.lane h channel) ha
      have eq := expected_logical h p selected a.heap logical inputs row time channel
      unfold Grid.offset at output
      rw [Grid.owner_coordinate] at output
      exact output.trans eq
    · intro addr hl
      exact (good.2 addr (laws.fresh addr hl)).trans (callerFrame addr hl)
    · intro addr hn
      have no : ¬Storage.writes h p (Grid.programs h) addr := fun hw => hn (Host.output_within_preparation h p _ addr hw)
      exact (good.2 addr no).trans (prepareFrame addr hn)

theorem launches_outcome (h : Call) (layout : Geometry.OutputLayout h) (p : Storage.Placement h)
    (padding : Oracle h) (api : Host.API MemCell) (n : Nat) (cs : List Config)
    (valid : ∀ c∈cs, c∈configurations) (s : Heap) :
    match Host.launches api h (grid h p padding) n cs s with
    | .ok t => Host.Good h p s t
    | .error e => e.stage=.setup ∧ Host.Good h p s e.heap := by
  have works : ∀ n c, c∈configurations → ∀ s, ∃ t, grid h p padding n c s=some t ∧ Host.Good h p s t := by
    intro n c hc s
    obtain ⟨t,he,hg,_⟩ := grid_complete h layout p padding n c hc s
    exact ⟨t,he,hg⟩
  have result := Host.launches_outcome api h (grid h p padding) (Host.Good h p)
    (Host.good_refl h p) (Host.good_trans h p) works n cs valid s
  cases eq : Host.launches api h (grid h p padding) n cs s <;> simpa only [eq] using result

#axiomsClean complete
#axiomsClean launches_outcome
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealWrapper
