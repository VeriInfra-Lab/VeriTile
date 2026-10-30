import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealPhysicalStream
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Host

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealWrapper
open VeriTile.Triton RealStorage
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

def grid (h : Call) (p : Storage.Placement h) : Config → Heap → Option (Heap) :=
  fun c => RealPhysicalStream.run h p c (RealPhysicalStream.rows h)

def Good (h : Call) (p : Storage.Placement h) (s t : Heap) : Prop :=
  RealPhysicalStream.StableInput h p s t ∧ (∀ addr, ¬Storage.writes h p (RealPhysicalStream.rows h) addr → t addr=s addr)

theorem good_refl (h : Call) (p : Storage.Placement h) (s : Heap) : Good h p s s :=
  ⟨fun _ _ => rfl,fun _ _ => rfl⟩
theorem good_trans (h : Call) (p : Storage.Placement h) (a b c : Heap)
    (hab : Good h p a b) (hbc : Good h p b c) : Good h p a c :=
  ⟨fun off hi => (hbc.1 off hi).trans (hab.1 off hi),fun addr hn => (hbc.2 addr hn).trans (hab.2 addr hn)⟩

theorem grid_complete (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (c : Config) (hc : c∈configurations) (s : Heap) :
    ∃ t, grid h p c s=some t ∧ Good h p s t := by
  obtain ⟨t,he,stable,_,frame⟩ := RealPhysicalStream.complete h layout p c
    (Kernels.configurations_positive c hc).1 (RealPhysicalStream.rows h) s
  exact ⟨t,he,stable,frame⟩

theorem raw_complete (h : Call)
    (layout : Geometry.OutputLayout h) (p : Storage.Placement h) (cs : List Config)
    (valid : ∀ c∈cs, c∈configurations) (s : Heap) :
    ∃ t, Host.raw (grid h p) cs s=some t ∧ Good h p s t := by
  induction cs generalizing s with
  | nil => exact ⟨s,rfl,good_refl h p s⟩
  | cons c rest ih =>
    obtain ⟨mid,he,good⟩ := grid_complete h layout p c (valid c (by simp)) s
    obtain ⟨t,ht,good'⟩ := ih (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
    exact ⟨t,by simp only [Host.raw,he,Option.bind_some,ht],good_trans h p s mid t good good'⟩

theorem tuned_complete (h : Call)
    (layout : Geometry.OutputLayout h) (p : Storage.Placement h) (tried : List Config)
    (valid : ∀ c∈tried, c∈configurations) (selected : Config) (chosen : selected∈configurations) (s : Heap) :
    ∃ t, Host.raw (grid h p) (tried++[selected]) s=some t ∧ Good h p s t ∧
      (∀ row : RealPhysicalStream.Row h, ∀ time, time<h.T →
        t (Storage.address p.backing.output (h.outputStrides.address h.H row.val time))=
          expected h p selected (Kernels.configurations_positive selected chosen).1 s row.val time) := by
  obtain ⟨mid,he,good⟩ := raw_complete h layout p tried valid s
  obtain ⟨t,ht,stable,out,frame⟩ := RealPhysicalStream.complete h layout p selected
    (Kernels.configurations_positive selected chosen).1 (RealPhysicalStream.rows h) mid
  refine ⟨t,?_,good_trans h p s mid t good ⟨stable,frame⟩,?_⟩
  · rw [Host.raw_append,he]
    simp only [Option.bind_some,Host.raw,grid,ht]
  · intro row time hv
    exact (out row (by simp [RealPhysicalStream.rows]) time hv).trans
      (RealPhysicalStream.expected_stable h p selected _ s mid good.1 row time hv)

def ReturnedPost (h : Call) (p : Storage.Placement h)
    (selected : Config) (positive : 0<selected.BT) (a result : Host.Allocated MemCell) : Prop :=
  result.identity=a.identity ∧ result.shape=a.shape ∧ result.strides=a.strides ∧
    result.dtype=a.dtype ∧ result.base=a.base ∧ result.extent=a.extent ∧ Good h p a.heap result.heap ∧
    (∀ row : RealPhysicalStream.Row h, ∀ time, time<h.T →
      result.heap (Storage.address p.backing.output (h.outputStrides.address h.H row.val time))=
        expected h p selected positive a.heap row.val time)

theorem ready_outcomes (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (api : Host.API MemCell) (admitted : Bool) (tried : List Config)
    (valid : ∀ c∈tried, c∈configurations) (selected : Config) (chosen : selected∈configurations)
    (caller : Heap) (a : Host.Allocated MemCell) (prepared : Host.prepare api admitted caller=.ok a) :
    match Host.run api (grid h p) h admitted tried selected caller with
    | .error e => e.stage=.setup ∧ Good h p a.heap e.heap
    | .ok result => ReturnedPost h p selected (Kernels.configurations_positive selected chosen).1 a result := by
  by_cases empty : h.B*h.H*h.T=0
  · rw [Host.empty_return api (grid h p) h admitted tried selected caller a prepared empty]
    refine ⟨rfl,rfl,rfl,rfl,rfl,rfl,good_refl h p a.heap,?_⟩
    intro row time ht
    have hr := row.isLt
    have hb : 0<h.B*h.H := Nat.zero_lt_of_lt hr
    have hp : 0<h.B*h.H*h.T := Nat.mul_pos hb (Nat.zero_lt_of_lt ht)
    omega
  · have validAll : ∀ c∈tried++[selected], c∈configurations := by
      intro c hc
      rcases List.mem_append.mp hc with hc | hc
      · exact valid c hc
      · have eq : c=selected := List.mem_singleton.mp hc
        simpa only [eq] using chosen
    have safe := Host.launches_outcome api (grid h p) (Good h p) (good_refl h p) (good_trans h p)
      (grid_complete h layout p) 0 (tried++[selected]) validAll a.heap
    obtain ⟨expectedFinal,raw,good,out⟩ := tuned_complete h layout p tried valid selected chosen a.heap
    simp only [Host.run,prepared,empty,if_false]
    cases launchEq : Host.launches api (grid h p) 0 (tried++[selected]) a.heap with
    | error e => simpa only [launchEq] using safe
    | ok result =>
      have rawResult := Host.launches_success_raw api (grid h p) 0 (tried++[selected]) a.heap result launchEq
      rw [raw] at rawResult
      have same : expectedFinal=result := Option.some.inj rawResult
      subst result
      exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,good,out⟩

/-- Compose the actual original logical input and allocated return with the
complete kernel/tuner proof, and retain caller/all-other-cell frames. -/
theorem returned_logical (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (api : Host.API MemCell) (admitted : Bool) (tried : List Config)
    (valid : ∀ c∈tried, c∈configurations) (selected : Config) (chosen : selected∈configurations)
    (caller : Heap) (logical : Nat → Nat → ℝ) (live : Nat → Prop)
    (laws : Host.PreparationLaws api h p RealStorage.read logical caller live)
    (a : Host.Allocated MemCell) (prepared : Host.prepare api admitted caller=.ok a)
    (result : Host.Allocated MemCell)
    (returned : Host.run api (grid h p) h admitted tried selected caller=.ok result) :
    result.identity=a.identity ∧ Host.metadata h p result ∧
    (∀ row : RealPhysicalStream.Row h, ∀ time, time<h.T →
      result.heap (Storage.address p.backing.output (h.outputStrides.address h.H row.val time))=
        MemCell.real (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix (logical row.val) h.T time)) ∧
    Good h p a.heap result.heap ∧
    (∀ addr, live addr → result.heap addr=caller addr) ∧
    (∀ addr, ¬Host.preparationWrites h p addr → result.heap addr=caller addr) := by
  have post := ready_outcomes h layout p api admitted tried valid selected chosen caller a prepared
  rw [returned] at post
  obtain ⟨identity,shape,strides,dtype,base,extent,good,out⟩ := post
  obtain ⟨metadataOk,coordinates,liveFrame,fullFrame⟩ := Host.prepare_correct api h p _ logical caller live laws admitted a prepared
  refine ⟨identity,?_,?_,good,?_,?_⟩
  · simpa only [Host.metadata,shape,strides,dtype,base,extent] using metadataOk
  · intro row time ht
    rw [out row time ht]
    unfold expected
    apply congrArg MemCell.real
    unfold VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix
      VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.initialSum
    congr 1
    · apply Finset.sum_congr rfl
      intro k hk
      exact coordinates row.val k row.isLt (Finset.mem_range.mp hk)
    · apply Finset.sum_congr rfl
      intro k hk
      exact coordinates row.val k row.isLt ((Finset.mem_range.mp hk).trans ht)
  · intro addr hl
    exact (good.2 addr (laws.fresh addr hl)).trans (liveFrame addr hl)
  · intro addr hn
    have outside : ¬Storage.writes h p (RealPhysicalStream.rows h) addr := fun hw => hn (Host.output_within_preparation h p _ addr hw)
    exact (good.2 addr outside).trans (fullFrame addr hn)

#print axioms ready_outcomes
#print axioms returned_logical
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealWrapper
