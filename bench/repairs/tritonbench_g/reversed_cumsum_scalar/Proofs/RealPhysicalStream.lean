import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealStorage

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealPhysicalStream
open VeriTile.Triton RealStorage
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

abbrev Row (h : Call) := Fin (h.B*h.H)

def run (h : Call) (p : Storage.Placement h) (c : Config) :
    List (Row h) → Heap → Option (Heap)
  | [],s => some s
  | row::rest,s => (execute h p c row.val s).bind (run h p c rest)

def StableInput (h : Call) (p : Storage.Placement h) (a b : Heap) : Prop :=
  ∀ off, off<p.backing.input.extent → b (Storage.address p.backing.input off)=a (Storage.address p.backing.input off)

theorem expected_stable (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (a b : Heap) (stable : StableInput h p a b)
    (row : Row h) (time : Nat) (ht : time<h.T) : expected h p c positive b row.val time=expected h p c positive a row.val time := by
  unfold expected
  apply congrArg MemCell.real
  unfold VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix
    VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.initialSum
  have eq : ∀ k, k<h.T → input h p b row.val k=input h p a row.val k := by
    intro k hk
    unfold input
    rw [stable _ (p.backing.inputWindow _ _ row.isLt hk)]
  congr 1
  · apply Finset.sum_congr rfl
    intro k hk
    exact eq k (Finset.mem_range.mp hk)
  · apply Finset.sum_congr rfl
    intro k hk
    exact eq k ((Finset.mem_range.mp hk).trans ht)


theorem complete (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (c : Config) (positive : 0<c.BT) (rows : List (Row h)) (s : Heap) :
    ∃ t, run h p c rows s=some t ∧ StableInput h p s t ∧
      (∀ row∈rows, ∀ time, time<h.T → t (Storage.address p.backing.output (h.outputStrides.address h.H row.val time))=
        expected h p c positive s row.val time) ∧
      (∀ addr, ¬Storage.writes h p rows addr → t addr=s addr) := by
  induction rows generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,by simp,fun _ _ => rfl⟩
  | cons row rest ih =>
    obtain ⟨mid,he,stable,out,frame⟩ := RealStorage.complete h layout p c positive row s
    obtain ⟨t,ht,stable',out',frame'⟩ := ih mid
    refine ⟨t,?_,fun off hi => (stable' off hi).trans (stable off hi),?_,?_⟩
    · simp only [run,he,Option.bind_some,ht]
    · intro r member time hv
      have eq := expected_stable h p c positive s mid stable r time hv
      rcases List.mem_cons.mp member with same | member
      · subst r
        by_cases member : row∈rest
        · exact (out' row member time hv).trans eq
        · have no : ¬Storage.writes h p rest (Storage.address p.backing.output (h.outputStrides.address h.H row.val time)) := by
            rintro ⟨r,hr,t,ht,eq⟩
            have addr := Storage.address_injective p.backing.output eq
            have same := (layout.unique _ _ _ _ row.isLt r.isLt hv ht addr).1
            exact member (Fin.ext same ▸ hr)
          rw [frame' _ no]
          exact out time hv
      · exact (out' r member time hv).trans eq
    · intro addr outside
      have a : ¬Storage.writes h p [row] addr := by
        rintro ⟨r,hr,time,ht,eq⟩
        simp only [List.mem_singleton] at hr
        subst r
        exact outside ⟨row,by simp,time,ht,eq⟩
      have b : ¬Storage.writes h p rest addr := by
        rintro ⟨r,hr,time,ht,eq⟩
        exact outside ⟨r,List.mem_cons_of_mem row hr,time,ht,eq⟩
      exact (frame' addr b).trans (frame addr a)

def rows (h : Call) : List (Row h) := List.finRange (h.B*h.H)

def trials (h : Call) (p : Storage.Placement h) :
    List Config → Heap → Option (Heap)
  | [],s => some s
  | c::rest,s => (run h p c (rows h) s).bind (trials h p rest)

theorem trials_complete (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (configs : List Config) (valid : ∀ c∈configs, 0<c.BT) (s : Heap) :
    ∃ t, trials h p configs s=some t ∧ StableInput h p s t ∧
      (∀ addr, ¬Storage.writes h p (rows h) addr → t addr=s addr) := by
  induction configs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,fun _ _ => rfl⟩
  | cons c rest ih =>
    obtain ⟨mid,he,stable,_,frame⟩ := complete h layout p c (valid c (by simp)) (rows h) s
    obtain ⟨t,ht,stable',frame'⟩ := ih (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
    exact ⟨t,by simp only [trials,he,Option.bind_some,ht],
      fun off hi => (stable' off hi).trans (stable off hi),fun addr hn => (frame' addr hn).trans (frame addr hn)⟩

def tuned (h : Call) (p : Storage.Placement h)
    (tried : List Config) (selected : Config) (s : Heap) : Option (Heap) :=
  (trials h p tried s).bind (run h p selected (rows h))

theorem tuned_complete (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations) (s : Heap) :
    ∃ t, tuned h p tried selected s=some t ∧ StableInput h p s t ∧
      (∀ row : Row h, ∀ time, time<h.T → t (Storage.address p.backing.output (h.outputStrides.address h.H row.val time))=
        expected h p selected (Kernels.configurations_positive selected chosen).1 s row.val time) ∧
      (∀ addr, ¬Storage.writes h p (rows h) addr → t addr=s addr) := by
  obtain ⟨mid,he,stable,frame⟩ := trials_complete h layout p tried
    (fun c hc => (Kernels.configurations_positive c (valid c hc)).1) s
  obtain ⟨t,ht,stable',out,frame'⟩ := complete h layout p selected
    (Kernels.configurations_positive selected chosen).1 (rows h) mid
  refine ⟨t,by simp only [tuned,he,Option.bind_some,ht],fun off hi => (stable' off hi).trans (stable off hi),?_,
    fun addr hn => (frame' addr hn).trans (frame addr hn)⟩
  intro row time hv
  exact (out row (by simp [rows]) time hv).trans (expected_stable h p selected _ s mid stable row time hv)

#print axioms complete
#print axioms tuned_complete
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealPhysicalStream
