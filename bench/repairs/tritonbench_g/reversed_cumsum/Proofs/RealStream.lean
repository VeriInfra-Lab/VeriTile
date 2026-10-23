import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealStorage
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Host

namespace VeriTile.Bench.Repairs.ReversedCumsum.RealStream
open VeriTile.Triton Geometry RealStorage
noncomputable section
set_option maxHeartbeats 4800000
set_option maxRecDepth 24000

def run (h : Call) (p : Storage.Placement h) (c : Config) : List (Program h) → Heap → Option Heap
  | [],s => some s
  | program::rest,s => (execute h p c program s).bind (run h p c rest)

def StableInput (h : Call) (p : Storage.Placement h) (a b : Heap) : Prop :=
  ∀ off, off<p.backing.input.extent → b (Storage.address p.backing.input off)=a (Storage.address p.backing.input off)

theorem expected_stable (h : Call) (p : Storage.Placement h) (a b : Heap) (stable : StableInput h p a b)
    (row : Row h) (t s : Nat) (ht : t<h.T) (hs : s<h.S) :
    expected h p b row.val t s=expected h p a row.val t s := by
  unfold expected
  apply congrArg MemCell.real
  unfold VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffixValue
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.initialSum
  have eq : ∀ k, k<h.T → input h p b row.val k s=input h p a row.val k s := by
    intro k hk
    unfold input
    rw [stable _ (window h p row k s hk hs).1]
  congr 1
  · apply Finset.sum_congr rfl
    intro k hk
    exact eq k (Finset.mem_range.mp hk)
  · apply Finset.sum_congr rfl
    intro k hk
    exact eq k ((Finset.mem_range.mp hk).trans ht)

theorem complete (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (programs : List (Program h)) (s : Heap) :
    ∃ t, run h p c programs s=some t ∧ StableInput h p s t ∧
      (∀ program∈programs, ∀ time, ∀ j : Fin BS, time<h.T → program.2.val*BS+j.val<h.S →
        t (Storage.address p.backing.output (Geometry.address h program.1.val time (program.2.val*BS+j.val)))=
          expected h p s program.1.val time (program.2.val*BS+j.val)) ∧
      (∀ addr, ¬Storage.writes h p programs addr → t addr=s addr) := by
  induction programs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,by simp,fun _ _ => rfl⟩
  | cons program rest ih =>
    obtain ⟨mid,he,stable,out,frame⟩ := RealStorage.complete h distinct p c positive program s
    obtain ⟨t,ht,stable',out',frame'⟩ := ih mid
    refine ⟨t,by simp only [run,he,Option.bind_some,ht],fun off hi => (stable' off hi).trans (stable off hi),?_,?_⟩
    · intro r member time j hv hs
      have eq := expected_stable h p s mid stable r.1 time _ hv hs
      rcases List.mem_cons.mp member with same | member
      · subst r
        by_cases member : program∈rest
        · exact (out' program member time j hv hs).trans eq
        · have no : ¬Storage.writes h p rest (Storage.address p.backing.output
              (Geometry.address h program.1.val time (program.2.val*BS+j.val))) := by
            rintro ⟨r,hr,u,k,hu,hk,eq⟩
            have addr := ReversedCumsumScalar.Storage.address_injective p.backing.output eq
            have same := Storage.program_unique h program r time u j k hv hu hs hk addr
            exact member (same ▸ hr)
          rw [frame' _ no]
          exact out time j hv hs
      · exact (out' r member time j hv hs).trans eq
    · intro addr outside
      have a : ¬Storage.writes h p [program] addr := by
        rintro ⟨r,hr,time,j,ht,hs,eq⟩
        simp only [List.mem_singleton] at hr
        subst r
        exact outside ⟨program,by simp,time,j,ht,hs,eq⟩
      have b : ¬Storage.writes h p rest addr := by
        rintro ⟨r,hr,time,j,ht,hs,eq⟩
        exact outside ⟨r,List.mem_cons_of_mem program hr,time,j,ht,hs,eq⟩
      exact (frame' addr b).trans (frame addr a)

def grid (h : Call) (p : Storage.Placement h) (c : Config) := run h p c (Storage.programs h)

theorem grid_complete (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (s : Heap) :
    ∃ t, grid h p c s=some t ∧ StableInput h p s t ∧
      (∀ row : Row h, ∀ time feature, time<h.T → feature<h.S →
        t (Storage.address p.backing.output (Geometry.address h row.val time feature))=expected h p s row.val time feature) ∧
      (∀ addr, ¬Storage.writes h p (Storage.programs h) addr → t addr=s addr) := by
  obtain ⟨t,he,stable,out,frame⟩ := complete h distinct p c positive (Storage.programs h) s
  refine ⟨t,he,stable,?_,frame⟩
  intro row time feature ht hs
  obtain ⟨f,j,eq⟩ := Storage.feature_decompose h feature hs
  have hv := out (row,f) (by simp [Storage.programs]) time j ht (by simpa [eq] using hs)
  simpa only [eq] using hv

def trials (h : Call) (p : Storage.Placement h) := ReversedCumsumScalar.Host.raw (grid h p)

theorem trials_complete (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (configs : List Config) (valid : ∀ c∈configs,0<c.BT) (s : Heap) :
    ∃ t, trials h p configs s=some t ∧ StableInput h p s t ∧
      (∀ addr, ¬Storage.writes h p (Storage.programs h) addr → t addr=s addr) := by
  induction configs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,fun _ _ => rfl⟩
  | cons c rest ih =>
    obtain ⟨mid,he,stable,_,frame⟩ := grid_complete h distinct p c (valid c (by simp)) s
    obtain ⟨t,ht,stable',frame'⟩ := ih (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
    exact ⟨t,by simp only [trials,ReversedCumsumScalar.Host.raw,he,Option.bind_some] at *; exact ht,
      fun off hi => (stable' off hi).trans (stable off hi),fun addr hn => (frame' addr hn).trans (frame addr hn)⟩

theorem tuned_complete (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (tried : List Config) (valid : ∀ c∈tried,c∈configurations) (selected : Config)
    (chosen : selected∈configurations) (s : Heap) :
    ∃ t, trials h p (tried++[selected]) s=some t ∧ StableInput h p s t ∧
      (∀ row : Row h, ∀ time feature, time<h.T → feature<h.S →
        t (Storage.address p.backing.output (Geometry.address h row.val time feature))=expected h p s row.val time feature) ∧
      (∀ addr, ¬Storage.writes h p (Storage.programs h) addr → t addr=s addr) := by
  obtain ⟨mid,he,stable,frame⟩ := trials_complete h distinct p tried
    (fun c hc => (Kernels.configurations_positive c (valid c hc)).1) s
  obtain ⟨t,ht,stable',out,frame'⟩ := grid_complete h distinct p selected
    (Kernels.configurations_positive selected chosen).1 mid
  refine ⟨t,?_,fun off hi => (stable' off hi).trans (stable off hi),?_,fun addr hn => (frame' addr hn).trans (frame addr hn)⟩
  · rw [trials,ReversedCumsumScalar.Host.raw_append]
    change ((trials h p tried s).bind _) = some t
    rw [he]
    simp only [Option.bind_some,ReversedCumsumScalar.Host.raw,ht]
  · intro row time feature ht hs
    exact (out row time feature ht hs).trans (expected_stable h p s mid stable row time feature ht hs)

#print axioms tuned_complete
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealStream
