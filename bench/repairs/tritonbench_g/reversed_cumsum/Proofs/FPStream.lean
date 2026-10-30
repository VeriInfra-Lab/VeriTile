import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPStorage
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Host

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPStream
open VeriTile.Triton FP.Structural FPValues Geometry FPStorage
noncomputable section
set_option maxHeartbeats 4800000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def run (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h) (c : Config) : List (Program h) → Heap α → Option (Heap α)
  | [],s => some s
  | program::rest,s => (execute M dot h p c program s).bind (run M dot h p c rest)

def StableInput (h : Call) (p : Storage.Placement h) (a b : Heap α) : Prop :=
  ∀ off, off<p.backing.input.extent → b (Storage.address p.backing.input off)=a (Storage.address p.backing.input off)

theorem local_stable (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (a b : Heap α) (stable : StableInput h p a b)
    (row : Row h) (f t : Nat) (j : Fin BS) :
    localExpected M dot h p c positive b row.val f t j=localExpected M dot h p c positive a row.val f t j := by
  unfold localExpected
  apply congrArg (FPMemory.stored M h)
  apply FPReference.expected_congr
  intro time feature ht hs
  unfold input
  rw [stable _ (Storage.window h p row time feature ht hs).1]

theorem expected_stable (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (a b : Heap α) (stable : StableInput h p a b)
    (row : Row h) (t s : Nat) :
    expected M dot h p c positive b row.val t s=expected M dot h p c positive a row.val t s :=
  local_stable M dot h p c positive a b stable row (s/BS) t ⟨s%BS,Nat.mod_lt _ (by decide)⟩

theorem complete (M : Algebra α) (dot : DotBackend α) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (programs : List (Program h)) (s : Heap α) :
    ∃ t, run M dot h p c programs s=some t ∧ StableInput h p s t ∧
      (∀ program∈programs, ∀ time, ∀ j : Fin BS, time<h.T → program.2.val*BS+j.val<h.S →
        t (Storage.address p.backing.output (Geometry.address h program.1.val time (program.2.val*BS+j.val)))=
          localExpected M dot h p c positive s program.1.val program.2.val time j) ∧
      (∀ addr, ¬Storage.writes h p programs addr → t addr=s addr) := by
  induction programs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,by simp,fun _ _ => rfl⟩
  | cons program rest ih =>
    obtain ⟨mid,he,stable,out,frame⟩ := FPStorage.complete M dot h distinct p c positive program s
    obtain ⟨t,ht,stable',out',frame'⟩ := ih mid
    refine ⟨t,by simp only [run,he,Option.bind_some,ht],fun off hi => (stable' off hi).trans (stable off hi),?_,?_⟩
    · intro r member time j hv hs
      have eq := local_stable M dot h p c positive s mid stable r.1 r.2.val time j
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

def grid (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h) (c : Config) := run M dot h p c (Storage.programs h)

theorem grid_complete (M : Algebra α) (dot : DotBackend α) (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (c : Config) (positive : 0<c.BT) (s : Heap α) :
    ∃ t, grid M dot h p c s=some t ∧ StableInput h p s t ∧
      (∀ row : Row h, ∀ time feature, time<h.T → feature<h.S →
        t (Storage.address p.backing.output (Geometry.address h row.val time feature))=expected M dot h p c positive s row.val time feature) ∧
      (∀ addr, ¬Storage.writes h p (Storage.programs h) addr → t addr=s addr) := by
  obtain ⟨t,he,stable,out,frame⟩ := complete M dot h distinct p c positive (Storage.programs h) s
  refine ⟨t,he,stable,?_,frame⟩
  intro row time feature ht hs
  obtain ⟨f,j,eq⟩ := Storage.feature_decompose h feature hs
  have hv := out (row,f) (by simp [Storage.programs]) time j ht (by simpa [eq] using hs)
  rw [local_global] at hv
  simpa only [eq] using hv

def trials (models : Config → Algebra α) (dots : Config → DotBackend α) (h : Call) (p : Storage.Placement h) :=
  ReversedCumsumScalar.Host.raw (fun c => grid (models c) (dots c) h p c)

theorem trials_complete (models : Config → Algebra α) (dots : Config → DotBackend α)
    (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (configs : List Config) (valid : ∀ c∈configs,0<c.BT) (s : Heap α) :
    ∃ t, trials models dots h p configs s=some t ∧ StableInput h p s t ∧
      (∀ addr, ¬Storage.writes h p (Storage.programs h) addr → t addr=s addr) := by
  induction configs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ => rfl,fun _ _ => rfl⟩
  | cons c rest ih =>
    obtain ⟨mid,he,stable,_,frame⟩ := grid_complete (models c) (dots c) h distinct p c (valid c (by simp)) s
    obtain ⟨t,ht,stable',frame'⟩ := ih (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
    exact ⟨t,by simp only [trials,ReversedCumsumScalar.Host.raw,he,Option.bind_some] at *; exact ht,
      fun off hi => (stable' off hi).trans (stable off hi),fun addr hn => (frame' addr hn).trans (frame addr hn)⟩

theorem tuned_complete (models : Config → Algebra α) (dots : Config → DotBackend α)
    (h : Call) (distinct : h.input≠h.output) (p : Storage.Placement h)
    (tried : List Config) (valid : ∀ c∈tried,c∈configurations) (selected : Config)
    (chosen : selected∈configurations) (s : Heap α) :
    ∃ t, trials models dots h p (tried++[selected]) s=some t ∧ StableInput h p s t ∧
      (∀ row : Row h, ∀ time feature, time<h.T → feature<h.S →
        t (Storage.address p.backing.output (Geometry.address h row.val time feature))=
          expected (models selected) (dots selected) h p selected (Kernels.configurations_positive selected chosen).1 s row.val time feature) ∧
      (∀ addr, ¬Storage.writes h p (Storage.programs h) addr → t addr=s addr) := by
  obtain ⟨mid,he,stable,frame⟩ := trials_complete models dots h distinct p tried
    (fun c hc => (Kernels.configurations_positive c (valid c hc)).1) s
  obtain ⟨t,ht,stable',out,frame'⟩ := grid_complete (models selected) (dots selected) h distinct p selected
    (Kernels.configurations_positive selected chosen).1 mid
  refine ⟨t,?_,fun off hi => (stable' off hi).trans (stable off hi),?_,fun addr hn => (frame' addr hn).trans (frame addr hn)⟩
  · rw [trials,ReversedCumsumScalar.Host.raw_append]
    change ((trials models dots h p tried s).bind _) = some t
    rw [he]
    simp only [Option.bind_some,ReversedCumsumScalar.Host.raw,ht]
  · intro row time feature ht hs
    exact (out row time feature ht hs).trans (expected_stable (models selected) (dots selected) h p selected _ s mid stable row time feature)

#print axioms tuned_complete
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPStream
