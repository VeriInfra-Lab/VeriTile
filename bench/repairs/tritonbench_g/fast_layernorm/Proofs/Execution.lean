import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Storage

/-! Concrete FP32 stores and complete byte frames. Values are supplied by the
three source graphs below. No successful-kernel or final-output equation is an
external API contract. Prefixes model completed writes on exceptional paths. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage
noncomputable section
open scoped Classical

structure Region (α : Type) where
  buffer : Dense
  value : Fin buffer.count → α

structure Trace (α : Type) (K : Nat) where
  owner : Nat
  regions : Fin K → Region α
  owned : ∀ k,(regions k).buffer.owner=owner
  separated : ∀ i j,i≠j → Storage.Disjoint (regions i).buffer.tensor.backing (regions j).buffer.tensor.backing

def Region.writes (r : Region α) : List (Memory.Write α) :=
  (List.finRange r.buffer.count).map fun i => ⟨r.buffer.address i.val,r.value i⟩

def Trace.writes (t : Trace α K) : List (Memory.Write α) :=
  (List.finRange K).flatMap fun k => (t.regions k).writes

theorem trace_mem (t : Trace α K) (e : Memory.Write α) :
    e∈t.writes ↔ ∃ k i,e=⟨(t.regions k).buffer.address i.val,(t.regions k).value i⟩ := by
  simp [Trace.writes,Region.writes,List.mem_flatMap,List.mem_map,eq_comm]

theorem buffer_apart (a b : Dense) (h : Storage.Disjoint a.tensor.backing b.tensor.backing)
    (owner : a.owner=b.owner) (i j : Nat) (hi : i<a.count) (hj : j<b.count) :
    Memory.Apart (a.address i) (b.address j) := by
  have sa := a.span
  have sb := b.span
  have la := a.lower (by omega)
  have lb := b.lower (by omega)
  unfold Storage.Disjoint at h
  unfold Dense.owner at owner
  unfold Memory.Apart Dense.address
  rcases h with h|h|h <;> omega

theorem trace_readback (api : Memory.API α Store Byte) (t : Trace α K)
    (m : World Store) (k : Fin K) (i : Fin (t.regions k).buffer.count) :
    api.read (runWorld api t.owner t.writes m t.owner) ((t.regions k).buffer.address i.val)=
      (t.regions k).value i := by
  apply runWorld_readback
  · refine ⟨⟨_,_⟩,(trace_mem t _).mpr ⟨k,i,rfl⟩,rfl,rfl⟩
  · intro e he
    obtain ⟨j,l,rfl⟩ := (trace_mem t e).mp he
    by_cases h : j=k
    · subst j
      by_cases hl : l=i
      · subst l
        exact Or.inl ⟨rfl,rfl⟩
      · exact Or.inr (dense_apart _ _ _ (by exact fun hv => hl (Fin.ext hv)))
    · exact Or.inr (buffer_apart _ _ (t.separated j k h)
        ((t.owned j).trans (t.owned k).symm) _ _ l.isLt i.isLt)

theorem trace_prefix_frame (api : Memory.API α Store Byte) (t : Trace α K)
    (m : World Store) (callerTensors : List Tensor)
    (fresh : ∀ k old,old∈callerTensors → Storage.Disjoint (t.regions k).buffer.tensor.backing old.backing)
    (completed : List (Memory.Write α)) (sub : ∀ e∈completed,e∈t.writes)
    (key : Key) (hk : caller callerTensors key) :
    observe api (runWorld api t.owner completed m) key=observe api m key := by
  apply runWorld_frame
  intro ho e he
  obtain ⟨k,i,rfl⟩ := (trace_mem t e).mp (sub e he)
  obtain ⟨old,hold,hkey⟩ := hk
  exact disjoint_byte _ _ (fresh k old hold) _ i.isLt key hkey (ho.trans (t.owned k).symm)

/-- All input words remain readable after every completed write completed.
This justifies evaluating independent program loads from the launch snapshot. -/
theorem trace_prefix_input (api : Memory.API α Store Byte) (t : Trace α K)
    (m : World Store) (input : Dense) (same : input.owner=t.owner)
    (fresh : ∀ k,Storage.Disjoint (t.regions k).buffer.tensor.backing input.tensor.backing)
    (completed : List (Memory.Write α)) (sub : ∀ e∈completed,e∈t.writes)
    (i : Fin input.count) :
    api.read (runWorld api t.owner completed m input.owner) (input.address i.val)=
      api.read (m input.owner) (input.address i.val) := by
  rw [same]
  simp only [runWorld,Function.update_self]
  apply Memory.run_read_frame
  intro e he
  obtain ⟨k,j,rfl⟩ := (trace_mem t e).mp (sub e he)
  exact buffer_apart _ _ (fresh k) ((t.owned k).trans same.symm) _ _ j.isLt i.isLt

def rows (x : Dense) (N : Nat) (m : World Store) (api : Memory.API α Store Byte) : Nat → Nat → α :=
  fun row col => api.read (m x.owner) (x.address (row*N+col))
def vector (x : Dense) (m : World Store) (api : Memory.API α Store Byte) : Nat → α :=
  fun col => api.read (m x.owner) (x.address col)

/-- Forward stores saved inverse, saved mean and actual Y. Native scalar
statistics for the save instruction are those of lane zero. -/
def forwardRegions (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (x w bias y r mu : Dense) (eps : α) (m : World Store) : Fin 3 → Region α
  | ⟨0,_⟩ => ⟨r,fun i => (statistics ops plan (rows x N m api i.val) eps 0).inverse⟩
  | ⟨1,_⟩ => ⟨mu,fun i => (statistics ops plan (rows x N m api i.val) eps 0).mean⟩
  | ⟨2,_⟩ => ⟨y,fun i => forwardReference ops plan (rows x N m api (i.val/N))
      (vector w m api) (vector bias m api) eps (i.val%N)⟩

def forwardRegionsNoBias (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (x w y r mu : Dense) (eps : α) (m : World Store) : Fin 3 → Region α
  | ⟨0,_⟩ => ⟨r,fun i => (statistics ops plan (rows x N m api i.val) eps 0).inverse⟩
  | ⟨1,_⟩ => ⟨mu,fun i => (statistics ops plan (rows x N m api i.val) eps 0).mean⟩
  | ⟨2,_⟩ => ⟨y,fun i => forwardReference ops plan (rows x N m api (i.val/N))
      (vector w m api) (fun _ => ops.zero) eps (i.val%N)⟩

def saved (api : Memory.API α Store Byte) (r mu : Dense) (m : World Store) (row : Nat) : Statistics α :=
  ⟨vector mu m api row,vector r m api row⟩

def backwardRegion (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (x w dy r mu dx : Dense) (m : World Store) : Region α :=
  ⟨dx,fun i => backwardReference ops plan (saved api r mu m (i.val/N))
    (rows x N m api (i.val/N)) (rows dy N m api (i.val/N)) (vector w m api) (i.val%N)⟩

def affineRegions (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (rowCount : Nat) (x dy r mu dw db : Dense) (m : World Store) : Fin 2 → Region α
  | ⟨0,_⟩ => ⟨dw,fun i => (affineReference ops plan rowCount (rows x N m api)
      (rows dy N m api) (saved api r mu m) i.val).1⟩
  | ⟨1,_⟩ => ⟨db,fun i => (affineReference ops plan rowCount (rows x N m api)
      (rows dy N m api) (saved api r mu m) i.val).2⟩

/-- A runtime receipt lists actual completed stores. It supplies no values.
Successful coverage is the ordinary scheduler contract; every value is still
computed by the selected source graph. Failed receipts need no coverage. -/
structure Schedule (K : Nat) where
  (objectIdentity owner stream : Nat)
  grid : List Nat
  pointers : List Key
  order : List (Fin K × Nat)
  status : Except Nat Unit

def scheduledWrites (t : Trace α K) (schedule : Schedule K) : List (Memory.Write α) :=
  schedule.order.filterMap fun entry =>
    if h : entry.2<(t.regions entry.1).buffer.count then
      some ⟨(t.regions entry.1).buffer.address entry.2,(t.regions entry.1).value ⟨entry.2,h⟩⟩
    else none

theorem scheduled_subset (t : Trace α K) (schedule : Schedule K) (event : Memory.Write α)
    (he : event∈scheduledWrites t schedule) : event∈t.writes := by
  obtain ⟨entry,_,h⟩ := List.mem_filterMap.mp he
  split at h
  · rename_i hi
    have heq : event=⟨(t.regions entry.1).buffer.address entry.2,(t.regions entry.1).value ⟨entry.2,hi⟩⟩ := by
      simpa using h.symm
    exact (trace_mem t event).mpr ⟨entry.1,⟨entry.2,hi⟩,heq⟩
  · contradiction

theorem scheduled_readback (api : Memory.API α Store Byte) (t : Trace α K) (schedule : Schedule K)
    (covered : ∀ k i,i<(t.regions k).buffer.count → (k,i)∈schedule.order)
    (m : Storage.World Store) (k : Fin K) (i : Fin (t.regions k).buffer.count) :
    api.read (runWorld api t.owner (scheduledWrites t schedule) m t.owner)
      ((t.regions k).buffer.address i.val)=(t.regions k).value i := by
  apply runWorld_readback
  · refine ⟨⟨_,_⟩,List.mem_filterMap.mpr ⟨(k,i.val),covered k i.val i.isLt,?_⟩,rfl,rfl⟩
    simp [i.isLt,Dense.address]
  · intro event he
    obtain ⟨j,l,rfl⟩ := (trace_mem t event).mp (scheduled_subset t schedule event he)
    by_cases h : j=k
    · subst j
      by_cases hl : l=i
      · subst l
        exact Or.inl ⟨rfl,rfl⟩
      · exact Or.inr (dense_apart _ _ _ (by exact fun hv => hl (Fin.ext hv)))
    · exact Or.inr (buffer_apart _ _ (t.separated j k h)
        ((t.owned j).trans (t.owned k).symm) _ _ l.isLt i.isLt)

/-- Grid-product and segmentation admissibility are explicitly left to the
runtime boundary. These are kernel integer proofs, derived from allocations. -/
theorem row_offset_safe (x : Dense) (rowCount N row col : Nat)
    (shape : x.count=rowCount*N) (hr : row<rowCount) (hc : col<N) :
    row*N+col<x.count ∧ row*N+col<2^63 := by
  have h := dense_count_bound x
  have hm : (row+1)*N≤rowCount*N := Nat.mul_le_mul_right N (by omega)
  constructor <;> nlinarith

theorem masked_affine_offset_safe (x : Dense) (rowCount N row col : Nat)
    (shape : x.count=rowCount*N) (hn : 0<N) (hr : row<rowCount) :
    row*N+(if col<N then col else 0)<x.count ∧
    row*N+(if col<N then col else 0)<2^63 := by
  apply row_offset_safe x rowCount N row _ shape hr
  split <;> omega

theorem loop_counter_safe (x : Dense) (rowCount N : Nat) (shape : x.count=rowCount*N)
    (hn : 0<N) (row : Nat) (hr : row≤rowCount) : row<2^63 := by
  have h := dense_count_bound x
  have hm : rowCount≤rowCount*N := Nat.le_mul_of_pos_right _ hn
  norm_num at *
  omega

#axiomsClean scheduled_subset
#axiomsClean scheduled_readback
#axiomsClean trace_readback
#axiomsClean trace_prefix_frame
#axiomsClean trace_prefix_input
#axiomsClean row_offset_safe
#axiomsClean masked_affine_offset_safe
#axiomsClean loop_counter_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Execution
