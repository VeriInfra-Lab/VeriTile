import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Conformance

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Physical
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

abbrev Lane (c : Config) := TileIndex [block c.heads,block c.dims]
def lanes (c : Config) : List (Lane c) :=
  (TileShape.allIndices [block c.heads,block c.dims]).filter
    (fun i => Native.mask c i.1.val i.2.1.val)
def ptr (base : Nat) (offset : Word) : Nat :=
  (BitVec.ofInt 64 ((base:Int)+4*offset.toInt)).toNat
def sourceAddress (m : Inputs) (p : Nat) (i : Lane (config m)) : Nat :=
  ptr (base m .source) (Native.index (config m) p i.1.val i.2.1.val)
def outputAddress (m : Inputs) (p : Nat) (s : Memory) (i : Lane (config m)) : Nat :=
  ptr (base m .output)
    (s (addr (base m .destinations) p) * BitVec.ofNat 32 ((config m).heads*(config m).dims) +
      BitVec.ofNat 32 (config m).dims * BitVec.ofNat 32 i.1.val + BitVec.ofNat 32 i.2.1.val)

def stores {ι : Type} (key : ι → Nat) (value : ι → Word) (xs : List ι) (s : Memory) : Memory :=
  xs.foldl (fun t i => Function.update t (key i) (value i)) s
/-- One original source program: actual signed metadata, finite masks and
pointers, all payload loads buffered before the stores, and evolving physical memory. -/
def program (m : Inputs) (p : Nat) (s : Memory) : Memory :=
  stores (outputAddress m p s) (fun i => s (sourceAddress m p i)) (lanes (config m)) s
def launch (m : Inputs) (ps : List Nat) (s : Memory) : Memory :=
  ps.foldl (fun t p => program m p t) s

def Linked (m : Inputs) (s : State Word) (physical : Memory) : Prop :=
  ∀ x, s.mem outputRegion x=.mk .real (physical (addr (base m .output) x))
def Reads (m : Inputs) (w : Nat → Word) (initial current : Memory) : Prop :=
  (∀ x, x<(m .source).numel → current (addr (base m .source) x)=initial (addr (base m .source) x)) ∧
  (∀ p, p<(config m).count → current (addr (base m .destinations) p)=w p)

theorem lanes_exact (m : Inputs) (w : Nat → Word) (h : InputContract m w) :
    lanes (config m)=(TileShape.allIndices [block (config m).heads,block (config m).dims]).filter
      (fun i => decide (Tile.active (layout (config m)) i)) := by
  apply List.filter_congr
  intro i _
  exact Native.mask_exact m w h i.1.val i.2.1.val i.1.isLt i.2.1.isLt

theorem lane_active (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (i : Lane (config m)) (hi : i∈lanes (config m)) : Tile.active (layout (config m)) i := by
  rw [lanes_exact m w h] at hi
  exact of_decide_eq_true (List.mem_filter.mp hi).2

theorem ptr_exact (base size x : Nat) (cap : base+4*size≤2^64)
    (hx : x<size) (small : x<2^31) : ptr base (BitVec.ofNat 32 x)=addr base x := by
  rw [ptr,Native.signed32 x small]
  have eq : (base:Int)+4*(x:Int)=(addr base x:Int) := by unfold addr; omega
  rw [eq,BitVec.ofInt_natCast,BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by unfold addr; omega)]

theorem addresses (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (initial current : Memory) (binding : TensorBinding m w initial) (reads : Reads m w initial current)
    (p : Nat) (hp : p<(config m).count) (i : Lane (config m)) (hi : i∈lanes (config m)) :
    sourceAddress m p i=addr (base m .source) (flat (config m) p i.1.val i.2.1.val) ∧
    outputAddress m p current i=addr (base m .output) (flat (config m) (destination w p) i.1.val i.2.1.val) := by
  have active := lane_active m w h i hi
  have hs := source_bound m w h p _ _ hp active.1 active.2
  have ho := output_bound m w h p _ _ hp active.1 active.2
  have nonneg := (h.bounds p hp).1
  have small : destination w p<2^31 := by have upper := (w p).toInt_lt; unfold destination; omega
  have word : w p=BitVec.ofNat 32 (destination w p) := by
    apply BitVec.eq_of_toInt_eq
    rw [Native.signed32 _ small]
    exact (Int.toNat_of_nonneg nonneg).symm
  constructor
  · rw [sourceAddress,Native.index_word]
    exact ptr_exact _ _ _ (binding.capacity .source) hs (lt_trans hs (h.extent .source))
  · rw [outputAddress,reads.2 p hp,word]
    change ptr _ (Native.index (config m) (destination w p) _ _)=_
    rw [Native.index_word]
    exact ptr_exact _ _ _ (binding.capacity .output) ho (lt_trans ho (h.extent .output))

theorem linked_write (m : Inputs) (s : State Word) (physical : Memory)
    (link : Linked m s physical) (x : Nat) (value : Word) :
    Linked m (s.write outputRegion x (.mk .real value))
      (Function.update physical (addr (base m .output) x) value) := by
  intro q
  by_cases same : q=x
  · subst q; simp [State.write_same]
  · rw [State.write_other _ _ _ _ _ _ (Or.inr same),link,
      Function.update_of_ne (by unfold addr; omega)]

theorem linked_fold {ι : Type} (m : Inputs) (s : State Word) (physical : Memory)
    (link : Linked m s physical) (key : ι → Nat) (value : ι → Word) (xs : List ι) :
    Linked m (xs.foldl (fun t i => t.write outputRegion (key i) (.mk .real (value i))) s)
      (stores (fun i => addr (base m .output) (key i)) value xs physical) := by
  induction xs generalizing s physical with
  | nil => exact link
  | cons i tail ih => exact ih _ _ (linked_write m s physical link (key i) (value i))

theorem stores_congr {ι : Type} (k k' : ι → Nat) (v v' : ι → Word) (xs : List ι) (s : Memory)
    (keys : ∀ i∈xs, k i=k' i) (values : ∀ i∈xs, v i=v' i) : stores k v xs s=stores k' v' xs s := by
  induction xs generalizing s with
  | nil => rfl
  | cons i tail ih =>
    simp only [stores,List.foldl_cons,keys i (by simp),values i (by simp)]
    exact ih _ (fun j hj => keys j (by simp [hj])) (fun j hj => values j (by simp [hj]))

theorem program_linked (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (initial current : Memory) (binding : TensorBinding m w initial) (reads : Reads m w initial current)
    (s : State Word) (bound : Grid.Bound s (fun x => initial (addr (base m .source) x)) (destination w))
    (link : Linked m s current) (p : Nat) (hp : p<(config m).count) :
    Linked m (Grid.step rest (config m) s p) (program m p current) := by
  have effects : program m p current=
      stores (fun i : Lane (config m) => addr (base m .output)
        (flat (config m) (destination w p) i.1.val i.2.1.val))
        (fun i => initial (addr (base m .source) (flat (config m) p i.1.val i.2.1.val)))
        (lanes (config m)) current := by
    apply stores_congr
    · intro i hi; exact (addresses m w h initial current binding reads p hp i hi).2
    · intro i hi
      rw [(addresses m w h initial current binding reads p hp i hi).1]
      have ha := lane_active m w h i hi
      exact reads.1 _ (source_bound m w h p _ _ hp ha.1 ha.2)
  rw [effects,lanes_exact m w h]
  intro x
  rw [Grid.step,Tile.execution_memory]
  unfold Tile.copied
  rw [State.masked_scatter_eq_filter]
  simp_rw [Grid.offset (config m) s p _ (destination w) bound]
  simp only [Tile.loaded,Tile.inputOffset,layout,ite_true,Nat.one_mul,bound.input,Cell.read_same]
  have linked := linked_fold m {s with pids := fun d => if d=0 then p else s.pids d} current
    link (fun i : Lane (config m) => flat (config m) (destination w p) i.1.val i.2.1.val)
    (fun i => initial (addr (base m .source) (flat (config m) p i.1.val i.2.1.val)))
    ((TileShape.allIndices [block (config m).heads,block (config m).dims]).filter
      (fun i => decide (Tile.active (layout (config m)) i))) x
  simpa only [flat,Nat.mul_comm] using linked

theorem stores_frame {ι : Type} (key : ι → Nat) (value : ι → Word) (xs : List ι) (s : Memory)
    (cell : Nat) (miss : ∀ i∈xs, cell≠key i) : stores key value xs s cell=s cell := by
  induction xs generalizing s with
  | nil => rfl
  | cons i tail ih =>
    change stores key value tail (Function.update s (key i) (value i)) cell=s cell
    rw [ih _ (fun j hj => miss j (by simp [hj])),Function.update_of_ne (miss i (by simp))]

theorem program_frame (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (initial current : Memory) (binding : TensorBinding m w initial) (reads : Reads m w initial current)
    (p : Nat) (hp : p<(config m).count) (cell : Nat)
    (miss : ∀ a b, a<(config m).heads → b<(config m).dims →
      cell≠addr (base m .output) (flat (config m) (destination w p) a b)) :
    program m p current cell=current cell := by
  apply stores_frame
  intro i hi
  rw [(addresses m w h initial current binding reads p hp i hi).2]
  have active := lane_active m w h i hi
  exact miss _ _ active.1 active.2

theorem program_reads (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (initial current : Memory) (binding : TensorBinding m w initial) (reads : Reads m w initial current)
    (p : Nat) (hp : p<(config m).count) : Reads m w initial (program m p current) := by
  have unchanged (r : Role) (hr : r≠.output) (y : Nat) (hy : y<(m r).numel) :
      program m p current (addr (base m r) y)=current (addr (base m r) y) := by
    apply program_frame m w h initial current binding reads p hp
    intro a b ha hb same
    have bound := output_bound m w h p a b hp ha hb
    have separate := h.separate r hr
    unfold addr at same
    rcases separate with left|right <;> omega
  constructor
  · intro x hx; rw [unchanged .source (by decide) x hx]; exact reads.1 x hx
  · intro q hq; rw [unchanged .destinations (by decide) q hq]; exact reads.2 q hq

theorem grid_linked (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (initial current : Memory) (binding : TensorBinding m w initial) (reads : Reads m w initial current)
    (s : State Word) (bound : Grid.Bound s (fun x => initial (addr (base m .source) x)) (destination w))
    (link : Linked m s current) (ps : List Nat) (members : ∀ p∈ps, p<(config m).count) :
    Linked m (Grid.run rest (config m) ps s) (launch m ps current) := by
  induction ps generalizing s current with
  | nil => exact link
  | cons p tail ih =>
    exact ih _ (program_reads m w h initial current binding reads p (members p (by simp))) _
      (Grid.step_bound rest (config m) s p _ _ bound)
      (program_linked rest m w h initial current binding reads s bound link p (members p (by simp)))
      (fun q hq => members q (by simp [hq]))

theorem grid_frame (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (initial current : Memory) (binding : TensorBinding m w initial) (reads : Reads m w initial current)
    (ps : List Nat) (members : ∀ p∈ps, p<(config m).count) (cell : Nat)
    (miss : ¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) :
    launch m ps current cell=current cell := by
  induction ps generalizing current with
  | nil => rfl
  | cons p tail ih =>
    change launch m tail (program m p current) cell=current cell
    rw [ih _ (program_reads m w h initial current binding reads p (members p (by simp)))
      (fun q hq => members q (by simp [hq]))]
    apply program_frame m w h initial current binding reads p (members p (by simp))
    intro a b ha hb same
    exact miss ⟨_,⟨p,a,b,members p (by simp),ha,hb,rfl⟩,same⟩

/-- The Python pointer additions sign-extend their three int32 terms separately.
The admitted extent makes this exact staged chain equal to the combined native pointer. -/
def stagedPointer (base : Nat) (c : Config) (p a b : Nat) : Nat :=
  ((BitVec.ofNat 64 base + BitVec.ofInt 64
    (4*(BitVec.ofNat 32 p*BitVec.ofNat 32 (c.heads*c.dims)).toInt)) +
    BitVec.ofInt 64 (4*(BitVec.ofNat 32 c.dims*BitVec.ofNat 32 a).toInt) +
    BitVec.ofInt 64 (4*(BitVec.ofNat 32 b).toInt)).toNat

theorem staged_pointer (base : Nat) (c : Config) (p a b : Nat)
    (bound : flat c p a b<2^31) : stagedPointer base c p a b=ptr base (Native.index c p a b) := by
  have terms := pointer_terms c p a b bound
  simp only [stagedPointer,terms.1,terms.2.1,terms.2.2,ptr,Native.index_word,Native.signed32 _ bound]
  rw [←BitVec.ofInt_natCast]
  simp only [←BitVec.ofInt_add]
  congr 2
  unfold flat
  push_cast
  ring

#axiomsClean staged_pointer
#axiomsClean program_linked
#axiomsClean grid_linked
#axiomsClean grid_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Physical
