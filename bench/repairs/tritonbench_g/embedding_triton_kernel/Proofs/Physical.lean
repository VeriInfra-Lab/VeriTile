import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Bounds
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Grid

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Physical
open VeriTile.Triton FP.Structural Request
noncomputable section
set_option maxHeartbeats 4000000

abbrev Word := BitVec 32
abbrev Memory := Nat → Word
abbrev Event := Nat × Nat × Nat

def address (t : VeriTile.Torch.TensorMetadata) (i : Nat) : Nat := t.dataPtr.toNat+4*i

def row (e : Event) : Nat := e.1*64+e.2.1
def column (e : Event) : Nat := e.2.2

def active (c : Request) (e : Event) : Prop := row e<c.n ∧ column e<c.hidden
instance (c : Request) (e : Event) : Decidable (active c e) := inferInstanceAs (Decidable (_ ∧ _))

def idAt (c : Request) (m : Memory) (r : Nat) : Int := (m (address c.ids r)).toInt

def desired (c : Request) (m : Memory) (r d : Nat) : Word :=
  target 0 c.first c.last (idAt c m)
    (fun i j=>m (address c.weight (i*c.hidden+j))) r d

def outputAddress (c : Request) (e : Event) : Nat := address c.output (row e*c.hidden+column e)

/-- One source iteration/lane: masked row and column stores are no-ops.
This event reads the evolving memory, not a desired-output oracle. Native.lean
proves that the widened source expressions produce these exact addresses. -/
def step (c : Request) (m : Memory) (e : Event) : Memory :=
  if active c e then
    let id := idAt c m (row e)
    let value := if c.firstId ≤ id ∧ id<c.lastId then
      m (address c.weight ((id-c.firstId).toNat*c.hidden+column e)) else 0
    Function.update m (outputAddress c e) value
  else m

def events (c : Request) : List Event :=
  (List.range (grid c.n)).flatMap (fun pid=>(List.range 64).flatMap
    (fun a=>(List.range c.hidden.nextPowerOfTwo).map (fun d=>(pid,a,d))))

def launch (c : Request) (m : Memory) : Memory := (events c).foldl (step c) m

def Outside (c : Request) (byte : Nat) : Prop :=
  ∀ r,r<c.n → ∀ d,d<c.hidden → byte≠address c.output (r*c.hidden+d)

theorem id_bounds (c : Request) (m : Memory) (r : Nat) :
    -(2^31) ≤ idAt c m r ∧ idAt c m r<2^31 := by
  exact ⟨BitVec.le_toInt _,BitVec.toInt_lt⟩

theorem normalized_value (c : Request) (m : Memory) (r d : Nat) :
    (if c.firstId ≤ idAt c m r ∧ idAt c m r<c.lastId then
      m (address c.weight ((idAt c m r-c.firstId).toNat*c.hidden+d)) else 0)=
      desired c m r d := by
  obtain ⟨lo,hi⟩ := id_bounds c m r
  simp only [Bounds.interval c _ lo hi]
  unfold desired target
  split_ifs with h
  · rw [Bounds.first_eq c _ lo hi h]
  · rfl

theorem separated_address (a b : VeriTile.Torch.TensorMetadata) (h : Request.disjoint a b)
    (i j : Nat) (hi : i<a.numel) (hj : j<b.numel) : address a i≠address b j := by
  unfold address
  rcases h with h|h|h|h <;> omega

theorem input_frame (c : Request) (h : InputContract c) (m : Memory) (e : Event) :
    (∀ r,r<c.n → step c m e (address c.ids r)=m (address c.ids r)) ∧
    (∀ r,r<c.rows → ∀ d,d<c.hidden →
      step c m e (address c.weight (r*c.hidden+d))=m (address c.weight (r*c.hidden+d))) := by
  obtain ⟨hn,hw,ho⟩ := Bounds.sizes c h
  have sep := h.2.2.2.2.2.2.1
  by_cases ha : active c e
  · have bound : row e*c.hidden+column e<c.output.numel := by
      rw [ho]; exact Bounds.flat_bound _ _ _ _ ha.1 ha.2
    constructor
    · intro r hr
      have ne := separated_address c.output c.ids sep.1 _ r bound (by omega)
      simp [step,ha,outputAddress,Function.update_of_ne ne.symm]
    · intro r hr d hd
      have ne := separated_address c.output c.weight sep.2 _ (r*c.hidden+d) bound
        (by rw [hw]; exact Bounds.flat_bound _ _ _ _ hr hd)
      simp [step,ha,outputAddress,Function.update_of_ne ne.symm]
  · simp [step,ha]

theorem desired_stable (c : Request) (h : InputContract c) (m u : Memory)
    (ids : ∀ r,r<c.n → u (address c.ids r)=m (address c.ids r))
    (weight : ∀ r,r<c.rows → ∀ d,d<c.hidden →
      u (address c.weight (r*c.hidden+d))=m (address c.weight (r*c.hidden+d)))
    (r d : Nat) (hr : r<c.n) (hd : d<c.hidden) : desired c u r d=desired c m r d := by
  have hid : idAt c u r=idAt c m r := by simp [idAt,ids r hr]
  simp only [desired,target,hid]
  split_ifs with hv
  · obtain ⟨lo,hi⟩ := id_bounds c m r
    exact weight _ (Bounds.selected_row c h (by omega) _ lo hi hv) d hd
  · rfl

theorem injective (c : Request) (r s d e : Nat) (hd : d<c.hidden) (he : e<c.hidden)
    (eq : address c.output (r*c.hidden+d)=address c.output (s*c.hidden+e)) : r=s ∧ d=e := by
  have off : r*c.hidden+d=s*c.hidden+e := by unfold address at eq; omega
  have hs : 0<c.hidden := by omega
  have div := congrArg (fun n=>n/c.hidden) off
  have rows : r=s := by
    simpa [Nat.mul_comm r c.hidden,Nat.mul_comm s c.hidden,Nat.mul_add_div hs,
      Nat.div_eq_of_lt hd,Nat.div_eq_of_lt he] using div
  exact ⟨rows,by rw [rows] at off; omega⟩

structure Invariant (c : Request) (initial current : Memory) (seen : List Event) : Prop where
  ids : ∀ r,r<c.n → current (address c.ids r)=initial (address c.ids r)
  weight : ∀ r,r<c.rows → ∀ d,d<c.hidden →
    current (address c.weight (r*c.hidden+d))=initial (address c.weight (r*c.hidden+d))
  output : ∀ e,e∈seen → active c e →
    current (outputAddress c e)=desired c initial (row e) (column e)
  frame : ∀ byte,Outside c byte → current byte=initial byte

theorem invariant_step (c : Request) (h : InputContract c) (initial m : Memory)
    (seen : List Event) (inv : Invariant c initial m seen) (e : Event) :
    Invariant c initial (step c m e) (e::seen) := by
  obtain ⟨ids,weight⟩ := input_frame c h m e
  refine ⟨fun r hr=>(ids r hr).trans (inv.ids r hr),
    fun r hr d hd=>(weight r hr d hd).trans (inv.weight r hr d hd),?_,?_⟩
  · intro f hf ha
    by_cases he : active c e
    · have val := desired_stable c h initial m inv.ids inv.weight (row e) (column e) he.1 he.2
      have update : step c m e=Function.update m (outputAddress c e)
          (desired c initial (row e) (column e)) := by
        simp only [step,if_pos he,normalized_value,val]
      rw [update]
      by_cases eq : outputAddress c f=outputAddress c e
      · rw [eq,Function.update_self]
        have same := injective c (row f) (row e) (column f) (column e) ha.2 he.2 eq
        rw [same.1,same.2]
      · rw [Function.update_of_ne eq]
        have mem : f∈seen := by
          rcases List.mem_cons.mp hf with same|mem
          · subst f; exact False.elim (eq rfl)
          · exact mem
        exact inv.output f mem ha
    · have mem : f∈seen := by
        rcases List.mem_cons.mp hf with same|mem
        · subst f; contradiction
        · exact mem
      simpa [step,he] using inv.output f mem ha
  · intro byte miss
    by_cases he : active c e
    · have ne : byte≠outputAddress c e := miss _ he.1 _ he.2
      simpa [step,he,Function.update_of_ne ne] using inv.frame byte miss
    · simpa [step,he] using inv.frame byte miss

theorem fold_invariant (c : Request) (h : InputContract c) (initial m : Memory)
    (todo seen : List Event) (inv : Invariant c initial m seen) :
    Invariant c initial (todo.foldl (step c) m) (todo.reverse++seen) := by
  induction todo generalizing m seen with
  | nil => simpa using inv
  | cons e tail ih =>
    simpa [List.foldl_cons,List.reverse_cons,List.append_assoc] using
      ih (step c m e) (e::seen) (invariant_step c h initial m seen inv e)

theorem coverage (c : Request) (r d : Nat) (hr : r<c.n) (hd : d<c.hidden) :
    (r/64,r%64,d)∈events c := by
  have hp := Grid.row_covered c.n r hr
  have ha := Nat.mod_lt r (by decide : 0<64)
  have hwidth := lt_of_lt_of_le hd (Bounds.nextPower_covers c.hidden)
  simp only [events,List.mem_flatMap,List.mem_range,List.mem_map]
  exact ⟨r/64,hp,r%64,ha,d,hwidth,rfl⟩

/-- Full source event order, all 64 loop iterations and padded feature lanes.
The output formula uses the entry-state inputs; every other word is framed. -/
theorem launch_correct (c : Request) (h : InputContract c) (m : Memory) :
    (∀ r,r<c.n → ∀ d,d<c.hidden →
      launch c m (address c.output (r*c.hidden+d))=desired c m r d) ∧
    (∀ byte,Outside c byte → launch c m byte=m byte) := by
  have base : Invariant c m m [] := ⟨by intros; rfl,by intros; rfl,by simp,by intros; rfl⟩
  have inv := fold_invariant c h m m (events c) [] base
  refine ⟨?_,inv.frame⟩
  intro r hr d hd
  have e : row (r/64,r%64,d)=r := by simp only [row]; omega
  have read := inv.output (r/64,r%64,d)
    (by simpa using coverage c r d hr hd) (by simpa [active,e,column] using And.intro hr hd)
  simpa [launch,outputAddress,e,column] using read

/-- Any complete source-event order gives the same public output. The input
regions are immutable and repeated stores to one coordinate carry one value. -/
theorem complete_order (c : Request) (h : InputContract c) (m : Memory) (order : List Event)
    (complete : ∀ e,e∈events c → active c e → e∈order) :
    (∀ r,r<c.n → ∀ d,d<c.hidden →
      order.foldl (step c) m (address c.output (r*c.hidden+d))=desired c m r d) ∧
    (∀ byte,Outside c byte → order.foldl (step c) m byte=m byte) := by
  have base : Invariant c m m [] := ⟨by intros; rfl,by intros; rfl,by simp,by intros; rfl⟩
  have inv := fold_invariant c h m m order [] base
  refine ⟨?_,inv.frame⟩
  intro r hr d hd
  have e : row (r/64,r%64,d)=r := by simp only [row]; omega
  have active : active c (r/64,r%64,d) := by
    simpa [active,e,column] using And.intro hr hd
  have read := inv.output (r/64,r%64,d)
    (by simpa using complete _ (coverage c r d hr hd) active) active
  simpa [outputAddress,e,column] using read

#axiomsClean launch_correct
#axiomsClean complete_order
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Physical
