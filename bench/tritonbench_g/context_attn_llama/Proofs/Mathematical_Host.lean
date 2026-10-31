import bench.tritonbench_g.context_attn_llama.Sources.Mathematical_HostInterface
import bench.tritonbench_g.context_attn_llama.Proofs.Mathematical_RealIO
import bench.tritonbench_g.context_attn_llama.Proofs.Mathematical_HostComposition

namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.Host
open VeriTile.Triton Formula Program
namespace Composition
export VeriTile.Bench.TritonBenchG.ContextAttnLlama.HostComposition (finite_grid input_stable)
end Composition
noncomputable section
set_option maxHeartbeats 1000000

def Call.active (c : Call) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) : Prop :=
  queryRow c.layout (c.launchAt idx) i.1.val < c.length (batch c.layout (c.launchAt idx))
def Call.offset (c : Call) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) : Nat :=
  c.output (batch c.layout (c.launchAt idx)) (head c.layout (c.launchAt idx))
    (queryRow c.layout (c.launchAt idx) i.1.val) i.2.1.val

/-- Ordinary tensor interface, expressed directly in its actual strides and
finite allocations. These are metadata, bounds and nonaliasing conditions,
not conclusions about execution or values. Underlaunch is permitted: coverage
of every logical row is a separate consequence of `length ≤ maxInput`. -/
structure Valid (c : Call) (A : FlatAlloc) (s : BlockState) : Prop where
  start_value : ∀ b, b < c.batches → (inputs s c.regions).start b = (c.start b : Int)
  length_value : ∀ b, b < c.batches → (inputs s c.regions).sequence b = ((c.length b+c.prompt b : Nat) : Int)
  prompt_value : ∀ b, b < c.batches → (inputs s c.regions).prompt b = (c.prompt b : Int)
  request_value : ∀ b, b < c.batches → (inputs s c.regions).request b = (c.request b : Int)
  page_value : ∀ b, b < c.batches → ∀ j, j < c.length b+c.prompt b →
    (inputs s c.regions).pages (c.layout.reqRow*c.request b+c.layout.reqToken*j) = (c.page b j : Int)
  prompt_bound : c.batches ≤ A.extent c.regions.prompt
  request_bound : c.batches ≤ A.extent c.regions.request
  page_bound : ∀ b, b < c.batches → ∀ j, j < c.length b+c.prompt b →
    c.layout.reqRow*c.request b+c.layout.reqToken*j < A.extent c.regions.pages
  start_bound : c.batches ≤ A.extent c.regions.start
  length_bound : c.batches ≤ A.extent c.regions.sequence
  group_pos : 0 < c.layout.kvGroup
  grouped : c.layout.heads = c.kvHeads*c.layout.kvGroup
  query : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.qRow+h*c.layout.qHead+d*c.layout.qDim < A.extent c.regions.query
  key : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b+c.prompt b →
    ∀ d, d < c.layout.D →
      c.page b row*c.layout.kRow+(h/c.layout.kvGroup)*c.layout.kHead+d*c.layout.kDim < A.extent c.regions.key
  value : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b+c.prompt b →
    ∀ d, d < c.layout.D →
      c.page b row*c.layout.vRow+(h/c.layout.kvGroup)*c.layout.vHead+d*c.layout.vDim < A.extent c.regions.value
  out : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D → c.output b h row d < A.extent c.regions.out
  packed : ∀ b b' row row', b < c.batches → b' < c.batches →
    row < c.length b → row' < c.length b' →
    c.start b+row = c.start b'+row' → b=b' ∧ row=row'
  output_view : ∀ b b' h h' row row' d d', b < c.batches → b' < c.batches →
    h < c.layout.heads → h' < c.layout.heads → row < c.length b → row' < c.length b' →
    d < c.layout.D → d' < c.layout.D →
    c.output b h row d = c.output b' h' row' d' →
    c.start b+row = c.start b'+row' ∧ h=h' ∧ d=d'
  input_separate : ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,
    c.regions.start,c.regions.sequence,c.regions.prompt,c.regions.request,c.regions.pages] → c.regions.out ≠ rg

open Call

theorem bm_positive (c : Call) : 0 < c.layout.BM := by
  rcases c.blockM with h | h <;> omega

theorem batch_bound (c : Call) (idx : GridIndex c.grid) :
    batch c.layout (c.launchAt idx) < c.batches := by
  have h := (idx ⟨1,by simp [grid,Grid.rank]⟩).isLt
  exact (Nat.div_lt_iff_lt_mul c.headsPositive).mpr h
theorem head_bound (c : Call) (idx : GridIndex c.grid) :
    head c.layout (c.launchAt idx) < c.layout.heads :=
  Nat.mod_lt _ c.headsPositive
theorem block_bound (c : Call) (idx : GridIndex c.grid) :
    (c.launchAt idx).block < c.blocks :=
  (idx ⟨0,by simp [grid,Grid.rank]⟩).isLt

theorem launchAt_injective (c : Call) {i j : GridIndex c.grid}
    (hb : batch c.layout (c.launchAt i) = batch c.layout (c.launchAt j))
    (hh : head c.layout (c.launchAt i) = head c.layout (c.launchAt j))
    (hm : (c.launchAt i).block = (c.launchAt j).block) : i=j := by
  have hbh : (c.launchAt i).batchHead = (c.launchAt j).batchHead := by
    have hi := Nat.div_add_mod (c.launchAt i).batchHead c.layout.heads
    have hj := Nat.div_add_mod (c.launchAt j).batchHead c.layout.heads
    change (c.launchAt i).batchHead / c.layout.heads = (c.launchAt j).batchHead / c.layout.heads at hb
    change (c.launchAt i).batchHead % c.layout.heads = (c.launchAt j).batchHead % c.layout.heads at hh
    rw [hb,hh] at hi
    omega
  funext ax
  apply Fin.ext
  have ha : ax.val < 3 := ax.isLt
  rcases Nat.lt_trichotomy ax.val 1 with h0 | h1 | h2
  · have he : ax = ⟨0,by simp [grid,Grid.rank]⟩ := Fin.ext (by change ax.val=0; omega)
    subst ax
    exact hm
  · have he : ax = ⟨1,by simp [grid,Grid.rank]⟩ := Fin.ext h1
    subst ax
    exact hbh
  · have he : ax = ⟨2,by simp [grid,Grid.rank]⟩ := Fin.ext (by change ax.val=2; omega)
    subst ax
    have hi : (i ⟨2,by simp [grid,Grid.rank]⟩).val < 1 := (i _).isLt
    have hj : (j ⟨2,by simp [grid,Grid.rank]⟩).val < 1 := (j _).isLt
    omega

theorem launch_grid (c : Call) (s : BlockState) (idx : GridIndex c.grid) :
    launch (s.withGridIndex idx) = c.launchAt idx := rfl

theorem inputs_grid (c : Call) (s : BlockState) (idx : GridIndex c.grid) :
    inputs (s.withGridIndex idx) c.regions = inputs s c.regions := rfl

theorem sequence_eq (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) :
    Formula.sequence c.layout (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions) = (c.length (batch c.layout (c.launchAt idx)) : Int) := by
  change (inputs s c.regions).sequence (batch c.layout (c.launchAt idx)) -
    (inputs s c.regions).prompt (batch c.layout (c.launchAt idx)) = _
  rw [v.length_value _ (batch_bound c idx),v.prompt_value _ (batch_bound c idx)]
  omega

theorem active_eq (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) :
    RealIO.active c.layout (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions) i ↔ c.active idx i := by
  unfold RealIO.active
  rw [sequence_eq c A s v idx]
  exact Int.ofNat_lt

theorem offset_eq (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) :
    RealIO.write c.layout (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions) i = c.offset idx i := by
  change (((inputs s c.regions).start (batch c.layout (c.launchAt idx)) +
    queryRow c.layout (c.launchAt idx) i.1.val) * c.layout.outRow +
    (head c.layout (c.launchAt idx))*c.layout.outHead+i.2.1.val*c.layout.outDim).toNat = _
  rw [v.start_value _ (batch_bound c idx)]
  dsimp only [offset,output]
  norm_cast

theorem writer_unique (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx jdx : GridIndex c.grid) (i j : RealIO.Lane c.layout)
    (hi : c.active idx i) (hj : c.active jdx j) (he : c.offset idx i = c.offset jdx j) :
    idx=jdx ∧ i=j := by
  obtain ⟨hrow,hh,hd⟩ := v.output_view _ _ _ _ _ _ _ _
    (batch_bound c idx) (batch_bound c jdx) (head_bound c idx) (head_bound c jdx)
    hi hj i.2.1.isLt j.2.1.isLt he
  obtain ⟨hb,hr⟩ := v.packed _ _ _ _ (batch_bound c idx) (batch_bound c jdx) hi hj hrow
  have him := i.1.isLt
  have hjm := j.1.isLt
  have hblock : (c.launchAt idx).block = (c.launchAt jdx).block := by
    rcases c.blockM with hm | hm <;> simp only [queryRow,hm] at hr him hjm <;> omega
  have hidx := launchAt_injective c hb hh hblock
  refine ⟨hidx,?_⟩
  have hilane : i.1=j.1 := by
    apply Fin.ext
    simp only [queryRow,hblock] at hr
    omega
  rcases i with ⟨ir,id,iu⟩
  rcases j with ⟨jr,jd,ju⟩
  have hid : id=jd := Fin.ext hd
  simp only [Prod.mk.injEq] at *
  cases hilane
  cases hid
  simp

theorem limit_bound (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) (j : Nat)
    (hj : (j : Int) < Formula.limit c.layout (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions)) :
    j < c.length (batch c.layout (c.launchAt idx))+c.prompt (batch c.layout (c.launchAt idx)) := by
  have h := lt_of_lt_of_le hj (min_le_right _ _)
  change (j : Int) < (inputs s c.regions).sequence (batch c.layout (c.launchAt idx)) at h
  rw [v.length_value _ (batch_bound c idx)] at h
  exact Int.ofNat_lt.mp h

theorem page_eq (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) (j : Nat)
    (hj : (j : Int) < Formula.limit c.layout (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions)) :
    Formula.page c.layout (launch (s.withGridIndex idx)) (inputs (s.withGridIndex idx) c.regions) j =
      (c.page (batch c.layout (c.launchAt idx)) j : Int) := by
  rw [Formula.page,if_pos hj]
  simp only [launch_grid,inputs_grid]
  rw [v.request_value _ (batch_bound c idx)]
  have ht : ((c.layout.reqRow : Int)*c.request (batch c.layout (c.launchAt idx))+
    c.layout.reqToken*j).toNat = c.layout.reqRow*c.request (batch c.layout (c.launchAt idx))+c.layout.reqToken*j := by
    norm_cast
  rw [ht]
  exact v.page_value _ (batch_bound c idx) j (limit_bound c A s v idx j hj)

theorem program_bounds (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) :
    AddressBounds.Valid c.layout c.regions (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions) A.extent := by
  have hb := batch_bound c idx
  have hh := head_bound c idx
  have hs := v.start_value _ hb
  constructor
  · exact Nat.lt_of_lt_of_le hb v.start_bound
  · exact Nat.lt_of_lt_of_le hb v.prompt_bound
  · exact Nat.lt_of_lt_of_le hb v.length_bound
  · exact Nat.lt_of_lt_of_le hb v.request_bound
  · intro n _ j hj
    have ht := v.page_bound _ hb _ (limit_bound c A s v idx _ hj)
    unfold AddressBounds.inside AddressBounds.pageAddress
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ c.layout.reqRow*(inputs s c.regions).request _+_ ∧
      c.layout.reqRow*(inputs s c.regions).request _+_ < _
    simp only [launch_grid]
    rw [v.request_value _ hb]
    constructor
    · positivity
    · exact_mod_cast ht
  · intro i d hi
    rw [sequence_eq c A s v idx] at hi
    have ht := v.query _ hb _ hh _ (Int.ofNat_lt.mp hi) _ d.isLt
    unfold AddressBounds.inside AddressBounds.queryAddress
    change 0 ≤ ((inputs s c.regions).start _+_)*_+_+_ ∧
      ((inputs s c.regions).start _+_)*_+_+_ < _
    simp only [launch_grid]
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast ht
  · intro n _ j d hj
    have ht := v.key _ hb _ hh _ (limit_bound c A s v idx _ hj) _ d.isLt
    unfold AddressBounds.inside AddressBounds.keyAddress
    rw [page_eq c A s v idx _ hj]
    constructor
    · positivity
    · exact_mod_cast ht
  · intro n _ j d hj
    have ht := v.value _ hb _ hh _ (limit_bound c A s v idx _ hj) _ d.isLt
    unfold AddressBounds.inside AddressBounds.valueAddress
    rw [page_eq c A s v idx _ hj]
    constructor
    · positivity
    · exact_mod_cast ht
  · intro i d hi
    rw [sequence_eq c A s v idx] at hi
    have ht := v.out _ hb _ hh _ (Int.ofNat_lt.mp hi) _ d.isLt
    unfold AddressBounds.inside AddressBounds.outputAddress
    change 0 ≤ ((inputs s c.regions).start _+_)*_+_+_ ∧
      ((inputs s c.regions).start _+_)*_+_+_ < _
    simp only [launch_grid]
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast ht

def Call.writes (c : Call) (A : FlatAlloc) : WriteFootprint :=
  VeriTile.Bench.TritonBenchG.ContextAttnLlama.HostComposition.footprint
    c.grid A c.active c.regions.out c.offset

/-- The Python wrapper updates the caller's `o` and returns no tensor. -/
def returned : Option Unit := none
theorem returns_none : returned = none := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.Host
