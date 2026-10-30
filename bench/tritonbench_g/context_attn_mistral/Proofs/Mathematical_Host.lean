import bench.tritonbench_g.context_attn_mistral.Proofs.Mathematical_HostInterface
import bench.tritonbench_g.context_attn_mistral.Proofs.Mathematical_RealIO
import bench.tritonbench_g.context_attn_llama.Proofs.Mathematical_HostComposition

namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.Host
open VeriTile.Triton Formula Program
namespace Composition
export VeriTile.Bench.TritonBenchG.ContextAttnLlama.HostComposition (finite_grid input_stable)
end Composition
noncomputable section
set_option maxHeartbeats 1000000

def Call.active (c : Call) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) : Prop :=
  queryRow c.layout (c.launchAt idx) i.1.val < c.length (c.launchAt idx).batch
def Call.offset (c : Call) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) : Nat :=
  c.output (c.launchAt idx).batch (c.launchAt idx).head
    (queryRow c.layout (c.launchAt idx) i.1.val) i.2.1.val

/-- Ordinary tensor interface, expressed directly in its actual strides and
finite allocations. These are metadata, bounds and nonaliasing conditions,
not conclusions about execution or values. Underlaunch is permitted: coverage
of every logical row is a separate consequence of `length ≤ maxInput`. -/
structure Valid (c : Call) (A : FlatAlloc) (s : BlockState) : Prop where
  start_value : ∀ b, b < c.batches → (inputs s c.regions).start b = (c.start b : Int)
  length_value : ∀ b, b < c.batches → (inputs s c.regions).sequence b = (c.length b : Int)
  start_bound : c.batches ≤ A.extent c.regions.start
  length_bound : c.batches ≤ A.extent c.regions.sequence
  group_pos : 0 < c.layout.kvGroup
  grouped : c.heads = c.kvHeads*c.layout.kvGroup
  query : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.qRow+h*c.layout.qHead+d*c.layout.qDim < A.extent c.regions.query
  key : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.kRow+(h/c.layout.kvGroup)*c.layout.kHead+d*c.layout.kDim < A.extent c.regions.key
  value : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.vRow+(h/c.layout.kvGroup)*c.layout.vHead+d*c.layout.vDim < A.extent c.regions.value
  out : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D → c.output b h row d < A.extent c.regions.out
  packed : ∀ b b' row row', b < c.batches → b' < c.batches →
    row < c.length b → row' < c.length b' →
    c.start b+row = c.start b'+row' → b=b' ∧ row=row'
  output_view : ∀ b b' h h' row row' d d', b < c.batches → b' < c.batches →
    h < c.heads → h' < c.heads → row < c.length b → row' < c.length b' →
    d < c.layout.D → d' < c.layout.D →
    c.output b h row d = c.output b' h' row' d' →
    c.start b+row = c.start b'+row' ∧ h=h' ∧ d=d'
  input_separate : ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,
    c.regions.start,c.regions.sequence] → c.regions.out ≠ rg

open Call

theorem batch_bound (c : Call) (idx : GridIndex c.grid) :
    (c.launchAt idx).batch < c.batches := by
  exact (idx ⟨0,by simp [grid,Grid.rank]⟩).isLt
theorem head_bound (c : Call) (idx : GridIndex c.grid) :
    (c.launchAt idx).head < c.heads := by
  exact (idx ⟨1,by simp [grid,Grid.rank]⟩).isLt
theorem block_bound (c : Call) (idx : GridIndex c.grid) :
    (c.launchAt idx).block < c.blocks := by
  exact (idx ⟨2,by simp [grid,Grid.rank]⟩).isLt

theorem launchAt_injective (c : Call) {i j : GridIndex c.grid}
    (hb : (c.launchAt i).batch = (c.launchAt j).batch)
    (hh : (c.launchAt i).head = (c.launchAt j).head)
    (hm : (c.launchAt i).block = (c.launchAt j).block) : i=j := by
  funext ax
  apply Fin.ext
  have ha : ax.val < 3 := ax.isLt
  rcases Nat.lt_trichotomy ax.val 1 with h0 | h1 | h2
  · have he : ax = ⟨0,by simp [grid,Grid.rank]⟩ := Fin.ext (by change ax.val = 0; omega)
    subst ax
    exact hb
  · have he : ax = ⟨1,by simp [grid,Grid.rank]⟩ := Fin.ext h1
    subst ax
    exact hh
  · have he : ax = ⟨2,by simp [grid,Grid.rank]⟩ := Fin.ext (by change ax.val = 2; omega)
    subst ax
    exact hm

theorem active_eq (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) :
    RealIO.active c.layout (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions) i ↔ c.active idx i := by
  change (queryRow c.layout (c.launchAt idx) i.1.val : Int) <
    (inputs s c.regions).sequence (c.launchAt idx).batch ↔ _
  rw [v.length_value _ (batch_bound c idx)]
  exact Int.ofNat_lt

theorem offset_eq (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) :
    RealIO.write c.layout (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions) i = c.offset idx i := by
  change (((inputs s c.regions).start (c.launchAt idx).batch +
    queryRow c.layout (c.launchAt idx) i.1.val) * c.layout.outRow +
    (c.launchAt idx).head*c.layout.outHead+i.2.1.val*c.layout.outDim).toNat = _
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
  have him : i.1.val < 128 := c.blockM ▸ i.1.isLt
  have hjm : j.1.val < 128 := c.blockM ▸ j.1.isLt
  have hblock : (c.launchAt idx).block = (c.launchAt jdx).block := by
    simp only [queryRow,c.blockM] at hr
    omega
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

theorem program_bounds (c : Call) (A : FlatAlloc) (s : BlockState) (v : Valid c A s)
    (idx : GridIndex c.grid) :
    AddressBounds.Valid c.layout c.regions (launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) c.regions) A.extent := by
  have hb := batch_bound c idx
  have hh := head_bound c idx
  have hs := v.start_value _ hb
  have hl := v.length_value _ hb
  constructor
  · change 0 ≤ (inputs s c.regions).start (c.launchAt idx).batch
    rw [hs]
    positivity
  · exact Nat.lt_of_lt_of_le hb v.start_bound
  · exact Nat.lt_of_lt_of_le hb v.length_bound
  · intro i d hi
    change (queryRow c.layout (c.launchAt idx) i.val : Int) <
      (inputs s c.regions).sequence (c.launchAt idx).batch at hi
    rw [hl] at hi
    have ht := v.query _ hb _ hh _ (Int.ofNat_lt.mp hi) _ d.isLt
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ ∧ _
    simp only [AddressBounds.queryAddress] at *
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ _ ∧ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ < _
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast ht
  · intro n _ j d hi
    change ((n*c.layout.BN+j.val : Nat) : Int) <
      (inputs s c.regions).sequence (c.launchAt idx).batch at hi
    rw [hl] at hi
    have ht := v.key _ hb _ hh _ (Int.ofNat_lt.mp hi) _ d.isLt
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ ∧ _
    simp only [AddressBounds.keyAddress] at *
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ _ ∧ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ < _
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast ht
  · intro n _ j d hi
    change ((n*c.layout.BN+j.val : Nat) : Int) <
      (inputs s c.regions).sequence (c.launchAt idx).batch at hi
    rw [hl] at hi
    have ht := v.value _ hb _ hh _ (Int.ofNat_lt.mp hi) _ d.isLt
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ ∧ _
    simp only [AddressBounds.valueAddress] at *
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ _ ∧ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ < _
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast ht
  · intro i d hi
    change (queryRow c.layout (c.launchAt idx) i.val : Int) <
      (inputs s c.regions).sequence (c.launchAt idx).batch at hi
    rw [hl] at hi
    have ht := v.out _ hb _ hh _ (Int.ofNat_lt.mp hi) _ d.isLt
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ ∧ _
    simp only [AddressBounds.outputAddress] at *
    change 0 ≤ _ ∧ _ < (_ : Int)
    change 0 ≤ _ ∧ ((inputs s c.regions).start (c.launchAt idx).batch+_)*_+_+_ < _
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast ht

def Call.writes (c : Call) (A : FlatAlloc) : WriteFootprint :=
  VeriTile.Bench.TritonBenchG.ContextAttnLlama.HostComposition.footprint
    c.grid A c.active c.regions.out c.offset

/-- Both source/candidate launch every grid program. The result is a single
finite memory with each active logical output, including ragged last tiles,
and a full-cell frame. The signed window and rounded finite-sentinel target
are inherited without alteration from the original formula. -/
theorem whole_grid (c : Call) (A : FlatAlloc) (s : BlockState)
    (v : Valid c A s) (ha : A.Disjoint) (hc : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (swapped : Bool) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (kernel c.layout c.regions .fp32 swapped (scale c.layout.D)).eraseDType.toAlgKernel)
      c.grid (A.flattenState s) final) ∧
      (∀ idx i, c.active idx i → final.readMem A.flat (A.addr c.regions.out (c.offset idx i)) =
        RealIO.closedExpected c.layout (scale c.layout.D) (c.launchAt idx) (inputs s c.regions) i) ∧
      BlockState.WriteWithin (c.writes A) (A.flattenState s) final := by
  apply Composition.finite_grid
  · intro idx
    have hv : RealIO.valid c.layout (launch (s.withGridIndex idx))
        (inputs (s.withGridIndex idx) c.regions) := by
      intro i j hi hj he
      rw [offset_eq c A s v idx i,offset_eq c A s v idx j] at he
      exact (writer_unique c A s v idx idx i j
        ((active_eq c A s v idx i).mp hi) ((active_eq c A s v idx j).mp hj) he).2
    obtain ⟨t,he,ho,hf⟩ := RealIO.implements_closed c.layout c.regions .fp32 swapped
      (scale c.layout.D) (by rw [c.blockN]; omega) A ha hc (s.withGridIndex idx)
      hv (program_bounds c A s v idx)
    dsimp only [RealIO.sourceIO] at he ho hf
    refine ⟨t,he,?_,?_⟩
    · intro i hi
      have hx := ho i ((active_eq c A s v idx i).mpr hi)
      change t.readMem A.flat (A.addr c.regions.out (RealIO.write _ _ _ i)) = _ at hx
      rw [offset_eq c A s v idx i] at hx
      exact hx
    · intro rg off hn
      apply hf rg off
      rintro ⟨hr,rr,oo,⟨i,hi,hrr,hoo⟩,heq⟩
      apply hn
      simp only [Prod.fst,Prod.snd] at hrr hoo
      subst rr
      subst oo
      exact ⟨hr,i,(active_eq c A s v idx i).mp hi,by
        rw [←offset_eq c A s v idx i]
        exact heq⟩
  · intro idx jdx i j hi hj he
    exact (writer_unique c A s v idx jdx i j hi hj he).1

/-- The Python wrapper updates the caller's `o` and returns no tensor. -/
def returned : Option Unit := none
theorem returns_none : returned = none := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.Host
