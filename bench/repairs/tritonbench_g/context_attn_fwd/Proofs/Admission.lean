import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Geometry
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Geometry
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
open Formula Program
noncomputable section
set_option maxHeartbeats 4000000

/-- Same three observed INT32 vectors in the real source state. -/
def Metadata (c : Host.Call) (s : BlockState) : Prop :=
  (∀ b,b<c.batches → (inputs s c.regions).start b=(c.start b : Int)) ∧
  (∀ b,b<c.batches → (inputs s c.regions).sequence b=((c.length b+c.prompt b : Nat) : Int)) ∧
  (∀ b,b<c.batches → (inputs s c.regions).prompt b=(c.prompt b : Int))

theorem query_bound (c : Host.Call) (hc : Checked c extent) (b h row d : Nat)
    (hb : b<c.batches) (hh : h<c.layout.heads) (hr : row<c.length b) (hd : d<c.layout.D) :
    (c.start b+row)*c.layout.qRow+h*c.layout.qHead+d*c.layout.qDim<extent c.regions.query := by
  have hm := hc.metadata b hb
  have qr : c.start b+row<c.args.q.rows := by omega
  have qh : h<c.args.q.heads := hh
  have qd : d<c.args.q.dims := by simpa [Host.Call.layout,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.layout,hc.qk_dims] using hd
  exact lt_of_lt_of_le (address3_lt c.args.q _ _ _ qr qh qd)
    (hc.query_span (by omega) (by omega) (by omega))

theorem output_bound (c : Host.Call) (hc : Checked c extent) (b h row d : Nat)
    (hb : b<c.batches) (hh : h<c.layout.heads) (hr : row<c.length b) (hd : d<c.layout.D) :
    c.output b h row d<extent c.regions.out := by
  have hm := hc.metadata b hb
  have qr : c.start b+row<c.args.o.rows := by rw [hc.output_rows]; omega
  have qh : h<c.args.o.heads := by rw [hc.output_heads]; exact hh
  have qd : d<c.args.o.dims := by rw [hc.output_dims]; exact hd
  exact lt_of_lt_of_le (address3_lt c.args.o _ _ _ qr qh qd)
    (hc.output_span (by omega) (by omega) (by omega))

theorem key_bound (c : Host.Call) (hc : Checked c extent) (b h row d : Nat)
    (hb : b<c.batches) (hh : h<c.layout.heads)
    (hr : row<c.length b+c.prompt b) (hd : d<c.layout.D) :
    b*c.layout.kBatch+row*c.layout.kRow+(h/c.layout.kvGroup)*c.layout.kHead+d*c.layout.kDim<extent c.regions.key := by
  have hm := hc.metadata b hb
  have qb : b<c.args.k.batches := lt_of_lt_of_le hb hc.key_batches
  have qr : row<c.args.k.rows := by omega
  have qh : h/c.layout.kvGroup<c.args.k.heads := kv_head_bound c hc h hh
  have qd : d<c.args.k.dims := hd
  exact lt_of_lt_of_le (address4_lt c.args.k _ _ _ _ qb qr qh qd)
    (hc.key_span (Nat.zero_lt_of_lt qb) (Nat.zero_lt_of_lt qr) hc.kv_positive (Nat.zero_lt_of_lt qd))

theorem value_bound (c : Host.Call) (hc : Checked c extent) (b h row d : Nat)
    (hb : b<c.batches) (hh : h<c.layout.heads)
    (hr : row<c.length b+c.prompt b) (hd : d<c.layout.D) :
    b*c.layout.vBatch+row*c.layout.vRow+(h/c.layout.kvGroup)*c.layout.vHead+d*c.layout.vDim<extent c.regions.value := by
  have hm := hc.metadata b hb
  have qb : b<c.args.v.batches := lt_of_lt_of_le hb hc.value_batches
  have qr : row<c.args.v.rows := by omega
  have qh : h/c.layout.kvGroup<c.args.v.heads := by rw [hc.value_heads]; exact kv_head_bound c hc h hh
  have qd : d<c.args.v.dims := by rw [←hc.kv_dims]; exact hd
  exact lt_of_lt_of_le (address4_lt c.args.v _ _ _ _ qb qr qh qd)
    (hc.value_span (Nat.zero_lt_of_lt qb) (Nat.zero_lt_of_lt qr) (Nat.zero_lt_of_lt qh) (Nat.zero_lt_of_lt qd))

theorem address_bounds (c : Host.Call) (hc : Checked c extent) (s : BlockState)
    (hm : Metadata c s) (idx : GridIndex c.grid) :
    AddressBounds.Valid c.layout c.regions (c.launchAt idx) (inputs s c.regions) extent := by
  have hb := Host.batch_bound c idx
  have hh := Host.head_bound c idx
  have hs := hm.1 _ hb
  have hl := hm.2.1 _ hb
  have hp := hm.2.2 _ hb
  have seq : Formula.sequence c.layout (c.launchAt idx) (inputs s c.regions) =
      (c.length (batch c.layout (c.launchAt idx)) : Int) := by simp [Formula.sequence,hl,hp]
  have lim : Formula.limit c.layout (c.launchAt idx) (inputs s c.regions) ≤
      ((c.length (batch c.layout (c.launchAt idx))+c.prompt (batch c.layout (c.launchAt idx)) : Nat) : Int) := by
    rw [Formula.limit,hl]; exact min_le_right _ _
  refine ⟨lt_of_lt_of_le hb hc.start_span,lt_of_lt_of_le hb hc.prompt_span,
    lt_of_lt_of_le hb hc.sequence_span,?_,?_,?_,?_⟩
  · intro i d hi
    rw [seq] at hi
    have h := query_bound c hc _ _ _ _ hb hh (by exact_mod_cast hi) d.isLt
    unfold AddressBounds.inside AddressBounds.queryAddress
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast h
  · intro step _ j d hj
    have h := key_bound c hc _ _ _ _ hb hh (by exact_mod_cast lt_of_lt_of_le hj lim) d.isLt
    unfold AddressBounds.inside AddressBounds.keyAddress
    constructor
    · positivity
    · exact_mod_cast h
  · intro step _ j d hj
    have h := value_bound c hc _ _ _ _ hb hh (by exact_mod_cast lt_of_lt_of_le hj lim) d.isLt
    unfold AddressBounds.inside AddressBounds.valueAddress
    constructor
    · positivity
    · exact_mod_cast h
  · intro i d hi
    rw [seq] at hi
    have h := output_bound c hc _ _ _ _ hb hh (by exact_mod_cast hi) d.isLt
    unfold AddressBounds.inside AddressBounds.outputAddress
    rw [hs]
    constructor
    · positivity
    · exact_mod_cast h

theorem packed_unique (c : Host.Call) (hc : Checked c extent) (b z : Nat)
    (hb : b<c.batches) (hz : z<c.batches) (i j : Int)
    (hi : 0 ≤ i) (hil : i<(c.length b : Int)) (hj : 0 ≤ j) (hjl : j<(c.length z : Int))
    (he : (c.start b : Int)+i=(c.start z : Int)+j) : b=z ∧ i=j := by
  by_cases hbz : b=z
  · subst z; exact ⟨rfl,by omega⟩
  · have hp := hc.packed b z hb hz (by omega) (by omega) hbz
    rcases hp with hp|hp <;> omega

/-- All active address checks and unique output ownership follow from the
executable descriptor checks. They are not additional admission assumptions. -/
theorem admissible (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (hc : Checked c A.extent) (hm : Metadata c s) (sep : FPSuccess.Separate c) :
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.Admissible c.args c.regions A s := by
  refine ⟨c.headsPositive,hc.kv_positive,hc.grouped,hc.qk_dims,hc.kv_dims,hc.output_dims,
    hc.output_heads,hc.value_heads,hc.width,hc.key_batches,hc.value_batches,?_,?_,
    contiguous_injective _ hc.contiguous,?_,?_⟩
  · intro b hb
    rw [hm.1 b hb,hm.2.1 b hb,hm.2.2 b hb]
    have h := hc.metadata b hb
    rw [hc.output_rows]
    simp only [Nat.cast_add,add_sub_cancel_right]
    constructor
    · positivity
    constructor
    · positivity
    constructor
    · omega
    exact ⟨by exact_mod_cast h.1,by exact_mod_cast h.2.1,by exact_mod_cast h.2.1,by exact_mod_cast (show c.length b+c.prompt b≤c.args.k.rows by omega),by exact_mod_cast (show c.length b+c.prompt b≤c.args.v.rows by omega)⟩
  · intro b z hb hz i j hi hil hj hjl he
    rw [hm.2.1 b hb,hm.2.2 b hb] at hil
    rw [hm.2.1 z hz,hm.2.2 z hz] at hjl
    rw [hm.1 b hb,hm.1 z hz] at he
    exact packed_unique c hc b z hb hz i j hi (by omega) hj (by omega) he
  · exact ⟨sep _ (by simp),sep _ (by simp),sep _ (by simp),sep _ (by simp),sep _ (by simp),sep _ (by simp)⟩
  · intro idx
    exact address_bounds c hc s hm idx

end
end VeriTile.Bench.Repairs.ContextAttnFwd.Geometry
