import bench.tritonbench_g.context_attn_nopad.Proofs.HostConfig
import bench.tritonbench_g.context_attn_nopad.Proofs.RealIO

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 800000

variable {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}

theorem active_iff (hm : Memory c r A s) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) :
    RealIO.active c.layout (c.launch idx) (inputs s r) i ↔
      queryRow c.layout (c.launch idx) i.1.val < c.length (c.launch idx).batch := by
  simp only [RealIO.active, Formula.sequence, hm.length _ (launch_batch_lt c idx)]
  exact Int.ofNat_lt

theorem write_eq (hm : Memory c r A s) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout) :
    RealIO.write c.layout (c.launch idx) (inputs s r) i =
      c.output (c.launch idx).batch (c.launch idx).head
        (queryRow c.layout (c.launch idx) i.1.val) i.2.1.val := by
  simp only [RealIO.write, AddressBounds.outputAddress, hm.start _ (launch_batch_lt c idx),
    Call.output, Tensor3.offset, Call.layout, ← Int.natCast_add, ← Int.natCast_mul, Int.toNat_natCast]

theorem logical_row_lt (hm : Memory c r A s) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout)
    (hi : RealIO.active c.layout (c.launch idx) (inputs s r) i) :
    c.start (c.launch idx).batch + queryRow c.layout (c.launch idx) i.1.val < c.o.rows := by
  have ht := (active_iff hm idx i).mp hi
  have hr := hm.shape.oRows _ (launch_batch_lt c idx)
  omega

theorem bounds (hm : Memory c r A s) (idx : GridIndex c.grid) :
    AddressBounds.Valid c.layout r (c.launch idx) (inputs s r) A.extent := by
  have hb := launch_batch_lt c idx
  have hh := launch_head_lt c idx
  have hs := hm.start _ hb
  have hn := hm.length _ hb
  constructor
  · rw [hs]; exact Int.natCast_nonneg _
  · exact lt_of_lt_of_le hb hm.startExtent
  · exact lt_of_lt_of_le hb hm.lengthExtent
  · intro i d hi
    have ht : queryRow c.layout (c.launch idx) i.val < c.length (c.launch idx).batch := by
      simp only [Formula.sequence, hn] at hi
      exact_mod_cast hi
    have hr := hm.shape.qRows _ hb
    have hout := hm.qBounded (c.start (c.launch idx).batch + queryRow c.layout (c.launch idx) i.val)
      (c.launch idx).head d.val (by omega) hh d.isLt
    simp only [AddressBounds.inside, AddressBounds.queryAddress, hs]
    constructor
    · positivity
    · exact_mod_cast hout
  · intro count _ j d hi
    have ht : count*c.layout.BN+j.val < c.length (c.launch idx).batch := by
      simp only [Formula.sequence, hn] at hi
      exact_mod_cast hi
    have hr := hm.shape.kRows _ hb
    have hd : d.val < c.k.dims := by rw [hm.shape.kDims]; exact d.isLt
    have hout := hm.kBounded (c.start (c.launch idx).batch + (count*c.layout.BN+j.val))
      (c.launch idx).head d.val (by omega) (lt_of_lt_of_le hh hm.shape.kHeads) hd
    simp only [AddressBounds.inside, AddressBounds.keyAddress, hs]
    constructor
    · positivity
    · exact_mod_cast hout
  · intro count _ j d hi
    have ht : count*c.layout.BN+j.val < c.length (c.launch idx).batch := by
      simp only [Formula.sequence, hn] at hi
      exact_mod_cast hi
    have hr := hm.shape.vRows _ hb
    have hd : d.val < c.v.dims := by rw [hm.shape.vDims]; exact d.isLt
    have hout := hm.vBounded (c.start (c.launch idx).batch + (count*c.layout.BN+j.val))
      (c.launch idx).head d.val (by omega) (lt_of_lt_of_le hh hm.shape.vHeads) hd
    simp only [AddressBounds.inside, AddressBounds.valueAddress, hs]
    constructor
    · positivity
    · exact_mod_cast hout
  · intro i d hi
    have ht : queryRow c.layout (c.launch idx) i.val < c.length (c.launch idx).batch := by
      simp only [Formula.sequence, hn] at hi
      exact_mod_cast hi
    have hr := hm.shape.oRows _ hb
    have hout := hm.oBounded (c.start (c.launch idx).batch + queryRow c.layout (c.launch idx) i.val)
      (c.launch idx).head d.val (by omega) (lt_of_lt_of_le hh hm.shape.oHeads)
      (lt_of_lt_of_le d.isLt hm.shape.oDims)
    simp only [AddressBounds.inside, AddressBounds.outputAddress, hs]
    constructor
    · positivity
    · exact_mod_cast hout

theorem launch_injective (c : Call) {idx idx' : GridIndex c.grid}
    (he : c.launch idx = c.launch idx') : idx = idx' := by
  funext ax
  apply Fin.ext
  have hb := congrArg Launch.batch he
  have hh := congrArg Launch.head he
  have ht := congrArg Launch.block he
  rcases ax with ⟨ax, ha⟩
  have haxis : ax < 3 := ha
  interval_cases ax <;> assumption

/-- Unique physical writer across the entire actual grid, derived from logical
tensor injectivity and disjoint metadata intervals, including partial tiles. -/
theorem unique_writer (hm : Memory c r A s) (idx idx' : GridIndex c.grid)
    (i j : RealIO.Lane c.layout)
    (hi : RealIO.active c.layout (c.launch idx) (inputs s r) i)
    (hj : RealIO.active c.layout (c.launch idx') (inputs s r) j)
    (he : RealIO.write c.layout (c.launch idx) (inputs s r) i =
      RealIO.write c.layout (c.launch idx') (inputs s r) j) : idx = idx' ∧ i = j := by
  rw [write_eq hm, write_eq hm] at he
  obtain ⟨hrow, hhead, hdim⟩ := hm.shape.outputInjective _ _ _ _ _ _
    (logical_row_lt hm idx i hi) (logical_row_lt hm idx' j hj)
    (lt_of_lt_of_le (launch_head_lt c idx) hm.shape.oHeads)
    (lt_of_lt_of_le (launch_head_lt c idx') hm.shape.oHeads)
    (lt_of_lt_of_le i.2.1.isLt hm.shape.oDims)
    (lt_of_lt_of_le j.2.1.isLt hm.shape.oDims) he
  obtain ⟨hbatch, htime⟩ := hm.shape.intervals _ _ _ _
    (launch_batch_lt c idx) (launch_batch_lt c idx')
    ((active_iff hm idx i).mp hi) ((active_iff hm idx' j).mp hj) hrow
  have hiB : i.1.val < 128 := i.1.isLt
  have hjB : j.1.val < 128 := j.1.isLt
  have ht : (c.launch idx).block * 128 + i.1.val =
      (c.launch idx').block * 128 + j.1.val := htime
  have hblock : (c.launch idx).block = (c.launch idx').block := by omega
  have hlane : i.1.val = j.1.val := by omega
  constructor
  · apply launch_injective
    cases h₁ : c.launch idx
    cases h₂ : c.launch idx'
    simp_all
  · apply Prod.ext (Fin.ext hlane)
    exact Prod.ext (Fin.ext hdim) (Subsingleton.elim _ _)

theorem valid (hm : Memory c r A s) (idx : GridIndex c.grid) :
    RealIO.valid c.layout (c.launch idx) (inputs s r) := by
  intro i j hi hj he
  exact (unique_writer hm idx idx i j hi hj he).2

end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
