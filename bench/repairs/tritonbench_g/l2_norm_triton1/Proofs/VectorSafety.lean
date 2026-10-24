import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Evidence
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Native

namespace VeriTile.Bench.Repairs.L2NormTriton1.Native
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Host Physical
noncomputable section
open Classical

/-- Every supported compiler vector width is positive and divides N. This is
derived from block selection and ordinary JIT specialization, not a guard. -/
theorem vector_properties (N k : Nat) (bound : k≤14) :
    let V := vectorWidth N (2^k)
    (V=1 ∨ V=2 ∨ V=4) ∧ 0<V ∧ N%V=0 := by
  by_cases aligned : N%16=0
  · interval_cases k <;> norm_num [vectorWidth,aligned] <;> omega
  · simp [vectorWidth,aligned]
    omega

/-- An active, aligned vector beginning cannot cross an aligned row end. -/
theorem vector_tail (N V start lane : Nat) (positive : 0<V)
    (rowAligned : N%V=0) (startAligned : start%V=0)
    (active : start<N) (inside : lane<V) : start+lane<N := by
  have n : (N/V)*V=N := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero rowAligned)
  have s : (start/V)*V=start := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero startAligned)
  have h : start/V<N/V := (Nat.div_lt_iff_lt_mul positive).mpr (by rwa [n])
  have bound := Nat.mul_le_mul_right V (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul,s,n] at bound
  omega

theorem register_vector_aligned (V thread bundle : Nat) :
    registerIndex V thread (bundle*V)%V=0 := by
  simp [registerIndex,Nat.add_mod,Nat.mul_mod]

/-- Exact vector bundles in the actual prepared allocation: every enabled
v1/v2/v4 component has both backed four-byte spans and representable addresses.
This closes vector-tail addressing separately from compiler/numerical ports. -/
theorem prepared_vector_access {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (entry : World M) (q : Prepared) (ready : World M)
    (facts : PreparedFacts ops neg call entry q ready) (row thread bundle lane : Nat)
    (hr : row<q.layout.rows)
    (active : registerIndex (vectorWidth q.snapshot.cols q.block) thread
      (bundle*vectorWidth q.snapshot.cols q.block)<q.layout.cols)
    (inside : lane<vectorWidth q.snapshot.cols q.block) :
    let col := registerIndex (vectorWidth q.snapshot.cols q.block) thread
      (bundle*vectorWidth q.snapshot.cols q.block)+lane
    q.layout.input.contains (q.layout.xkey row col) ∧
    q.layout.output.contains (q.layout.ykey row col) ∧
    row*q.layout.cols+col<2^62 ∧ (q.layout.xkey row col).address+4≤2^64 ∧
      (q.layout.ykey row col).address+4≤2^64 := by
  have accepted : q.snapshot.cols≤16384 := le_trans facts.valid.blockCovers facts.valid.blockLimit
  have power := block_power q.snapshot.cols accepted
  have block : q.block=2^Nat.clog 2 q.snapshot.cols := facts.block.trans power.1
  have props := vector_properties q.snapshot.cols (Nat.clog 2 q.snapshot.cols) power.2
  rw [←block] at props
  have colBound := vector_tail q.layout.cols (vectorWidth q.snapshot.cols q.block) _ lane
    props.2.1 props.2.2 (register_vector_aligned _ thread bundle) active inside
  exact ⟨xbacked q.layout facts.valid row _ hr colBound,
    ybacked q.layout facts.valid row _ hr colBound,
    active_address q.layout facts.valid row _ hr colBound⟩

/-- Vector instructions require alignment for their complete byte width. -/
theorem vector_address_aligned (base row N start V : Nat)
    (baseAligned : base%(4*V)=0) (rowAligned : N%V=0) (startAligned : start%V=0) :
    (base+4*(row*N+start))%(4*V)=0 := by
  obtain ⟨b,hb⟩ := Nat.dvd_of_mod_eq_zero baseAligned
  obtain ⟨n,hn⟩ := Nat.dvd_of_mod_eq_zero rowAligned
  obtain ⟨j,hj⟩ := Nat.dvd_of_mod_eq_zero startAligned
  apply Nat.mod_eq_zero_of_dvd
  refine ⟨b+row*n+j,?_⟩
  rw [hb,hn,hj]
  ring

theorem prepared_vector_alignment {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (entry : World M) (q : Prepared) (ready : World M)
    (facts : PreparedFacts ops neg call entry q ready) (row thread bundle : Nat)
    (nonempty : 0<q.snapshot.count) :
    let V := vectorWidth q.snapshot.cols q.block
    let start := registerIndex V thread (bundle*V)
    (q.layout.xkey row start).address%(4*V)=0 ∧
      (q.layout.ykey row start).address%(4*V)=0 := by
  have accepted : q.snapshot.cols≤16384 := le_trans facts.valid.blockCovers facts.valid.blockLimit
  have power := block_power q.snapshot.cols accepted
  have block : q.block=2^Nat.clog 2 q.snapshot.cols := facts.block.trans power.1
  have props := vector_properties q.snapshot.cols (Nat.clog 2 q.snapshot.cols) power.2
  rw [←block] at props
  have divisor : 4*vectorWidth q.snapshot.cols q.block∣16 := by
    rcases props.1 with equal | equal | equal <;> rw [equal] <;> norm_num
  have ax := Nat.mod_eq_zero_of_dvd (dvd_trans divisor (Nat.dvd_of_mod_eq_zero (facts.inputAligned nonempty)))
  have ay := Nat.mod_eq_zero_of_dvd (dvd_trans divisor (Nat.dvd_of_mod_eq_zero facts.outputAligned))
  exact ⟨vector_address_aligned _ row q.snapshot.cols _ _ ax props.2.2 (register_vector_aligned _ thread bundle),
    vector_address_aligned _ row q.snapshot.cols _ _ ay props.2.2 (register_vector_aligned _ thread bundle)⟩

#axiomsClean vector_properties
#axiomsClean prepared_vector_access
#axiomsClean prepared_vector_alignment
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Native
