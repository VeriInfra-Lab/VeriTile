import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Evidence
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Native
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.VectorSafety

namespace VeriTile.Bench.Repairs.L2NormTriton2.Native
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Host Physical
open L2NormTriton1.Host (World block_power)
open L2NormTriton1.Native (registerIndex vector_properties vector_tail register_vector_aligned vector_address_aligned)
noncomputable section
open Classical

/-- Exact vector bundles in the actual prepared allocation: every enabled
v1/v2/v4 component has both backed four-byte spans and representable addresses.
This closes vector-tail addressing separately from compiler/numerical ports. -/
theorem prepared_vector_access {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (entry : World M) (q : Prepared) (ready : World M)
    (facts : PreparedFacts ops neg call entry q ready) (gradient : Bool) (row thread bundle lane : Nat)
    (hr : row<q.layout.rows)
    (active : registerIndex (vectorWidth q.snapshot.cols q.block) thread
      (bundle*vectorWidth q.snapshot.cols q.block)<q.layout.cols)
    (inside : lane<vectorWidth q.snapshot.cols q.block) :
    let col := registerIndex (vectorWidth q.snapshot.cols q.block) thread
      (bundle*vectorWidth q.snapshot.cols q.block)+lane
    (q.layout.inputFor gradient).contains (q.layout.xkey gradient row col) ∧
    q.layout.output.contains (q.layout.ykey row col) ∧
    row*q.layout.cols+col<2^62 ∧ (q.layout.xkey gradient row col).address+4≤2^64 ∧
      (q.layout.ykey row col).address+4≤2^64 := by
  have accepted : q.snapshot.cols≤16384 := le_trans facts.valid.main.blockCovers facts.valid.main.blockLimit
  have power := block_power q.snapshot.cols accepted
  have block : q.block=2^Nat.clog 2 q.snapshot.cols := facts.block.trans power.1
  have props := vector_properties q.snapshot.cols (Nat.clog 2 q.snapshot.cols) power.2
  rw [←block] at props
  have colBound := vector_tail q.layout.cols (vectorWidth q.snapshot.cols q.block) _ lane
    props.2.1 props.2.2 (register_vector_aligned _ thread bundle) active inside
  exact ⟨xbacked q.layout facts.valid gradient row _ hr colBound,
    ybacked q.layout facts.valid row _ hr colBound,
    active_address q.layout facts.valid gradient row _ hr colBound⟩

theorem prepared_vector_alignment {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (entry : World M) (q : Prepared) (ready : World M)
    (facts : PreparedFacts ops neg call entry q ready) (gradient : Bool) (row thread bundle : Nat)
    (nonempty : 0<q.snapshot.count) :
    let V := vectorWidth q.snapshot.cols q.block
    let start := registerIndex V thread (bundle*V)
    (q.layout.xkey gradient row start).address%(4*V)=0 ∧
      (q.layout.ykey row start).address%(4*V)=0 := by
  have accepted : q.snapshot.cols≤16384 := le_trans facts.valid.main.blockCovers facts.valid.main.blockLimit
  have power := block_power q.snapshot.cols accepted
  have block : q.block=2^Nat.clog 2 q.snapshot.cols := facts.block.trans power.1
  have props := vector_properties q.snapshot.cols (Nat.clog 2 q.snapshot.cols) power.2
  rw [←block] at props
  have divisor : 4*vectorWidth q.snapshot.cols q.block∣16 := by
    dsimp only [vectorWidth]
    rcases props.1 with equal | equal | equal <;> rw [equal] <;> norm_num
  have inputAligned : (q.layout.inputFor gradient).baseByte%16=0 := by
    cases gradient
    · exact facts.inputAligned nonempty
    · exact facts.gradientAligned nonempty
  have ax := Nat.mod_eq_zero_of_dvd (dvd_trans divisor (Nat.dvd_of_mod_eq_zero inputAligned))
  have ay := Nat.mod_eq_zero_of_dvd (dvd_trans divisor (Nat.dvd_of_mod_eq_zero facts.outputAligned))
  exact ⟨vector_address_aligned _ row q.snapshot.cols _ _ ax props.2.2 (register_vector_aligned _ thread bundle),
    vector_address_aligned _ row q.snapshot.cols _ _ ay props.2.2 (register_vector_aligned _ thread bundle)⟩

#axiomsClean prepared_vector_access
#axiomsClean prepared_vector_alignment
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Native
