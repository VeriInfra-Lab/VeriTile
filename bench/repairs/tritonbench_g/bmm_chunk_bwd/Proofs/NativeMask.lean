import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeAddress

namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeMask
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open NativeWord
set_option maxHeartbeats 3000000

/-- The scf.for i64 induction variable starts at zero and advances by one. -/
def counter (step : Nat) : Word := NativeAddress.advance (enc 0) (enc 1) step
@[simp] theorem counter_eq (step : Nat) : counter step = enc step := by
  rw [counter,NativeAddress.advance_word,enc_add]
  simp

def dout (h : HostCall) (c : Tuning) (q : HostFP.Pid) (step : Nat) (i : TileIndex [c.BM,c.BCS]) : Bool :=
  let s := NativeScalar.setup h c q
  (NativeScalar.row s c.BM i.1).slt (enc h.chunk) &&
    (enc i.2.1.val).slt (s.length-counter step*enc c.BCS)
def a (h : HostCall) (c : Tuning) (q : HostFP.Pid) (step : Nat) (i : TileIndex [c.BCS,c.BN]) : Bool :=
  let s := NativeScalar.setup h c q
  (enc i.1.val).slt (s.length-counter step*enc c.BCS) &&
    (NativeScalar.column s c.BN i.2.1).slt (enc h.K)
def output (h : HostCall) (c : Tuning) (q : HostFP.Pid) (i : TileIndex [c.BM,c.BN]) : Bool :=
  let s := NativeScalar.setup h c q
  (NativeScalar.row s c.BM i.1).slt s.length &&
    (NativeScalar.column s c.BN i.2.1).slt (enc h.K)

theorem scalar_bounds (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) :
    h.chunk < limit ∧ h.K < limit ∧ FPLogical.length h q < limit ∧
    (∀ i : Fin (tuning choice).BM, FPLogical.row h (tuning choice) q i < limit) ∧
    (∀ i : Fin (tuning choice).BN, FPLogical.column h (tuning choice) q i < limit) ∧
    (∀ i : Fin (tuning choice).BCS, i.val < limit) := by
  have qr := FPGridOutput.grid_member _ _ _ hq
  have sz := NativeControl.sizes h choice metadata _ qr
  have ln := NativeControl.lanes h choice metadata _ qr
  have tc := NativeControl.tuning_bounds choice
  have hc : h.chunk < limit := by norm_num [NativeControl.signedLimit] at sz ⊢; omega
  have hk : h.K < limit := by norm_num [NativeControl.signedLimit] at sz ⊢; omega
  refine ⟨hc,hk,lt_of_le_of_lt (Nat.min_le_left _ _) hc,ln.2.2.1,ln.2.2.2,?_⟩
  intro i
  have hh := i.isLt
  have hb := tc.2.2.2.2.2
  norm_num [NativeControl.signedLimit]
  omega

theorem dout_exact (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) (step : Nat)
    (hs : step < (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS)
    (i : TileIndex [(tuning choice).BM,(tuning choice).BCS]) :
    dout h (tuning choice) q step i = decide (FPLogical.row h (tuning choice) q i.1 < h.chunk ∧
      i.2.1.val < FPLogical.length h q-step*(tuning choice).BCS) := by
  have b := scalar_bounds h choice metadata q hq
  have remBound : FPLogical.length h q-step*(tuning choice).BCS < limit :=
    lt_of_le_of_lt (Nat.sub_le _ _) b.2.2.1
  dsimp only [dout]
  rw [counter_eq,NativeScalar.remaining_exact h choice metadata q hq step hs,
    NativeScalar.setup_exact h choice metadata q hq,NativeScalar.row_exact,
    enc_lt _ _ (b.2.2.2.1 i.1) b.1,enc_lt _ _ (b.2.2.2.2.2 i.2.1) remBound]
  simp

theorem a_exact (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) (step : Nat)
    (hs : step < (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS)
    (i : TileIndex [(tuning choice).BCS,(tuning choice).BN]) :
    a h (tuning choice) q step i = decide (i.1.val < FPLogical.length h q-step*(tuning choice).BCS ∧
      FPLogical.column h (tuning choice) q i.2.1 < h.K) := by
  have b := scalar_bounds h choice metadata q hq
  have remBound : FPLogical.length h q-step*(tuning choice).BCS < limit :=
    lt_of_le_of_lt (Nat.sub_le _ _) b.2.2.1
  dsimp only [a]
  rw [counter_eq,NativeScalar.remaining_exact h choice metadata q hq step hs,
    NativeScalar.setup_exact h choice metadata q hq,NativeScalar.column_exact,
    enc_lt _ _ (b.2.2.2.2.2 i.1) remBound,enc_lt _ _ (b.2.2.2.2.1 i.2.1) b.2.1]
  simp

theorem output_exact (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice))
    (i : TileIndex [(tuning choice).BM,(tuning choice).BN]) :
    output h (tuning choice) q i = decide (FPLogical.row h (tuning choice) q i.1 < FPLogical.length h q ∧
      FPLogical.column h (tuning choice) q i.2.1 < h.K) := by
  have b := scalar_bounds h choice metadata q hq
  dsimp only [output]
  rw [NativeScalar.setup_exact h choice metadata q hq,NativeScalar.row_exact,NativeScalar.column_exact]
  change ((enc (FPLogical.row h (tuning choice) q i.1)).slt (enc (FPLogical.length h q)) && _) = _
  rw [enc_lt _ _ (b.2.2.2.1 i.1) b.2.2.1,enc_lt _ _ (b.2.2.2.2.1 i.2.1) b.2.1]
  simp

theorem stop_exact (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) :
    (NativeScalar.setup h (tuning choice) q).stop.toNat =
      (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS := by
  rw [NativeScalar.setup_exact h choice metadata q hq]
  apply enc_nat
  have hb := (NativeControl.chunk_and_loop h choice metadata _ (FPGridOutput.grid_member _ _ _ hq)).2.1
  have hd := Nat.div_le_self (FPLogical.length h q+(tuning choice).BCS-1) (tuning choice).BCS
  change FPLogical.length h q+(tuning choice).BCS-1 < limit at hb
  norm_num [NativeControl.signedLimit] at hb ⊢
  omega

/-- Every reached loop test, including the final false test, uses the same
signed comparison as the source native range. -/
theorem counter_test (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) (step : Nat)
    (hs : step ≤ (NativeScalar.setup h (tuning choice) q).stop.toNat) :
    (counter step).slt (NativeScalar.setup h (tuning choice) q).stop =
      decide (step < (NativeScalar.setup h (tuning choice) q).stop.toNat) := by
  have hb := (NativeControl.chunk_and_loop h choice metadata _ (FPGridOutput.grid_member _ _ _ hq)).2.1
  have hd := Nat.div_le_self (FPLogical.length h q+(tuning choice).BCS-1) (tuning choice).BCS
  have stopBound : (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS < limit :=
    lt_of_le_of_lt hd hb
  rw [stop_exact h choice metadata q hq] at hs ⊢
  rw [counter_eq,NativeScalar.setup_exact h choice metadata q hq]
  exact enc_lt _ _ (lt_of_le_of_lt hs stopBound) stopBound

#axiomsClean dout_exact
#axiomsClean a_exact
#axiomsClean output_exact
#axiomsClean stop_exact
#axiomsClean counter_test
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeMask
