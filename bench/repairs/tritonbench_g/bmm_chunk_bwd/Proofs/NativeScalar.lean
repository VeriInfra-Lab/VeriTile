import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeWord
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPRoles

namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeScalar
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open NativeWord
set_option maxHeartbeats 3000000

/-- Triton specializes small Python integer arguments as signed i32 and large
ones as i64. The repaired entry casts both representations before arithmetic. -/
inductive ABIValue where
  | i32 (bits : BitVec 32)
  | i64 (bits : Word)
def ABIValue.widen : ABIValue → Word
  | .i32 bits => bits.signExtend 64
  | .i64 bits => bits

def abi (n : Nat) : ABIValue := if n < 2^31 then .i32 (BitVec.ofNat 32 n) else .i64 (enc n)

theorem widen32 (n : Nat) (hn : n < 2^31) : (BitVec.ofNat 32 n).signExtend 64 = enc n := by
  have hm : (BitVec.ofNat 32 n).msb = Bool.false := by
    apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (show n < 2^32 by omega)]
    omega
  rw [BitVec.signExtend_eq_setWidth_of_msb_false hm]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_setWidth,BitVec.toNat_ofNat,enc]
  rw [Nat.mod_eq_of_lt (show n < 2^32 by omega)]

theorem abi_exact (n : Nat) : (abi n).widen = enc n := by
  unfold abi
  split_ifs with h
  · exact widen32 n h
  · rfl

/-- Exact source ABI order: four dimensions, four A strides, five Dout
strides, four output strides, and four residual strides. -/
def scalars (h : HostCall) : List Nat :=
  [h.seqlen,h.chunk,h.K,h.groupCount,
   h.aStrides.batch,h.aStrides.sequence,h.headStride h.aStrides.head,h.aStrides.feature,
   h.doutStrides.batch,h.doutStrides.chunk,h.headStride h.doutStrides.head,h.doutStrides.row,h.doutStrides.column,
   h.outStrides.batch,h.outStrides.sequence,h.headStride h.outStrides.head,h.outStrides.feature,
   h.resStride h.residualStrides.batch,h.resStride h.residualStrides.sequence,
   h.resStride (h.headStride h.residualStrides.head),h.resStride h.residualStrides.feature]

theorem all_21_widened (h : HostCall) :
    (scalars h).length = 21 ∧ ((scalars h).map (fun n => (abi n).widen)) = (scalars h).map enc := by
  exact ⟨rfl,List.map_congr_left (fun n _ => abi_exact n)⟩

structure Control where
  batch : Word
  chunk : Word
  head : Word
  columns : Word
  tileM : Word
  tileN : Word
  length : Word
  stop : Word

def setup (h : HostCall) (c : Tuning) (q : HostFP.Pid) : Control :=
  let b := enc q.2.1
  let ch := enc q.2.2
  let chunk := ch.sdiv (enc h.groupCount)
  let head := ch-chunk*enc h.groupCount
  let columns := cdiv (enc h.K) (enc c.BN)
  let tileM := (enc q.1).sdiv columns
  let tileN := (enc q.1).srem columns
  let length := NativeWord.min (enc h.chunk) (enc h.seqlen-chunk*enc h.chunk)
  ⟨b,chunk,head,columns,tileM,tileN,length,cdiv length (enc c.BCS)⟩

def ideal (h : HostCall) (c : Tuning) (q : HostFP.Pid) : Control :=
  ⟨enc q.2.1,enc (FPLogical.chunk h q),enc (FPLogical.head h q),enc ((h.K+c.BN-1)/c.BN),
   enc (q.1/((h.K+c.BN-1)/c.BN)),enc (q.1%((h.K+c.BN-1)/c.BN)),
   enc (FPLogical.length h q),enc ((FPLogical.length h q+c.BCS-1)/c.BCS)⟩

/-- Complete signed setup, from successful allocation and hardware launch
metadata. No arithmetic or mask equivalence is assumed. -/
theorem setup_exact (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) :
    setup h (tuning choice) q = ideal h (tuning choice) q := by
  let c := tuning choice
  have hqr := FPGridOutput.grid_member h c q hq
  have sz := NativeControl.sizes h choice metadata _ hqr
  have lanes := NativeControl.lanes h choice metadata _ hqr
  have cl := NativeControl.chunk_and_loop h choice metadata _ hqr
  have tb := NativeControl.tuning_bounds choice
  have hp := (HostGrid.mem_programs _ _).mp hqr
  have hqt : q.1 < limit := by
    have hh := lt_of_lt_of_le hp.1 metadata.gridX
    change q.1 < 2147483647 at hh
    norm_num [NativeControl.signedLimit]; omega
  have hqc : q.2.2 < limit := by
    have hh := lt_of_lt_of_le hp.2.2 metadata.gridZ
    change q.2.2 < 65535 at hh
    norm_num [NativeControl.signedLimit]; omega
  have hg : h.groupCount < limit := by norm_num [NativeControl.signedLimit]; omega
  have hk : h.K < limit := by norm_num [NativeControl.signedLimit] at sz ⊢; omega
  have hs : h.seqlen < limit := by norm_num [NativeControl.signedLimit] at sz ⊢; omega
  have hc : h.chunk < limit := by norm_num [NativeControl.signedLimit] at sz ⊢; omega
  have hbn : c.BN < limit := by norm_num [NativeControl.signedLimit]; change _ ∧ _ ∧ _ ∧ c.BN ≤ _ ∧ _ at tb; omega
  have hbcs : c.BCS < limit := by norm_num [NativeControl.signedLimit]; change _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ c.BCS ≤ _ at tb; omega
  have hn : (h.K+c.BN-1)/c.BN < limit := lt_of_le_of_lt (Nat.div_le_self _ _) lanes.1
  have headBound : q.2.2/h.groupCount*h.groupCount < limit := lt_of_le_of_lt (Nat.div_mul_le_self _ _) hqc
  have remain : h.seqlen-q.2.2/h.groupCount*h.chunk < limit := lt_of_le_of_lt (Nat.sub_le _ _) hs
  have hstart : q.2.2/h.groupCount*h.chunk ≤ h.seqlen := Nat.le_of_lt cl.1
  have lengthBound : FPLogical.length h q < limit := lt_of_le_of_lt (Nat.min_le_left _ _) hc
  dsimp only [c] at hn hbn hbcs
  simp only [setup,enc_div _ _ hqc hg,enc_mul,
    enc_sub _ _ headBound (Nat.div_mul_le_self _ _),
    enc_cdiv _ _ tb.2.2.1 hbn lanes.1,enc_div _ _ hqt hn,enc_rem _ _ hqt hn,
    enc_sub _ _ (lt_of_le_of_lt hstart hs) hstart,enc_min _ _ hc remain,
    enc_cdiv _ _ tb.2.2.2.2.1 hbcs cl.2.1]
  rfl

def row (s : Control) (BM : Nat) (i : Fin BM) := s.tileM*enc BM+enc i.val
def column (s : Control) (BN : Nat) (j : Fin BN) := s.tileN*enc BN+enc j.val

theorem row_exact (h : HostCall) (c : Tuning) (q : HostFP.Pid) (i : Fin c.BM) :
    row (ideal h c q) c.BM i = enc (FPLogical.row h c q i) := by
  simp only [row,ideal,enc_mul,enc_add]
  rfl

theorem column_exact (h : HostCall) (c : Tuning) (q : HostFP.Pid) (j : Fin c.BN) :
    column (ideal h c q) c.BN j = enc (FPLogical.column h c q j) := by
  simp only [column,ideal,enc_mul,enc_add]
  rfl

/-- Exact dynamic range bound and every reached remaining-length subtraction.
The final pointer increment may wrap; no next load follows that increment. -/
theorem remaining_exact (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) (step : Nat)
    (hstep : step < (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS) :
    (setup h (tuning choice) q).length-enc step*enc (tuning choice).BCS =
      enc (FPLogical.length h q-step*(tuning choice).BCS) := by
  rw [setup_exact h choice metadata q hq]
  have hh := NativeControl.chunk_and_loop h choice metadata _ (FPGridOutput.grid_member _ _ _ hq)
  have hc := (NativeControl.sizes h choice metadata _ (FPGridOutput.grid_member _ _ _ hq)).2.2.1
  have he := hh.2.2 step hstep
  change enc (FPLogical.length h q)-enc step*enc (tuning choice).BCS = _
  rw [enc_mul]
  apply enc_sub
  swap
  · exact Nat.le_of_lt he.1
  have hl : FPLogical.length h q ≤ h.chunk := Nat.min_le_left _ _
  norm_num [NativeControl.signedLimit] at hc ⊢
  omega

/-- Hardware program IDs really arrive as i32; their sign extension agrees
with the encodings used in setup. The grid's dimensions retain their original
CUDA limits. Arange lanes fit in i32 for each of the nine source configurations. -/
theorem pids_widened (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) :
    (BitVec.ofNat 32 q.1).signExtend 64 = enc q.1 ∧
    (BitVec.ofNat 32 q.2.1).signExtend 64 = enc q.2.1 ∧
    (BitVec.ofNat 32 q.2.2).signExtend 64 = enc q.2.2 := by
  have hh := (HostGrid.mem_programs _ _).mp (FPGridOutput.grid_member _ _ _ hq)
  have hx := lt_of_lt_of_le hh.1 metadata.gridX
  have hy := lt_of_lt_of_le hh.2.1 metadata.gridY
  have hz := lt_of_lt_of_le hh.2.2 metadata.gridZ
  change q.1 < 2147483647 at hx
  change q.2.1 < 65535 at hy
  change q.2.2 < 65535 at hz
  exact ⟨widen32 _ (by omega),widen32 _ (by omega),widen32 _ (by omega)⟩

theorem lanes_widened (choice : Fin 9) :
    (∀ i : Fin (tuning choice).BM, (BitVec.ofNat 32 i.val).signExtend 64 = enc i.val) ∧
    (∀ i : Fin (tuning choice).BN, (BitVec.ofNat 32 i.val).signExtend 64 = enc i.val) ∧
    (∀ i : Fin (tuning choice).BCS, (BitVec.ofNat 32 i.val).signExtend 64 = enc i.val) := by
  have hb := NativeControl.tuning_bounds choice
  refine ⟨?_,?_,?_⟩ <;> intro i <;> apply widen32 <;> have hi := i.isLt <;> omega

/-- No reached signed division/remainder has a zero divisor. Zero-size calls
are handled by the empty grid, without strengthening the API shape domain. -/
theorem divisors_positive (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) :
    0 < h.groupCount ∧ 0 < (tuning choice).BN ∧ 0 < (tuning choice).BCS ∧
      0 < (h.K+(tuning choice).BN-1)/(tuning choice).BN := by
  have hn := NativeControl.nonempty h choice metadata _ (FPGridOutput.grid_member _ _ _ hq)
  have ht := NativeControl.tuning_bounds choice
  refine ⟨hn.2.2.1,ht.2.2.1,ht.2.2.2.2.1,?_⟩
  exact Nat.div_pos (by omega) ht.2.2.1

#axiomsClean all_21_widened
#axiomsClean setup_exact
#axiomsClean remaining_exact
#axiomsClean pids_widened
#axiomsClean lanes_widened
#axiomsClean divisors_positive
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeScalar
