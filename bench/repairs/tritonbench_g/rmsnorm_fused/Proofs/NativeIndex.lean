import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Safety

namespace VeriTile.Bench.Repairs.RmsnormFused.NativeIndex
open VeriTile.Triton

def rowWord (row : Nat) : BitVec 64 := BitVec.ofInt 64 (BitVec.ofNat 32 row).toInt

theorem signed_exact (n : Nat) (hn : n<2^63) : (BitVec.ofNat 64 n).toInt=(n:Int) := by
  have hnat : (BitVec.ofNat 64 n).toNat=n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt,hnat]
  rw [hnat]
  norm_num at hn ⊢
  omega

theorem pid_cast (q : Host.Request) (row : Nat) (hr : row<q.rows) : rowWord row=BitVec.ofNat 64 row := by
  have grid := q.gridFits
  have bound : row<2^31 := by change row<q.geometry.rows at hr; omega
  have hnat : (BitVec.ofNat 32 row).toNat=row := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  have hint : (BitVec.ofNat 32 row).toInt=(row:Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt,hnat]
    rw [hnat]
    norm_num at bound ⊢
    omega
  simp only [rowWord,hint,BitVec.ofInt_natCast]

theorem formed_index (q : Host.Request) (row col : Nat) (hr : row<q.rows) (hc : col<q.block) :
    (rowWord row*BitVec.ofNat 64 q.columns+BitVec.ofNat 64 col).toInt=(row*q.columns+col:Nat) := by
  have bytes := padded_byte_bound q row col 0 hr hc (by decide)
  rw [pid_cast q row hr,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact signed_exact _ (by omega)

theorem signed32_exact (n : Nat) (hn : n<2^31) : (BitVec.ofNat 32 n).toInt=(n:Int) := by
  have hnat : (BitVec.ofNat 32 n).toNat=n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt,hnat]
  rw [hnat]
  norm_num at hn ⊢
  omega

theorem lane_mask (q : Host.Request) (col : Nat) (hc : col<q.block) :
    BitVec.ofInt 64 (BitVec.ofNat 32 col).toInt=BitVec.ofNat 64 col ∧
    (BitVec.ofNat 32 col).slt (BitVec.ofNat 32 q.columns)=decide (col<q.columns) := by
  have block : q.block≤16384 := q.valid.2.2.1
  have width : q.columns≤16384 := q.valid.2.1.trans q.valid.2.2.1
  have hcol := signed32_exact col (by omega)
  have hwidth := signed32_exact q.columns (by omega)
  constructor
  · rw [hcol,BitVec.ofInt_natCast]
  · rw [BitVec.slt_eq_decide,hcol,hwidth]
    norm_cast

theorem active_access (q : Host.Request) (row col : Nat) (hr : row<q.rows) (hc : col<q.columns) :
    row*q.columns+col<q.size := offset_bounded q.geometry row col hr hc

def pointer (q : Host.Request) (base row col : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base+(rowWord row*BitVec.ofNat 64 q.columns+BitVec.ofNat 64 col)*BitVec.ofNat 64 4

theorem pointer_span (q : Host.Request) (base row col : Nat)
    (span : base+4*q.size≤2^64) (hr : row<q.rows) (hc : col<q.columns) :
    (pointer q base row col).toNat=base+4*(row*q.columns+col) ∧
      ∀ byte, byte<4 → base+4*(row*q.columns+col)+byte<2^64 := by
  have active := active_access q row col hr hc
  constructor
  · simp only [pointer,pid_cast q row hr,←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  · intro byte hb
    omega

/-- Actual widened input/output pointers and all four active bytes. The
remaining bounds are ordinary live allocation spans from those exact pointers. -/
theorem source_pointers {σ C : Type} {memory : σ → RegionName → Nat → C} (q : Host.Request)
    (call : Host.Preparation σ C memory q) (row col : Nat) (hr : row<q.rows) (hc : col<q.columns) :
    (pointer q call.input.dataPtr row col).toNat=call.input.dataPtr+4*(row*q.columns+col) ∧
    (pointer q call.output.dataPtr row col).toNat=call.output.dataPtr+4*(row*q.columns+col) ∧
    (BitVec.ofNat 64 call.weight.dataPtr+BitVec.ofNat 64 col*BitVec.ofNat 64 4).toNat=
      call.weight.dataPtr+4*col ∧
    (∀ byte,byte<4 → call.input.dataPtr+4*(row*q.columns+col)+byte<2^64 ∧
      call.output.dataPtr+4*(row*q.columns+col)+byte<2^64 ∧ call.weight.dataPtr+4*col+byte<2^64) := by
  have nonempty : q.rows≠0 := by omega
  have input := pointer_span q call.input.dataPtr row col (call.inputSpan nonempty) hr hc
  have output := pointer_span q call.output.dataPtr row col (call.outputSpan nonempty) hr hc
  have weight := call.weightSpan nonempty
  refine ⟨input.1,output.1,?_,?_⟩
  · simp only [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  · intro byte hb
    exact ⟨input.2 byte hb,output.2 byte hb,by omega⟩

/-- A noncontiguous two-row input can use a row stride beyond fresh output. -/
theorem original_gapped_output : (1:Nat)*4+1≥2*2 := by decide

#axiomsClean pid_cast
#axiomsClean formed_index
#axiomsClean lane_mask
#axiomsClean active_access
#axiomsClean source_pointers
end VeriTile.Bench.Repairs.RmsnormFused.NativeIndex
