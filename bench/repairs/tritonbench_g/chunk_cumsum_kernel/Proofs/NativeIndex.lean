import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Layout

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.NativeIndex
open Layout
noncomputable section

/-- An ordinary backed FP32 contiguous allocation. The byte span, including
the final byte, fits the actual pointer address space. No shape cap is added. -/
structure Buffer where
  base : Nat
  cells : Nat
  placement : base+4*cells≤2^64

structure Backing (c : Shape) where
  input : Buffer
  output : Buffer
  inputCells : c.B*c.H*c.T≤input.cells
  outputCells : c.B*c.H*c.T≤output.cells

/-- Ordinary CUDA program_id ABI, separate from deferred grid compatibility. -/
structure PID (c : Shape) where
  value : Nat
  inside : value<c.B*c.H
  signed32 : value<2^31

def chunks (T : Nat) (cfg : Config) := (T+cfg.block-1)/cfg.block

theorem block_bound (cfg : Config) : cfg.block≤64 := by cases cfg <;> decide

theorem cells_bound (b : Buffer) : b.cells≤2^62 := by
  have h := b.placement
  norm_num at h ⊢
  omega

theorem chunk_start (T : Nat) (cfg : Config) (q : Nat) (hq : q<chunks T cfg) : q*cfg.block<T := by
  have h := (Nat.lt_div_iff_mul_lt cfg.positive).mp hq
  simp only [chunks] at hq
  have positive := cfg.positive
  omega

theorem chunks_le (T : Nat) (cfg : Config) : chunks T cfg≤T := by
  apply Nat.le_of_not_gt
  intro h
  have start := chunk_start T cfg T h
  have mono := Nat.le_mul_of_pos_right T cfg.positive
  omega

theorem signed_exact (n : Nat) (hn : n<2^63) : (BitVec.ofNat 64 n).toInt=(n:Int) := by
  have hnat : (BitVec.ofNat 64 n).toNat=n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt,hnat]
  rw [hnat]
  norm_num at hn ⊢
  omega

theorem pid_cast (c : Shape) (pid : PID c) :
    BitVec.ofInt 64 (BitVec.ofNat 32 pid.value).toInt=BitVec.ofNat 64 pid.value := by
  have hnat : (BitVec.ofNat 32 pid.value).toNat=pid.value := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by have := pid.signed32; omega)]
  have hint : (BitVec.ofNat 32 pid.value).toInt=(pid.value:Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt,hnat]
    rw [hnat]
    have := pid.signed32
    norm_num at this ⊢
    omega
  rw [hint,BitVec.ofInt_natCast]

theorem loop_bounds (c : Shape) (cfg : Config) (backing : Backing c) (pid : PID c) :
    c.T<2^63 ∧ chunks c.T cfg<2^63 := by
  have cell := cells_bound backing.input
  have count := backing.inputCells
  have timeLe := Nat.le_mul_of_pos_left c.T (Nat.zero_lt_of_lt pid.inside)
  have ht : c.T<2^63 := by norm_num at cell ⊢; omega
  exact ⟨ht,(chunks_le c.T cfg).trans_lt ht⟩

theorem loop_native (c : Shape) (cfg : Config) (backing : Backing c) (pid : PID c)
    (q : Nat) (hq : q<chunks c.T cfg) :
    (BitVec.ofNat 64 (chunks c.T cfg)).toInt=(chunks c.T cfg:Int) ∧
    (BitVec.ofNat 64 q+BitVec.ofNat 64 1).toInt=(q+1:Nat) := by
  have stop := (loop_bounds c cfg backing pid).2
  refine ⟨signed_exact _ stop,?_⟩
  rw [←BitVec.ofNat_add]
  exact signed_exact _ (by omega)

theorem source_bounds (c : Shape) (cfg : Config) (backing : Backing c) (pid : PID c)
    (q : Nat) (hq : q<chunks c.T cfg) (lane : Fin cfg.block) :
    c.T<2^63 ∧ chunks c.T cfg<2^63 ∧ q*cfg.block<c.T ∧
    q*cfg.block+lane.val<2^63 ∧ pid.value*c.T<2^63 ∧
    pid.value*c.T+(q*cfg.block+lane.val)<2^63 := by
  have cell := cells_bound backing.input
  have count := backing.inputCells
  have rowPositive : 0<c.B*c.H := Nat.zero_lt_of_lt pid.inside
  have timeLe := Nat.le_mul_of_pos_left c.T rowPositive
  have start := chunk_start c.T cfg q hq
  have rowNext := Nat.mul_le_mul_right c.T (Nat.succ_le_of_lt pid.inside)
  have block := block_bound cfg
  have laneBound := lane.isLt
  simp only [Nat.succ_mul] at rowNext
  have timeBound : c.T<2^63 := by norm_num at cell ⊢; omega
  refine ⟨timeBound,(chunks_le c.T cfg).trans_lt timeBound,start,?_,?_,?_⟩ <;>
    norm_num at cell ⊢ <;> omega

def timeWord (cfg : Config) (q lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 q*BitVec.ofNat 64 cfg.block+BitVec.ofNat 64 lane

def elementWord (c : Shape) (cfg : Config) (pid : PID c) (q lane : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 pid.value).toInt*BitVec.ofNat 64 c.T+timeWord cfg q lane

/-- Exact signed induction/product/addition and predicate, including inactive
lanes. Their possible pointer wrap is never dereferenced by the masked load
or store; valid lanes have the separately proved ordinary backing bound. -/
theorem source_indices (c : Shape) (cfg : Config) (backing : Backing c) (pid : PID c)
    (q : Nat) (hq : q<chunks c.T cfg) (lane : Fin cfg.block) :
    (BitVec.ofNat 64 q*BitVec.ofNat 64 cfg.block).toInt=(q*cfg.block:Int) ∧
    (timeWord cfg q lane.val).toInt=(q*cfg.block+lane.val:Nat) ∧
    (timeWord cfg q lane.val).slt (BitVec.ofNat 64 c.T)=decide (q*cfg.block+lane.val<c.T) ∧
    (elementWord c cfg pid q lane.val).toInt=(pid.value*c.T+(q*cfg.block+lane.val):Nat) := by
  obtain ⟨ht,_,hs,htime,_,hoff⟩ := source_bounds c cfg backing pid q hq lane
  simp only [timeWord,elementWord,pid_cast,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  refine ⟨signed_exact _ (hs.trans ht),signed_exact _ htime,?_,signed_exact _ hoff⟩
  rw [BitVec.slt_eq_decide,signed_exact _ htime,signed_exact _ ht]
  norm_cast

theorem active_inside (c : Shape) (pid : PID c) (time : Nat) (ht : time<c.T) :
    pid.value*c.T+time<c.B*c.H*c.T := by
  have next := Nat.mul_le_mul_right c.T (Nat.succ_le_of_lt pid.inside)
  simp only [Nat.succ_mul] at next
  omega

theorem pointer_word (b : Buffer) (off : Nat) (inside : off<b.cells) :
    (BitVec.ofNat 64 b.base+BitVec.ofNat 64 off*BitVec.ofNat 64 4).toNat=b.base+off*4 := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  have place := b.placement
  omega

theorem active_pointers (c : Shape) (cfg : Config) (backing : Backing c) (pid : PID c)
    (q : Nat) (lane : Fin cfg.block) (active : q*cfg.block+lane.val<c.T) :
    let off := pid.value*c.T+(q*cfg.block+lane.val)
    (BitVec.ofNat 64 backing.input.base+elementWord c cfg pid q lane.val*BitVec.ofNat 64 4).toNat=
      backing.input.base+off*4 ∧
    (BitVec.ofNat 64 backing.output.base+elementWord c cfg pid q lane.val*BitVec.ofNat 64 4).toNat=
      backing.output.base+off*4 := by
  have inside := active_inside c pid (q*cfg.block+lane.val) active
  simp only [elementWord,timeWord,pid_cast,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  simpa only [←BitVec.ofNat_mul,←BitVec.ofNat_add] using
    And.intro (pointer_word backing.input _ (inside.trans_le backing.inputCells))
      (pointer_word backing.output _ (inside.trans_le backing.outputCells))

/-- Concrete original defect at a legal CUDA x-grid row. -/
theorem original_row_overflow :
    ((BitVec.ofNat 32 33038210)*(BitVec.ofNat 32 65)).toInt=(-2147483646:Int) ∧
    33038210*65=(2147483650:Nat) := by decide

/-- The superseded lower-bound-only cast was not sufficient: an unsigned32
upper constant 2^31 was sign-extended by range lowering. Explicitly casting
that upper bound from its unsigned source type retains the intended value. -/
theorem upper_cast_witness :
    (BitVec.ofInt 64 (BitVec.ofNat 32 (2^31)).toInt).toInt=(-(2^31:Int)) ∧
    (BitVec.ofNat 64 (BitVec.ofNat 32 (2^31)).toNat).toInt=(2^31:Int) := by decide

#axiomsClean loop_native
#axiomsClean source_indices
#axiomsClean active_pointers
#axiomsClean original_row_overflow
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.NativeIndex
