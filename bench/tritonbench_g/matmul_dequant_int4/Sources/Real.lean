import bench.tritonbench_g.matmul_dequant_int4.Sources.Memory
import bench.tritonbench_g.matmul_dequant_int4.Sources.Packing
import bench.tritonbench_g.matmul_dequant_int4.Sources.RealSum
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
open RealSum
set_option maxHeartbeats 6000000
noncomputable section

theorem nibble_digit (word : Word) (slot : Nat) : nibble word slot=digit word slot := by
  have hn : ((word >>> (slot%8*4)) &&& (15 : Word)).toNat≤15 := by
    simpa only [BitVec.toNat_and,BitVec.toNat_ofNat,Nat.reduceMod] using
      (Nat.and_le_right (n := (word >>> (slot%8*4)).toNat) (m := 15))
  unfold nibble
  rw [BitVec.toInt_eq_toNat_of_lt (by omega)]
  simp [digit,BitVec.toNat_and,BitVec.toNat_ushiftRight,mask15,Nat.shiftRight_eq_div_pow,Nat.mul_comm]

theorem standalone_real_value (s : Shape) (initial : State ℝ) (k n : Nat) :
    dequantValue s realDequant initial ⟨k,n⟩=dequantizedReal s initial k n := by
  simp [dequantValue,realDequant,difference,nibble_digit,dequantizedReal,logicalQ,logicalZ,
    logicalScale,qAddress,sAddress,zAddress]
theorem standalone_whole_real (s : Shape) (g : Grid s.k s.n) (order : List Work)
    (complete : g.Complete order) (initial : State ℝ) (k n : Nat) (hk : k<s.k) (hn : n<s.n) :
    (dequantRun s realDequant order initial).values .dequant (k*s.n+n)=dequantizedReal s initial k n := by
  rw [dequant_complete s realDequant g order complete initial k n hk hn,standalone_real_value]

theorem fused_operand_real (s : Shape) (t : Tile) (ng : Bool) (h : FusedDomain s t ng)
    (fusion : Nat → Nat → Fusion) (initial : State ℝ) (col block lane : Nat)
    (hb : block<blocksK s t) (hl : lane<t.bk) :
    fusedOperandB s t realFused ng fusion initial col block lane=
      dequantizedReal s initial (block*t.bk+lane) col := by
  have hg := fused_group_correct s t ng h block lane hb hl
  unfold fusedOperandB
  rw [hg]
  cases mode : fusion (block*t.bk+lane) col <;>
    simp [fusedDequant,realFused,dequantizedReal,logicalQ,logicalZ,logicalScale,
      nibble_digit,digit,shift_period t h.tile,Int.cast_sub,mode] <;> ring

theorem foldl_congr_mem (xs : List γ) (f g : α → γ → α)
    (h : ∀ x∈xs,∀ seed,f seed x=g seed x) (seed : α) : xs.foldl f seed=xs.foldl g seed := by
  induction xs generalizing seed with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons,h x (by simp)]
    exact ih (fun y hy => h y (by simp [hy])) _

def realProduct (s : Shape) (initial : State ℝ) (w : Work) (k : Nat) :=
  logicalA s initial w.row k*dequantizedReal s initial k w.col

theorem fused_instruction_real (s : Shape) (t : Tile) (ng : Bool) (h : FusedDomain s t ng)
    (fusion : Nat → Nat → Fusion) (initial : State ℝ) (w : Work) (hr : w.row<s.m)
    (block chunk : Nat) (hb : block<blocksK s t) (hc : chunk<t.bk/16) (seed : ℝ) :
    realFused.mma16
      (fun i => if w.row<s.m then initial.values .a (aAddress s w.row (block*t.bk+chunk*16+i)) else 0)
      (fun i => fusedOperandB s t realFused ng fusion initial w.col block (chunk*16+i)) seed=
      seed+sumReal (fun i => realProduct s initial w (block*t.bk+(chunk*16+i))) 16 := by
  change seed+sumReal (fun i =>
    (if w.row<s.m then initial.values .a (aAddress s w.row (block*t.bk+chunk*16+i%16)) else 0)*
    fusedOperandB s t realFused ng fusion initial w.col block (chunk*16+i%16)) 16=_
  congr 1
  apply sum_congr
  intro i hi
  have hl : chunk*16+i<t.bk := by
    have bound := rowMajor_bound (t.bk/16) 16 chunk i hc hi
    simpa [Nat.div_mul_cancel h.tile.bk16] using bound
  rw [Nat.mod_eq_of_lt hi,if_pos hr,fused_operand_real s t ng h fusion initial w.col block (chunk*16+i) hb hl]
  simp [realProduct,logicalA,aAddress,Nat.add_assoc]

theorem fused_block_real (s : Shape) (t : Tile) (ng : Bool) (h : FusedDomain s t ng)
    (fusion : Nat → Nat → Fusion) (initial : State ℝ) (w : Work) (hr : w.row<s.m)
    (block : Nat) (hb : block<blocksK s t) (seed : ℝ) :
    fusedBlock s t realFused ng fusion initial w block seed=
      seed+sumReal (fun lane => realProduct s initial w (block*t.bk+lane)) t.bk := by
  unfold fusedBlock
  simp only [show realFused.zeroInput=0 from rfl]
  rw [foldl_congr_mem _ _ (fun acc chunk => acc+sumReal
    (fun i => realProduct s initial w (block*t.bk+(chunk*16+i))) 16)
    (fun chunk hc acc => fused_instruction_real s t ng h fusion initial w hr block chunk hb
      (List.mem_range.mp hc) acc)]
  rw [range_fold,sum_chunks (fun lane => realProduct s initial w (block*t.bk+lane)) 16 (t.bk/16),Nat.div_mul_cancel h.tile.bk16]

theorem fused_value_real (s : Shape) (t : Tile) (ng : Bool) (h : FusedDomain s t ng)
    (fusion : Nat → Nat → Fusion) (initial : State ℝ) (w : Work) (hr : w.row<s.m) :
    fusedValue s t realFused ng fusion initial w=mathematical s initial w.row w.col := by
  unfold fusedValue
  change (List.range (blocksK s t)).foldl
    (fun acc block => fusedBlock s t realFused ng fusion initial w block acc) 0=_
  rw [foldl_congr_mem _ _ (fun acc block => acc+sumReal
    (fun lane => realProduct s initial w (block*t.bk+lane)) t.bk)
    (fun block hb acc => fused_block_real s t ng h fusion initial w hr block (List.mem_range.mp hb) acc)]
  rw [range_fold,sum_chunks (realProduct s initial w) t.bk (blocksK s t),zero_add]
  have hk : blocksK s t*t.bk=s.k := by
    rw [blocksK,ceil_of_dvd _ _ h.tile.bkpos h.fullK,Nat.div_mul_cancel h.fullK]
  rw [hk]
  rfl

theorem fused_whole_real (s : Shape) (t : Tile) (ng : Bool) (h : FusedDomain s t ng)
    (fusion : Nat → Nat → Fusion) (order : List Work)
    (complete : (fusedGrid s t h.shape h.tile).Complete order) (initial : State ℝ)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) :
    (fusedRun s t realFused ng fusion order initial).values .output (r*s.n+n)=mathematical s initial r n := by
  rw [fused_complete s t realFused ng fusion (fusedGrid s t h.shape h.tile) order complete initial r n hr hn]
  exact fused_value_real s t ng h fusion initial ⟨r,n⟩ hr

#axiomsClean nibble_digit
#axiomsClean standalone_whole_real
#axiomsClean fused_operand_real
#axiomsClean fused_instruction_real
#axiomsClean fused_block_real
#axiomsClean fused_value_real
#axiomsClean fused_whole_real
end
end VeriTile.Bench.Repairs.MatmulDequantInt4
