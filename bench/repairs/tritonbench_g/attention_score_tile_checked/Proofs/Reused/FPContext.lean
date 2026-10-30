import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPEntry

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPContext
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPEngine
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def baseQ (p : RealSource.Parameters) (s : State α) :=
  s.pids 1 / p.heads * p.queryBatch + s.pids 1 % p.heads * p.queryHead
def baseK (p : RealSource.Parameters) (s : State α) :=
  s.pids 1 / p.heads * p.keyBatch + (s.pids 1 % p.heads) / (p.heads / p.kvHeads) * p.keyHead
def baseM (p : RealSource.Parameters) (s : State α) := s.pids 1 * p.roundedContext

def queryPointer (B D : Nat) (p : RealSource.Parameters) (s : State α) (row : Nat) : BlockPtr :=
  ⟨p.Q,baseQ p s,[p.queryContext,D],[B,D],[p.queryRow,p.queryDim],[row,0]⟩

def key (M : Algebra α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (i : TileIndex [D,B]) : α :=
  M.fp32Load (if p.evenN || decide (s.pids 0*B+i.2.1.val<p.keyContext) then
    (s.mem p.K (baseK p s+i.1.val*p.keyDim+(s.pids 0*B+i.2.1.val)*p.keyRow)).read .real
    else zero M)

def scale (M : Algebra α) (p : RealSource.Parameters) : α :=
  M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real p.scale)
    (M.literal (some .fp32) .real 1.4426950408889634)

/-- The complete carried state consists of actual loaded key values, pointers,
scale, output register and address registers needed after the loop. -/
structure Context (M : Algebra α) (B D : Nat) (p : RealSource.Parameters)
    (initial : State α) (row : Nat) (acc : Fin B → α) (current : State α) : Prop where
  memory : current.mem=initial.mem
  startN : current.regs .int [] "start_n"=some (fun _ => (initial.pids 0 : Int))
  batch : current.regs .int [] "off_z"=some (fun _ => ((initial.pids 1 / p.heads : Nat) : Int))
  head : current.regs .int [] "off_h"=some (fun _ => ((initial.pids 1 % p.heads : Nat) : Int))
  query : current.regs .blockPtr [B,D] "Q_block_ptr"=some (fun _ => queryPointer B D p initial row)
  normalizer : current.regs .ptr [B] "m_ptrs"=some (fun i => (p.M,baseM p initial+row+i.1.val))
  key : current.regs .real [D,B] "k"=some (key M B D p initial)
  scale : current.regs .real [] "qk_scale"=some (fun _ => scale M p)
  output : current.regs .real [B] "o"=some (fun i => acc i.1)

theorem entry_context (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (keyOffsetBound : s.pids 0*B < 2147483648) :
    Context M B D p s 0 (fun _ => zero M) (FPEntry.executes M dot B D p s).val  := by
  have signedExact : (BitVec.ofInt 32 ((s.pids 0*B : Nat) : Int)).toInt = ((s.pids 0*B : Nat) : Int) :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by norm_num; omega)
  have remainder (a b : Nat) : (a : Int) - ((b*(a/b) : Nat) : Int) = ((a%b : Nat) : Int) := by
    rw [Nat.cast_mul, Int.natCast_ediv, ← Int.emod_def, ← Int.natCast_emod]
  have notBelowProduct (a b : Nat) : ¬ (a : Int) < ((b*(a/b) : Nat) : Int) := by
    have h := Nat.mod_add_div a b
    omega
  have bits : (s.pids 0*B : Int) % 4294967296 = ((s.pids 0*B : Nat) : Int) :=
    Int.emod_eq_of_lt (by positivity) (by exact_mod_cast (show s.pids 0*B < 4294967296 by omega))
  have bitsNat : s.pids 0*B % 4294967296 = s.pids 0*B := Nat.mod_eq_of_lt (by omega)
  have signBit : 2*(s.pids 0*B) < 4294967296 := by omega
  by_cases hn : p.evenN = Bool.true
  all_goals
    constructor
    all_goals simp [FPEntry.executes,hn,signedBits_int_mod,signedExact,queryPointer,baseQ,baseK,baseM,key,scale,zero,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
    all_goals funext i
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      FPContext.key, baseK, hn, zero, ← Int.ofNat_tdiv, ← Int.ofNat_tmod,
      ← Int.natCast_add, ← Int.natCast_mul, -Nat.cast_add, -Nat.cast_mul]
    all_goals try dsimp only
    all_goals try norm_num only [Int.natCast_nonneg, not_lt_of_ge, ↓reduceIte]
    all_goals try simp only [← Int.emod_def, ← Int.natCast_ediv, ← Int.natCast_emod,
      ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
    all_goals simp only [ite_false, ite_true, notBelowProduct, remainder,
      ← Int.natCast_ediv, ← Int.natCast_emod, ← Int.natCast_mul, ← Int.natCast_add,
      Int.toNat_natCast, bits, bitsNat, signBit, ↓reduceIte]
    all_goals try norm_num
    all_goals rw [bits]
    all_goals simp only [Int.toNat_natCast, bitsNat, signBit, ite_true,
      max_eq_left (Int.natCast_nonneg _)]
    all_goals rw [Int.emod_eq_of_lt (Int.natCast_nonneg _) (by
      exact_mod_cast (show s.pids 0*B < 4294967296 by omega))]
    all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPContext
