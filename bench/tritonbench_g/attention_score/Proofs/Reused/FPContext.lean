import bench.tritonbench_g.attention_score.Proofs.Reused.FPEntry

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

/-- Exact natural address projection of the explicit signed32 descriptor cast.
The public launch bounds justify its simplification; unrestricted symbolic
execution keeps wrapping and negative-to-address projection visible. -/
def keyStart (B : Nat) (s : State α) : Nat :=
  (if 2*((((s.pids 0:Int)*(B:Int))%4294967296).toNat%4294967296)<4294967296 then
    max (((s.pids 0:Int)*(B:Int))%4294967296) 0%4294967296 else
    max (((s.pids 0:Int)*(B:Int))%4294967296) 0%4294967296-4294967296).toNat

def key (M : Algebra α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (i : TileIndex [D,B]) : α :=
  M.fp32Load (if p.evenN || decide (keyStart B s+i.2.1.val<p.keyContext) then
    (s.mem p.K (baseK p s+i.1.val*p.keyDim+(keyStart B s+i.2.1.val)*p.keyRow)).read .real
    else zero M)

def scale (M : Algebra α) (p : RealSource.Parameters) : α :=
  M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real p.scale)
    (M.literal (some .fp32) .real 1.4426950408889634)

/-- The complete carried state consists of actual loaded key values, pointers,
scale, output register and address registers needed after the loop. -/
structure Context (M : Algebra α) (B D : Nat) (p : RealSource.Parameters)
    (initial : State α) (row : Nat) (acc : Fin B → α) (current : State α) : Prop where
  memory : current.mem=initial.mem
  startN : current.regs .int [] "start_n"=some (fun _ => (initial.pids 0:Int))
  batch : current.regs .int [] "off_z"=some (fun _ => ((initial.pids 1 / p.heads : Nat):Int))
  head : current.regs .int [] "off_h"=some (fun _ => ((initial.pids 1 % p.heads : Nat):Int))
  query : current.regs .blockPtr [B,D] "Q_block_ptr"=some (fun _ => queryPointer B D p initial row)
  normalizer : current.regs .ptr [B] "m_ptrs"=some (fun i => (p.M,baseM p initial+row+i.1.val))
  key : current.regs .real [D,B] "k"=some (key M B D p initial)
  scale : current.regs .real [] "qk_scale"=some (fun _ => scale M p)
  output : current.regs .real [B] "o"=some (fun i => acc i.1)

private theorem natural_div (x y : Nat) :
    (if (x:Int)<0 then -(-(x:Int)/(y:Int)) else (x:Int)/(y:Int)) = ((x/y:Nat):Int) := by
  simp [not_lt.mpr (Int.natCast_nonneg x),Int.natCast_ediv]

private theorem natural_mod (x y : Nat) :
    (x:Int)-(if (x:Int)<0 then -((y:Int)*(-(x:Int)/(y:Int)))
      else (y:Int)*((x:Int)/(y:Int))) = ((x%y:Nat):Int) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte]
  rw [←Int.emod_def,←Int.natCast_emod]

theorem entry_context (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α) :
    Context M B D p s 0 (fun _ => zero M) (FPEntry.executes M dot B D p s).val := by
  have divided : (p.heads:Int)*((s.pids 1:Int)/(p.heads:Int)) ≤ (s.pids 1:Int) := by
    exact_mod_cast Nat.mul_div_le (s.pids 1) p.heads
  by_cases hn : p.evenN = Bool.true
  all_goals
    constructor
    all_goals simp [FPEntry.executes,hn,queryPointer,baseQ,baseK,baseM,key,scale,zero,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
    all_goals funext i
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      FPContext.key, baseK, hn, zero]
    all_goals simp only [Value] at *
    all_goals try simp only [natural_div,natural_mod]
    all_goals simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,
      ←Int.natCast_ediv,←Int.natCast_emod]
    all_goals simp [keyStart,Value,not_lt.mpr (Int.natCast_nonneg (s.pids 1)),
      not_lt.mpr divided]
    all_goals try simp only [←Int.emod_def,←Int.natCast_emod,←Int.natCast_ediv,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPContext
