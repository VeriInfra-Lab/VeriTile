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
  startN : current.regs .nat [] "start_n"=some (fun _ => initial.pids 0)
  batch : current.regs .nat [] "off_z"=some (fun _ => initial.pids 1 / p.heads)
  head : current.regs .nat [] "off_h"=some (fun _ => initial.pids 1 % p.heads)
  query : current.regs .blockPtr [B,D] "Q_block_ptr"=some (fun _ => queryPointer B D p initial row)
  normalizer : current.regs .ptr [B] "m_ptrs"=some (fun i => (p.M,baseM p initial+row+i.1.val))
  key : current.regs .real [D,B] "k"=some (key M B D p initial)
  scale : current.regs .real [] "qk_scale"=some (fun _ => scale M p)
  output : current.regs .real [B] "o"=some (fun i => acc i.1)

theorem entry_context (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α) :
    Context M B D p s 0 (fun _ => zero M) (FPEntry.executes M dot B D p s).val := by
  by_cases hn : p.evenN = Bool.true
  all_goals
    constructor
    all_goals simp [FPEntry.executes,hn,queryPointer,baseQ,baseK,baseM,key,scale,zero,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
    all_goals funext i
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      FPContext.key, baseK, hn, zero]
    all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPContext
