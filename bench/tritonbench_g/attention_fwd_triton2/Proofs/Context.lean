import VeriTile.Triton.Math.IntegerIndex
import bench.tritonbench_g.attention_fwd_triton2.Proofs.Entry
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def base (p : Parameters) (s : State α) (st : Strides) :=
  s.pids 1 / p.heads * st.batch + s.pids 1 % p.heads * st.head

def query (M : Algebra α) (p : Parameters) (s : State α) (i : TileIndex [128,128]) : α :=
  M.fp32Load (if s.pids 0*128+i.1.val<p.length ∧ i.2.1.val<96 then
    (s.mem p.Q (base p s p.q+(s.pids 0*128+i.1.val)*p.q.row+i.2.1.val*p.q.dim)).read .real
    else zero M)

def queryScale (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.fp32Load ((s.mem p.QS (s.pids 1*((p.length+127)/128)+s.pids 0)).read .real)

structure Acc (α : Type) where
  maximum : Fin 128 → α
  normalizer : Fin 128 → α
  value : TileIndex [128,128] → α

def initialAcc (M : Algebra α) : Acc α :=
  ⟨fun _ => M.binary (some .fp32) .real .add (zero M) M.negInf,
   fun _ => M.binary (some .fp32) .real .add (zero M) (M.literal (some .fp32) .real 1),
   fun _ => zero M⟩

structure Context (M : Algebra α) (p : Parameters) (initial : State α)
    (row : Nat) (a : Acc α) (current : State α) : Prop where
  memory : current.mem=initial.mem
  aligned : row%64=0
  rows : current.regs .int [128] "offs_m"=some (fun i => ((initial.pids 0*128+i.1.val : Nat) : Int))
  cols : current.regs .int [64] "offs_n"=some (fun i => (i.1.val : Int))
  query : current.regs .real [128,128] "q"=some (Source.query M p initial)
  scale : current.regs .real [] "q_scale"=some (fun _ => queryScale M p initial)
  keyPointer : current.regs .ptr [128,64] "K_ptrs"=some (fun i =>
    (p.K,base p initial p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row))
  valuePointer : current.regs .ptr [64,128] "V_ptrs"=some (fun i =>
    (p.V,base p initial p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim))
  scalePointer : current.regs .ptr [] "K_scale_ptr"=some (fun _ =>
    (p.KS,initial.pids 1*((p.length+63)/64)+row/64))
  outputPointer : current.regs .ptr [128,128] "O_block_ptr"=some (fun i =>
    (p.Out,base p initial p.o+(initial.pids 0*128+i.1.val)*p.o.row+i.2.1.val*p.o.dim))
  maximum : current.regs .real [128] "m_i"=some (fun i => a.maximum i.1)
  normalizer : current.regs .real [128] "l_i"=some (fun i => a.normalizer i.1)
  value : current.regs .real [128,128] "acc"=some a.value

theorem entry_context (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    Context M p s 0 (initialAcc M) (entry M dot p s).val := by
  constructor
  all_goals try simp [entry,query,queryScale,base,zero,initialAcc,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try funext i
  all_goals try simp [query,queryScale,base,zero,initialAcc,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try simp only [IntegerIndex.nat_div,IntegerIndex.nat_mod,IntegerIndex.nat_cdiv,
    ←Int.natCast_ediv,←Int.natCast_emod,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  all_goals try norm_num
  all_goals try norm_cast
  all_goals try simp only [show (1.0 : ℝ)=1 from by norm_num]
  all_goals try simp [Value,Int.toNat_add,Int.toNat_mul,Int.natCast_nonneg]
  all_goals try congr 2
  all_goals try norm_cast
  all_goals try simp [Value,Int.toNat_add,Int.toNat_mul,Int.natCast_nonneg]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
