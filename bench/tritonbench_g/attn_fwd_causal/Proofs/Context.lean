import bench.tritonbench_g.attn_fwd_causal.Proofs.Entry
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source
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
  M.fp32Load (if s.pids 0*128+i.1.val<p.length ∧ i.2.1.val<128 then
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
    (row : Nat) (a : Acc α) (current : State α) (phase : Bool := Bool.false) : Prop where
  memory : current.mem=initial.mem
  startM : current.regs .nat [] (workName phase "start_m")=some (fun _=>initial.pids 0)
  firstLo : current.regs .nat [] (workName phase "lo")=some (fun _=>if phase then initial.pids 0*128 else 0)
  firstHi : current.regs .nat [] (workName phase "hi")=some (fun _=>if phase then stop p initial else initial.pids 0*128)
  keyBase : current.regs .ptr [128,64] "K_ptrs"=some (fun i=>
    (p.K,base p initial p.k+i.1.val*p.k.dim+i.2.1.val*p.k.row))
  valueBase : current.regs .ptr [64,128] "V_ptrs"=some (fun i=>
    (p.V,base p initial p.v+i.1.val*p.v.row+i.2.1.val*p.v.dim))
  scaleBase : current.regs .ptr [] "K_scale_ptr"=some (fun _=>
    (p.KS,initial.pids 1*((p.length+63)/64)))
  aligned : row%64=0
  rows : current.regs .nat [128] (workName phase "offs_m")=some (fun i => initial.pids 0*128+i.1.val)
  cols : current.regs .nat [64] (workName phase "offs_n")=some (fun i => i.1.val)
  query : current.regs .real [128,128] (workName phase "q")=some (Source.query M p initial)
  scale : current.regs .real [] (workName phase "q_scale")=some (fun _ => queryScale M p initial)
  keyPointer : current.regs .ptr [128,64] (workName phase "K_ptrs")=some (fun i =>
    (p.K,base p initial p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row))
  valuePointer : current.regs .ptr [64,128] (workName phase "V_ptrs")=some (fun i =>
    (p.V,base p initial p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim))
  scalePointer : current.regs .ptr [] (workName phase "K_scale_ptr")=some (fun _ =>
    (p.KS,initial.pids 1*((p.length+63)/64)+row/64))
  outputPointer : current.regs .ptr [128,128] "O_block_ptr"=some (fun i =>
    (p.Out,base p initial p.o+(initial.pids 0*128+i.1.val)*p.o.row+i.2.1.val*p.o.dim))
  maximum : current.regs .real [128] (workName phase "m_i")=some (fun i => a.maximum i.1)
  normalizer : current.regs .real [128] (workName phase "l_i")=some (fun i => a.normalizer i.1)
  value : current.regs .real [128,128] (workName phase "acc")=some a.value
  length : current.regs .nat [] (workName phase "N_CTX")=some (fun _=>p.length)
  callerStartM : current.regs .nat [] "start_m"=some (fun _=>initial.pids 0)
  callerRows : current.regs .nat [128] "offs_m"=some (fun i=>initial.pids 0*128+i.1.val)
  callerCols : current.regs .nat [64] "offs_n"=some (fun i=>i.1.val)
  callerQuery : current.regs .real [128,128] "q"=some (Source.query M p initial)
  callerScale : current.regs .real [] "q_scale"=some (fun _=>queryScale M p initial)

theorem entry_context (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    Context M p s 0 (initialAcc M) (entry M dot p s).val := by
  constructor
  all_goals simp [entry,query,queryScale,base,zero,initialAcc,
    workName,Int.natCast_ediv,Int.natCast_emod,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try funext i
  all_goals simp [query,queryScale,base,zero,initialAcc,
    workName,Int.natCast_ediv,Int.natCast_emod,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals norm_num
  all_goals try rfl
  all_goals simp only [show p.k.dim=1 by rfl,Nat.mul_one]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source
