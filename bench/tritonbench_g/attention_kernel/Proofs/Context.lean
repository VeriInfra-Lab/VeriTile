import bench.tritonbench_g.attention_kernel.Proofs.Entry
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) := M.literal (some .fp32) .real 0

def base (p : Parameters) (s : State α) (st : Strides) :=
  s.pids 1 / p.heads * st.batch + s.pids 1 % p.heads * st.head

def qptr (p : Parameters) (s : State α) : BlockPtr :=
  ⟨p.Q,base p s p.q,[p.length,p.dim],[p.bm,p.dim],[p.q.row,p.q.dim],[s.pids 0*p.bm,0]⟩
def kptr (p : Parameters) (s : State α) (row : Nat) : BlockPtr :=
  ⟨p.K,base p s p.k,[p.dim,p.length],[p.dim,p.bn],[p.k.dim,p.k.row],[0,row]⟩
def vptr (p : Parameters) (s : State α) (row : Nat) : BlockPtr :=
  ⟨p.V,base p s p.v,[p.length,p.dim],[p.bn,p.dim],[p.v.row,p.v.dim],[row,0]⟩
def optr (p : Parameters) (s : State α) : BlockPtr :=
  ⟨p.Out,base p s p.o,[p.length,p.dim],[p.bm,p.dim],[p.o.row,p.o.dim],[s.pids 0*p.bm,0]⟩

def query (M : Algebra α) (p : Parameters) (s : State α) (i : TileIndex [p.bm,p.dim]) : α :=
  M.cast (some .fp32) .real p.storage.float
    (M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) p.storage.float .real
        (if (qptr p s).inBounds (TileShape.indexToList _ i) [0] then
          toFloat p.storage.float ((s.mem p.Q
            (base p s p.q+(s.pids 0*p.bm+i.1.val)*p.q.row+i.2.1.val*p.q.dim)).read p.storage.tile)
         else M.literal (some .fp32) p.storage.float 0))
      (M.binary (some .fp32) .real .mul
        (M.literal (some .fp32) .real p.scale) (M.literal (some .fp32) .real 1.44269504)))

def biasStatic (M : Algebra α) (p : Parameters) (s : State α) (i : TileIndex [p.bm,p.bn]) : α :=
  if s.pids 0*p.bm+i.1.val<p.length then
    toFloat p.storage.float ((s.mem p.Bias
      (base p s p.bias+(s.pids 0*p.bm+i.1.val)*p.bias.row+(i.2.1.val%64+64))).read p.storage.tile)
  else M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 0)

structure Acc (p : Parameters) (α : Type) where
  maximum : Fin p.bm → α
  normalizer : Fin p.bm → α
  value : TileIndex [p.bm,p.dim] → α

def initialAcc (M : Algebra α) (p : Parameters) : Acc p α :=
  ⟨fun _=>M.binary (some .fp32) .real .add (zero M) M.negInf,
   fun _=>zero M,fun _=>zero M⟩

structure Context (M : Algebra α) (p : Parameters) (initial : State α)
    (row : Nat) (a : Acc p α) (current : State α) : Prop where
  memory : current.mem=initial.mem
  aligned : row%p.bn=0
  startM : current.regs .nat [] "start_m"=some (fun _=>initial.pids 0)
  rows : current.regs .nat [p.bm] "b_ptr_offsets_m"=some (fun i=>i.1.val)
  biasOffset : current.regs .nat [] "b_offset"=some (fun _=>base p initial p.bias)
  outputOffset : current.regs .nat [] "o_offset"=some (fun _=>base p initial p.o)
  biasStatic : current.regs p.storage.tile [p.bm,p.bn] "b1"=
    some (fun i=>ofFloat p.storage.float (Source.biasStatic M p initial i))
  query : current.regs p.storage.tile [p.bm,p.dim] "q"=
    some (fun i=>ofFloat p.storage.float (Source.query M p initial i))
  keyPointer : current.regs .blockPtr [p.dim,p.bn] "K_block_ptr"=some (fun _=>kptr p initial row)
  valuePointer : current.regs .blockPtr [p.bn,p.dim] "V_block_ptr"=some (fun _=>vptr p initial row)
  maximum : current.regs .real [p.bm] "m_i"=some (fun i=>a.maximum i.1)
  normalizer : current.regs .real [p.bm] "l_i"=some (fun i=>a.normalizer i.1)
  value : current.regs .real [p.bm,p.dim] "acc"=some a.value

theorem entry_context (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) :
    Context M p s 0 (initialAcc M p) (entry M dot p s).val := by
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage
  all_goals constructor
  all_goals try simp [entry,entryAs,query,biasStatic,base,zero,initialAcc,qptr,kptr,vptr,
    TypedSource.Storage.float,TypedSource.Storage.tile,ofFloat,toFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try funext i
  all_goals try simp [query,biasStatic,base,zero,initialAcc,qptr,kptr,vptr,
    TypedSource.Storage.float,TypedSource.Storage.tile,ofFloat,toFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try norm_num
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
