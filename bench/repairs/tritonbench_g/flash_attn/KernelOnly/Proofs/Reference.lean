import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Body

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 32000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def key (M : Algebra α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [p.DIM,64]) : α :=
  M.fp32Load (if (kptr p s row).inBounds (TileShape.indexToList _ i) [1] then
    (s.mem "K" (base p s p.stride_k_bs p.stride_k_head+
      i.1.val*p.stride_k_dim+(row+i.2.1.val)*p.stride_k_seqlen)).read .real else zero M)

def value (M : Algebra α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [64,p.DIM]) : α :=
  M.fp32Load (if (vptr p s row).inBounds (TileShape.indexToList _ i) [0] then
    (s.mem "V" (base p s p.stride_v_bs p.stride_v_head+
      (row+i.1.val)*p.stride_v_seqlen+i.2.1.val*p.stride_v_dim)).read .real else zero M)

def scores (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α)
    (row : Nat) (i : TileIndex [128,64]) : α :=
  M.binary (some .fp32) .real .add
    (if row+i.2.1.val<p.SEQLEN then
      if p.IS_CAUSAL then
        if row+i.2.1.val≤s.pids 0*128+i.1.val then zero M else M.negInf
      else zero M
     else M.negInf)
    (dot.apply (some .fp32) .real .real .real {} (batch:=[])
      (fun j=>M.cast (some .fp32) .fp16 .real (query M p s scale j))
      (key M p s row) i)

def maximum (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α)
    (row : Nat) (a : Acc p α) (i : Fin 128) : α :=
  M.binary (some .fp32) .real .max (a.maximum i)
    (M.reduceMax (some .fp32) ⟨1,by simp⟩ Bool.false (scores M dot p s scale row) (i,()))

def probability (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α)
    (row : Nat) (a : Acc p α) (i : TileIndex [128,64]) : α :=
  M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub
    (scores M dot p s scale row i) (maximum M dot p s scale row a i.1))

def rescale (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α)
    (row : Nat) (a : Acc p α) (i : Fin 128) : α :=
  M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub
    (a.maximum i) (maximum M dot p s scale row a i))

def contribution (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α)
    (row : Nat) (a : Acc p α) (i : TileIndex [128,p.DIM]) : α :=
  dot.apply (some .fp32) .real .real .real {} (batch:=[])
    (fun j=>M.cast (some .fp32) .fp16 .real (M.cast (some .fp32) .real .fp16
      (probability M dot p s scale row a j)))
    (value M p s row) i

/-- Independent initial-array recurrence, preserving every source cast and
ordered scalar operation. It contains no source interpreter or result selector. -/
def update (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α)
    (row : Nat) (a : Acc p α) : Acc p α :=
  ⟨maximum M dot p s scale row a,
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.normalizer i) (rescale M dot p s scale row a i))
     (M.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false (probability M dot p s scale row a) (i,())),
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.output i)
       (M.binary (some .fp32) .real .add
         (M.binary (some .fp32) .real .mul (a.normalizer i.1) (zero M))
         (rescale M dot p s scale row a i.1)))
     (contribution M dot p s scale row a i)⟩

@[simp] theorem lane2Eta {m n : Nat} (i : TileIndex [m,n]) : (i.1,i.2.1,PUnit.unit)=i := by
  rcases i with ⟨i,j,u⟩;cases u;rfl
@[simp] theorem lane1Eta {m : Nat} (i : TileIndex [m]) : (i.1,PUnit.unit)=i := by
  rcases i with ⟨i,u⟩;cases u;rfl

theorem body_context (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial scale row a current) :
    Context M p initial scale (row+64) (update M dot p initial scale row a)
      (bodyExec M dot p scale initial current row a ctx).val := by
  rcases p with ⟨qbs,qh,qr,qd,kbs,kh,kr,kd,vbs,vh,vr,vd,obs,oh,orr,od,bs,heads,len,dim,causal⟩
  cases causal
  all_goals constructor
  all_goals try simp [bodyExec,ctx.memory,ctx.startM,ctx.head,ctx.rows,ctx.columns,ctx.outputOffset,
    ctx.query,ctx.keyPointer,ctx.valuePointer,ctx.maximum,ctx.normalizer,ctx.output,
    update,maximum,rescale,probability,scores,contribution,key,value,zero,base,kptr,vptr,
    FloatDType.toTileDType,ofFloat,toFloat,FP.Structural.bop,Broadcast.leftIndex,
    Broadcast.rightIndex,Function.comp_def,BlockPtr.advance,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try funext i
  all_goals try simp [update,maximum,rescale,probability,scores,contribution,key,value,zero,base,kptr,vptr,
    FloatDType.toTileDType,ofFloat,toFloat,FP.Structural.bop,Broadcast.leftIndex,
    Broadcast.rightIndex,Function.comp_def,BlockPtr.advance,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try delta scores maximum rescale probability contribution key value zero
  all_goals try simp [base,kptr,vptr,FloatDType.toTileDType,ofFloat,toFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try rw [lane1Eta i]
  all_goals try rfl

#axiomsClean body_context
#specNonCircular update avoiding [run,FP.Structural.exec,bodyExec]
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
