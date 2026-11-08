import bench.tritonbench_g.attn_fwd_causal.Proofs.Body
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def key (M : Algebra α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [128,64]) : α :=
  M.fp32Load (if i.2.1.val<p.length-row ∧ i.1.val<128 then
    (s.mem p.K (base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row)).read .real
    else zero M)

def keyScale (M : Algebra α) (p : Parameters) (s : State α) (row : Nat) : α :=
  M.fp32Load ((s.mem p.KS (s.pids 1*((p.length+63)/64)+row/64)).read .real)

def value (M : Algebra α) (p : Parameters) (s : State α) (row : Nat)
    (i : TileIndex [64,128]) : α :=
  M.cast (some .fp32) .real .fp16
    (M.fp32Load (if i.1.val<p.length-row ∧ i.2.1.val<128 then
      (s.mem p.V (base p s p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim)).read .real
      else zero M))

def scores (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (i : TileIndex [128,64]) : α :=
  if i.2.1.val<p.length-row ∧ row+i.2.1.val≤s.pids 0*128+i.1.val then
    M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.cast (some .fp32) .real .real
          (dot.apply (some .fp32) .real .real .real {} (batch:=[])
            (query (α:=α) M p s) (key (α:=α) M p s row) i))
        (queryScale M p s)) (keyScale M p s row)
  else M.negInf

def maximum (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc α) (i : Fin 128) : α :=
  M.binary (some .fp32) .real .max (a.maximum i)
    (M.reduceMax (some .fp32) ⟨1,by decide⟩ Bool.false (scores M dot p s row) (i,()))

def probability (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc α) (i : TileIndex [128,64]) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (scores M dot p s row i) (maximum M dot p s row a i.1))

def rescale (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc α) (i : Fin 128) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (a.maximum i) (maximum M dot p s row a i))

def contribution (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc α) (i : TileIndex [128,128]) : α :=
  M.cast (some .fp32) .fp16 .real
    (dot.apply (some .fp32) .fp16 .fp16 .fp16 {outDType:=some .fp16} (batch:=[])
      (fun j => M.cast (some .fp32) .real .fp16 (probability (α:=α) M dot p s row a j))
      (value (α:=α) M p s row) i)

/-- Independent scalar/tile recurrence in source order; no execution or final
memory observation occurs in this definition. All casts remain explicit. -/
def update (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (row : Nat) (a : Acc α) : Acc α :=
  ⟨maximum M dot p s row a,
   fun i => M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.normalizer i) (rescale M dot p s row a i))
     (M.reduceSum (some .fp32) ⟨1,by decide⟩ Bool.false (probability M dot p s row a) (i,())),
   fun i => M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.value i) (rescale M dot p s row a i.1))
     (contribution M dot p s row a i)⟩

private theorem maskOrder {β : Type} (N row col query : Nat) (yes no : β) :
    (if row+col≤query ∧ col<N-row then yes else no)=
      (if col<N-row ∧ row+col≤query then yes else no) := by
  by_cases h : row+col≤query <;> by_cases t : col<N-row <;>
    simp only [h,t,and_true,and_false,true_and,false_and,ite_true,ite_false]

theorem body_context {phase : Bool} (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current (phase:=phase)) :
    Context M p initial (row+64) (update M dot p initial row a)
      (bodyExec M dot p initial current row a ctx).val (phase:=phase) := by
  have row_step : (row+64)/64=row/64+1 := by omega
  have h_length:=ctx.length
  have h_startM:=ctx.startM
  have h_firstLo:=ctx.firstLo
  have h_firstHi:=ctx.firstHi
  have h_rows:=ctx.rows
  have h_cols:=ctx.cols
  have h_query:=ctx.query
  have h_scale:=ctx.scale
  have h_keyPointer:=ctx.keyPointer
  have h_valuePointer:=ctx.valuePointer
  have h_scalePointer:=ctx.scalePointer
  have h_maximum:=ctx.maximum
  have h_normalizer:=ctx.normalizer
  have h_value:=ctx.value
  cases phase <;> simp [workName] at h_length h_startM h_firstLo h_firstHi h_rows h_cols h_query h_scale h_keyPointer h_valuePointer h_scalePointer h_maximum h_normalizer h_value <;> constructor
  all_goals simp [bodyExec,workName,h_length,ctx.callerStartM,ctx.callerRows,ctx.callerCols,ctx.callerQuery,ctx.callerScale,ctx.memory,h_startM,h_firstLo,h_firstHi,ctx.keyBase,ctx.valueBase,ctx.scaleBase,ctx.aligned,h_rows,h_cols,h_query,h_scale,
    h_keyPointer,h_valuePointer,h_scalePointer,ctx.outputPointer,h_maximum,h_normalizer,h_value,
    update,maximum,rescale,probability,scores,contribution,key,keyScale,value,zero,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,row_step]
  all_goals try funext i
  all_goals simp [update,maximum,rescale,probability,scores,contribution,key,keyScale,value,zero,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,row_step]
  all_goals try delta scores maximum rescale probability contribution key keyScale value zero
  all_goals try simp [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals norm_num
  all_goals try omega
  all_goals try rfl
  all_goals simp [show p.k.row=128 by rfl,show p.v.row=128 by rfl]
  all_goals first
  | omega
  | (simp only [maskOrder] <;> rfl)

#specNonCircular update avoiding [run,FP.Structural.exec,bodyExec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source
