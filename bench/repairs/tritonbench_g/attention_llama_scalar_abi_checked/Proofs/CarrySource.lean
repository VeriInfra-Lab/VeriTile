import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Output

/-! Constructive execution of the actual carry-first candidate. Source.Term,
Carries, Context, all input views, coordinates and frames are shared with the
original; only the normalizer addition order changes in the source graph. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Source
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def kernel (c : Config) : ComputeKernel :=
  (if c.fp8 then TypedSource.carryFirstBytes else TypedSource.carryFirstHalf)
    c.Q c.K c.V c.Out c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk
    c.svz c.svh c.svk c.svn c.soz c.soh c.som c.son c.NH c.H c.N c.SP 64 64 c.D c.causal

def parts (c : Config) : Triton.Source.DynamicRange :=
  (Triton.Source.dynamicRange? (kernel c).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry (c : Config) (s : State (Term α)) (scale : Term α) :
    {t : State (Term α) // run algebra (parts c).before (bindScale s scale)=some t} := by
  cases hf : c.fp8 <;> cases hc : c.causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,hf,hc,TypedSource.carryFirstHalf,TypedSource.carryFirstBytes,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,bindScale]
    rfl

def advance (c : Config) (s : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) : Carries α c.D :=
  let qk := scores c s scale n
  let m := fun r=>Term.maximum (.max 64 (qk r)) (a.maximum r)
  let prev := fun r=>Term.mul (a.denominator r) (.exp (.sub (a.maximum r) (m r)))
  let prob := fun r k=>Term.exp (.sub (qk r k) (m r))
  let denom := fun r=>Term.add (prev r) (.sum 64 (prob r))
  let recip := fun r=>Term.div (.literal 1) (denom r)
  let hp := fun r k=>Term.half (.mul (prob r k) (recip r))
  ⟨m,denom,fun i=>.add
    (.mul (a.accumulator i) (.mul (prev i.1) (recip i.1)))
    (.dot 64 (hp i.1) (fun k=>loaded c s c.V c.svz c.svh c.svk c.svn n (k,i.2.1,())))⟩

def accumulated (c : Config) (s : State (Term α)) (scale : Term α) : Nat → Carries α c.D
  | 0=>initial c.D
  | n+1=>advance c s scale n (accumulated c s scale n)

 theorem entry_context (c : Config) (s : State (Term α)) (scale : Term α) :
    Context c s scale 0 (initial c.D) (entry c s scale).val := by
  rcases c with ⟨Q,K,V,Out,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svk,svn,soz,soh,som,son,NH,H,N,SP,D,causal,fp8⟩
  cases fp8 <;> cases causal
  all_goals constructor <;> simp [entry,initial,query,pointers,rows,base,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]
  all_goals funext i; simp [query,pointers,rows,base,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]
  all_goals norm_num
  all_goals rfl

theorem before_eq (c : Config) : (parts c).before = (Source.parts c).before := by
  cases hf : c.fp8 <;> cases hc : c.causal <;>
    simp [parts,kernel,Source.parts,Source.kernel,hf,hc,
      TypedSource.carryFirstHalf,TypedSource.carryFirstBytes,TypedSource.half,TypedSource.bytes,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD]

theorem after_eq (c : Config) : (parts c).after = (Source.parts c).after := by
  cases hf : c.fp8 <;> cases hc : c.causal <;>
    simp [parts,kernel,Source.parts,Source.kernel,hf,hc,
      TypedSource.carryFirstHalf,TypedSource.carryFirstBytes,TypedSource.half,TypedSource.bytes,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD]

def finish (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) :
    {u : State (Term α) // run algebra (parts c).after t=some u} :=
  ⟨(Source.finish c s t scale n a h).val,
    by rw [after_eq]; exact (Source.finish c s t scale n a h).property⟩

theorem finish_memory (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) :
    (finish c s t scale n a h).val.mem=(stored c s a).mem :=
  Source.finish_memory c s t scale n a h

#axiomsClean entry
#axiomsClean entry_context
#axiomsClean finish
#axiomsClean finish_memory
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
