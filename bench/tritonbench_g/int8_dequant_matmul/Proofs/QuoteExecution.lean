import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteComplete
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteModel

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

@[simp] theorem masked_write_regs {α ι : Type} (l : List ι) (s : State α)
    (active : ι→Prop) [DecidablePred active] (offset : ι→Nat)
    (v : State α→ι→Cell α) (region : RegionName) :
    (l.foldl (fun t i=>if active i then t.write region (offset i) (v t i) else t) s).regs=s.regs := by
  induction l generalizing s with
  | nil=>rfl
  | cons i rest ih=>
    rw [List.foldl_cons,ih]
    split <;> rfl

def value (q : IntegerModel I F H) (s : Shape) (c : Config) (bias : Bool)
    [Inhabited (Sum F H)] (base : State (Sum F H)) (divisor : F)
    (i : TileIndex [c.BM,c.BN]) : H :=
  let x:=asF q.ops (register base .real [c.BM,1] "x_factor" (i.1,0,PUnit.unit))
  let w:=asF q.ops (register base .real [1,c.BN] "w_factor" (0,i.2.1,PUnit.unit))
  let v:=q.ops.narrow (q.ops.mul w (q.ops.mul x
    (q.ops.mul (q.ops.toFloat (q.encode (accumulation q s c base i))) divisor)))
  if bias && register base .int [] "pid_z" PUnit.unit==0 then
    q.ops.hadd v (q.ops.narrow (asF q.ops
      (if register base .int [c.BN] "rn" (i.2.1,PUnit.unit)<(s.N:Int) then
        (base.mem "bias" (register base .int [c.BN] "rn" (i.2.1,PUnit.unit)).toNat).read .real
       else .inl q.ops.fzero)))
  else v

theorem epilogue_value (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (base current : State (Sum F H))
    (divisor : F) (hasDivisor : base.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor))
    (ctx : Context s c base (steps s c) current)
    (acc : HasAcc c (accumulation q s c base) current) :
    let final:=(epilogueExecution (floating q) q s c bias base current (.inl divisor) hasDivisor
      (fun x y=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))) rfl ctx acc).val
    register final .fp16 [c.BM,c.BN] "out"=fun i=>.inr (value q s c bias base divisor i) := by
  dsimp only
  by_cases split : c.S=1
  all_goals
    cases bias
    all_goals
      by_cases first : register base .int [] "pid_z" PUnit.unit=0
      all_goals
        simp only [register] at first
        simp [epilogueExecution,split,first,register,masked_write_regs,
          floating,value,asH,asF,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    funext i
    rcases i with ⟨i,j,u⟩
    cases u
    simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,asF,asH]

theorem fullExecution (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (initial : State (Sum F H))
    (divisor : F) (hasDivisor : initial.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor)) :
    ∃ final, execute (floating q) q s.M s.N (source s c bias) initial=some final ∧
      register final .fp16 [c.BM,c.BN] "out"=
        fun i=>.inr (value q s c bias (preludeExecution (floating q) q s c bias initial).val divisor i) := by
  let base:=(preludeExecution (floating q) q s c bias initial).val
  have start:=preludeContext (floating q) q s c bias initial
  obtain ⟨current,hloop,ctx,acc⟩:=rangeExecution (floating q) q s c bias base base 0 (steps s c)
    (Nat.zero_le _) (fun _=>0) start.1 start.2
  have hd : base.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor) := by
    simpa [base,preludeExecution] using hasDivisor
  let final:=epilogueExecution (floating q) q s c bias base current (.inl divisor) hd
    (fun x y=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))) rfl ctx acc
  refine ⟨final.val,?_,epilogue_value q s c bias base current divisor hd ctx acc⟩
  unfold execute
  rw [source_decomposition,run_append,run_append,(preludeExecution (floating q) q s c bias initial).property]
  change ((run (floating q) q s.M s.N [.forRange "k" 0 (steps s c) 1 (body s c bias)] base).bind
      (run (floating q) q s.M s.N (epilogue s c bias)))=some final.val
  simp only [run,step,hloop,Option.bind_some]
  exact final.property

#axiomsClean epilogue_value
#axiomsClean fullExecution

def writeCell {α : Type} (mem : RegionName→Nat→Cell α) (r : RegionName) (o : Nat) (v : Cell α) :=
  fun r' o'=>if r'=r ∧ o'=o then v else mem r' o'

theorem masked_write_memory {α ι : Type} (l : List ι) (s : State α)
    (active : ι→Prop) [DecidablePred active] (offset : ι→Nat)
    (v : (RegionName→Nat→Cell α)→ι→Cell α) (region : RegionName) :
    (l.foldl (fun t i=>if active i then t.write region (offset i) (v t.mem i) else t) s).mem=
      l.foldl (fun mem i=>if active i then writeCell mem region (offset i) (v mem i) else mem) s.mem := by
  induction l generalizing s with
  | nil=>rfl
  | cons i rest ih=>
    rw [List.foldl_cons,List.foldl_cons,ih]
    split <;> rfl

def stores (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (base : State (Sum F H)) (divisor : F) :=
  (TileShape.allIndices [c.BM,c.BN]).foldl (fun mem i=>
    if active s c base i then
      writeCell mem "C" (offset s c base i) (.mk .fp16 (.inr
        (if c.S=1 then value q s c bias base divisor i else
          q.ops.hadd (asH q.ops ((mem "C" (offset s c base i)).read .fp16))
            (value q s c bias base divisor i))))
    else mem) base.mem

theorem fold_memory {α ι : Type} (l : List ι) (f : State α→ι→State α)
    (g : (RegionName→Nat→Cell α)→ι→(RegionName→Nat→Cell α))
    (s : State α) (m : RegionName→Nat→Cell α) (same : s.mem=m)
    (h : ∀ t i,(f t i).mem=g t.mem i) : (l.foldl f s).mem=l.foldl g m := by
  rw [←same]
  exact (List.foldl_hom State.mem (fun t i=>(h t i).symm)).symm

theorem epilogue_memory (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (base current : State (Sum F H))
    (divisor : F) (hasDivisor : base.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor))
    (ctx : Context s c base (steps s c) current)
    (acc : HasAcc c (accumulation q s c base) current) :
    let final:=(epilogueExecution (floating q) q s c bias base current (.inl divisor) hasDivisor
      (fun x y=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))) rfl ctx acc).val
    final.mem=stores q s c bias base divisor := by
  dsimp only
  by_cases split : c.S=1
  all_goals
    cases bias
    all_goals
      by_cases first : register base .int [] "pid_z" PUnit.unit=0
      all_goals
        simp only [register] at first
        simp [epilogueExecution,split,first,register,masked_write_memory,
          floating,value,asH,asF,bop,Broadcast.leftIndex,Broadcast.rightIndex,
          stores,active,offset,ctx.memory]
  all_goals
    apply fold_memory
    · simp [ctx.memory]
    · intro mem i
      rcases i with ⟨i,j,u⟩
      cases u
      simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,asF,asH,State.write,writeCell]
      split <;> rfl

#axiomsClean epilogue_memory

theorem fullExecution_memory (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (initial : State (Sum F H))
    (divisor : F) (hasDivisor : initial.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor)) :
    ∃ final, execute (floating q) q s.M s.N (source s c bias) initial=some final ∧
      (register final .fp16 [c.BM,c.BN] "out"=
        fun i=>.inr (value q s c bias (preludeExecution (floating q) q s c bias initial).val divisor i)) ∧
      final.mem=stores q s c bias (preludeExecution (floating q) q s c bias initial).val divisor := by
  let base:=(preludeExecution (floating q) q s c bias initial).val
  have start:=preludeContext (floating q) q s c bias initial
  obtain ⟨current,hloop,ctx,acc⟩:=rangeExecution (floating q) q s c bias base base 0 (steps s c)
    (Nat.zero_le _) (fun _=>0) start.1 start.2
  have hd : base.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor) := by
    simpa [base,preludeExecution] using hasDivisor
  let final:=epilogueExecution (floating q) q s c bias base current (.inl divisor) hd
    (fun x y=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))) rfl ctx acc
  refine ⟨final.val,?_,epilogue_value q s c bias base current divisor hd ctx acc,
    epilogue_memory q s c bias base current divisor hd ctx acc⟩
  unfold execute
  rw [source_decomposition,run_append,run_append,(preludeExecution (floating q) q s c bias initial).property]
  change ((run (floating q) q s.M s.N [.forRange "k" 0 (steps s c) 1 (body s c bias)] base).bind
      (run (floating q) q s.M s.N (epilogue s c bias)))=some final.val
  simp only [run,step,hloop,Option.bind_some]
  exact final.property

#axiomsClean fullExecution_memory
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
