import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.FPRow

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.FPIndexBridge
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def parts (input output : RegionName) (T BT : Nat) : Source.IntRange :=
  (Source.intRange? (repairedKernel .fp32 .fp32 input output T BT).surfaceBody "i_t").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem before (input output : RegionName) (T BT : Nat) :
    (parts input output T BT).before=(FPRow.parts .fp32 .fp32 input output T BT).before := rfl

/-- Only the registers read by the next body are observed. Pointer temporaries
and the signed/natural loop-index register need not be identical. -/
def Rel {α : Type} (a b : State α) : Prop :=
  a.mem=b.mem ∧ a.undef=b.undef ∧ a.pids=b.pids ∧ a.numPids=b.numPids ∧
    a.regs .nat [] "i_bh"=b.regs .nat [] "i_bh" ∧ a.regs .real [] "b_z"=b.regs .real [] "b_z"

@[simp] theorem int_address (p T j BT lane : Nat) :
    ((p:Int)*T+((j:Int)*BT+lane)).toNat=p*T+(j*BT+lane) := by omega

@[simp] theorem int_mask (j BT lane T : Nat) :
    decide ((j:Int)*BT+lane<(T:Int))=decide (j*BT+lane<T) := by
  norm_cast

@[simp] theorem int_lt (j BT lane T : Nat) :
    ((j:Int)*BT+lane<(T:Int)) ↔ j*BT+lane<T := by norm_cast

@[simp] theorem masked_undef {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).undef=s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split <;> rfl

@[simp] theorem masked_numPids {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).numPids=s.numPids := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split <;> rfl

theorem masked_mem {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s t : State α) (hm : s.mem=t.mem) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).mem=
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) t).mem := by
  induction l generalizing s t with
  | nil => exact hm
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases h : active i
    · simp only [if_pos h,State.write,hm]
    · simpa only [if_neg h] using hm

theorem body {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T BT pid index : Nat) (a b : State α)
    (related : Rel a b) (ready : FPRow.Ready pid b) :
    ∃ x y,
      run (engine ops.algebra) (parts input output T BT).body
        (a.setReg "i_t" .int [] (fun _ => (index:Int)))=some x ∧
      run (engine ops.algebra) (FPRow.parts .fp32 .fp32 input output T BT).body
        (b.setReg "i_t" .nat [] (fun _ => index))=some y ∧
      Rel x y ∧ FPRow.Ready pid y := by
  obtain ⟨hm,hu,hp,hn,hi,hc⟩ := related
  obtain ⟨hb,hz⟩ := ready
  obtain ⟨carry,hz⟩ := Option.isSome_iff_exists.mp hz
  have ha := hi.trans hb
  have hca := hc.trans hz
  simp (config := {maxSteps := 300000}) [parts,repairedKernel,FPRow.parts,kernel,TypedSource.original,
    ComputeKernel.surfaceBody,Source.intRange?,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,FP.Structural.store,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,FPRow.Ops.algebra,ComputeDType.eraseDType,numeric,numericLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FPRow.Ready,Rel,ha,hca,hb,hz,hm,hu,hp,hn,
    TileShape.indexToList]
  apply masked_mem
  rfl

theorem loop {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T BT pid cur stop : Nat) (a b : State α)
    (related : Rel a b) (ready : FPRow.Ready pid b) :
    ∃ x y,
      rangeInt (engine ops.algebra) "i_t" cur stop 1 (parts input output T BT).body a=some x ∧
      range (engine ops.algebra) "i_t" cur stop 1 (FPRow.parts .fp32 .fp32 input output T BT).body b=some y ∧
      Rel x y ∧ FPRow.Ready pid y := by
  by_cases hlt : cur<stop
  · obtain ⟨midA,midB,ha,hb,hr,hready⟩ := body ops input output T BT pid cur a b related ready
    obtain ⟨x,y,hx,hy,hrel,hready⟩ := loop ops input output T BT pid (cur+1) stop midA midB hr hready
    refine ⟨x,y,?_,?_,hrel,hready⟩
    · rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hlt),ha]
      simpa using hx
    · rw [range,if_neg (by decide : ¬(1:Nat)=0),if_pos hlt,hb]
      exact hy
  · refine ⟨a,b,?_,?_,related,ready⟩
    · rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_neg (by exact_mod_cast hlt)]
    · rw [range,if_neg (by decide : ¬(1:Nat)=0),if_neg hlt]
termination_by stop-cur
 decreasing_by omega

theorem split_source (input output : RegionName) (T BT : Nat) :
    (repairedKernel .fp32 .fp32 input output T BT).surfaceBody =
      (parts input output T BT).before ++
        [.forRangeInt "i_t" (parts input output T BT).start
          (parts input output T BT).stop (parts input output T BT).stride
          (parts input output T BT).body] := rfl

/-- Complete actual signed/masked-pointer source and old logical source agree
on tagged memory, arbitrary undefined lanes, carry and launch metadata. -/
theorem complete_related {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T BT : Nat) (s : State α) :
    ∃ x y,
      FP.Structural.exec (engine ops.algebra) (repairedKernel .fp32 .fp32 input output T BT) s=some x ∧
      FP.Structural.exec (engine ops.algebra) (kernel .fp32 .fp32 input output T BT) s=some y ∧
      Rel x y := by
  obtain ⟨initial,hpre,hready,_⟩ := FPRow.prelude ops .fp32 .fp32 input output T BT s
  obtain ⟨x,y,hx,hy,hr,_⟩ := loop ops input output T BT (s.pids 0) 0 ((T+BT-1)/BT)
    initial initial ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩ hready
  refine ⟨x,y,?_,?_,hr⟩
  · change run (engine ops.algebra) (repairedKernel .fp32 .fp32 input output T BT).surfaceBody s=some x
    rw [split_source,run_append,before,hpre]
    simp only [Option.bind_some,run]
    have hs : step (engine ops.algebra)
        (.forRangeInt "i_t" (parts input output T BT).start
          (parts input output T BT).stop (parts input output T BT).stride
          (parts input output T BT).body) initial=some x := by
      simp only [step]
      change rangeInt (engine ops.algebra) "i_t" 0 (((T+BT-1)/BT : Nat) : Int) 1
        (parts input output T BT).body initial=some x
      exact hx
    rw [hs]
    rfl
  · change run (engine ops.algebra) (kernel .fp32 .fp32 input output T BT).surfaceBody s=some y
    rw [FPRow.split_source,run_append,hpre]
    simp only [Option.bind_some,run]
    have hs : step (engine ops.algebra)
        (.forRangeDyn "i_t" (FPRow.parts .fp32 .fp32 input output T BT).start
          (FPRow.parts .fp32 .fp32 input output T BT).stop (FPRow.parts .fp32 .fp32 input output T BT).stride
          (FPRow.parts .fp32 .fp32 input output T BT).body) initial=some y := by
      simp only [step]
      change range (engine ops.algebra) "i_t" 0 ((T+BT-1)/BT) 1
        (FPRow.parts .fp32 .fp32 input output T BT).body initial=some y
      exact hy
    rw [hs]
    rfl

theorem complete {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T BT : Nat) (s : State α) :
    ∃ t, FP.Structural.exec (engine ops.algebra) (repairedKernel .fp32 .fp32 input output T BT) s=some t ∧
      FPRow.Frame output (s.pids 0) T s t := by
  obtain ⟨x,y,hx,hy,hr⟩ := complete_related ops input output T BT s
  obtain ⟨z,hz,hf⟩ := FPRow.complete ops .fp32 .fp32 input output T BT s
  have same : z=y := Option.some.inj (hz.symm.trans hy)
  subst z
  exact ⟨x,hx,fun rg off hn => (congrFun (congrFun hr.1 rg) off).trans (hf rg off hn)⟩

#axiomsClean body
#axiomsClean loop
#axiomsClean complete_related
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.FPIndexBridge
