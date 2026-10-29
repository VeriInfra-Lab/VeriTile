import bench.repairs.tritonbench_g.softmax_optimize.Proofs.FPComplete

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.FPExecution
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def RowFrame (Y : RegionName) (N pid : Nat) (s t : State α) : Prop :=
  ∀ r i,r≠Y ∨ i<pid*N ∨ (pid+1)*N≤ i → t.mem r i=s.mem r i

theorem output_row (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (N B pid : Nat) (tail : Bool) (s : State α)
    (ready : Ready [] N B pid s) (index : Nat)
    (fullBound : tail=Bool.false → index+B≤ N) :
    ∃ t, run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
      (if tail then TypedSource.tailOutput Y X N B else TypedSource.fullOutput Y X N B)
      (s.setReg "start_n" .int [] (fun _ => (index:Int)))=some t ∧
      Ready [] N B pid t ∧ RowFrame Y N pid s t := by
  rcases ready with ⟨hp,hc,hprev,hm,hz⟩
  obtain ⟨m,hm⟩ := Option.isSome_iff_exists.mp hm
  obtain ⟨z,hz⟩ := Option.isSome_iff_exists.mp hz
  cases tail <;>
    simp (config := {maxSteps := 300000}) [TypedSource.fullOutput,TypedSource.tailOutput,
      TypedSource.loopBody,TypedSource.body,kernel,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
      VeriTile.Triton.FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision,withEq,
      ComputeDType.eraseDType,numeric,numericEq,numericLt,numericLe,bop,ofFloat,toFloat,
      State.setReg,integralFloorDiv,integralMod,Ready,Region.cast,RowFrame,hp,hc,hprev,hm,hz]
  · intro r i outside
    apply State.scatter_frame
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    rcases outside with hr|lo|hi
    · exact Or.inl hr
    · right;intro lane _;omega
    · right
      intro lane _
      have hb := lane.1.isLt
      have bound := fullBound rfl
      have hi' : pid*N+N≤ i := by simpa [Nat.add_mul] using hi
      omega
  · intro r i outside
    apply State.masked_scatter_frame
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    rcases outside with hr|lo|hi
    · exact Or.inl hr
    · right;intro lane _ active;omega
    · right
      intro lane _ active
      have hi' : pid*N+N≤ i := by simpa [Nat.add_mul] using hi
      have activeNat : index+lane.1.val<N := Int.ofNat_lt.mp active
      omega


theorem previous_nat (N B : Nat) (hN : 0<N) (hB : 0<B) :
    previous N B=(Config.previous N B:Int) :=
  VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource.prevMultipleInt_eq_nat N B hN hB

/-- The full range visits exactly the source tile multiples. -/
theorem block_range (A : Algebra α) (body : List ComputeStmt)
    (P : State α → Prop) (B total : Nat) (hB : 0<B)
    (body_step : ∀ index s,index<total → P s → ∃ t,
      run A body (s.setReg "start_n" .int [] (fun _ => (index*B:Int)))=some t ∧ P t)
    (index : Nat) (s : State α) (hs : P s) :
    ∃ t,rangeInt A "start_n" (index*B) (total*B) B body s=some t ∧ P t := by
  by_cases hc : index<total
  · have hlt : (index*B:Int)<total*B := by exact_mod_cast Nat.mul_lt_mul_of_pos_right hc hB
    obtain ⟨mid,he,hi⟩ := body_step index s hc hs
    obtain ⟨t,ht,hI⟩ := block_range A body P B total hB body_step (index+1) mid hi
    refine ⟨t,?_,hI⟩
    rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_pos hlt,he]
    simp only [bind,Option.bind]
    simpa only [Nat.cast_add,Nat.cast_one,add_mul,one_mul] using ht
  · have hge : ¬(index*B:Int)<total*B := by
      exact not_lt.mpr (by exact_mod_cast Nat.mul_le_mul_right B (by omega : total≤ index))
    exact ⟨s,by rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_neg hge],hs⟩
termination_by total-index

theorem outputs_framed (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (N B pid : Nat) (hN : 0<N) (hB : 0<B)
    (s : State α) (ready : Ready [] N B pid s) :
    ∃ t,run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
      (loopStatements Y X N B Bool.false) s=some t ∧
      Ready [] N B pid t ∧ RowFrame Y N pid s t := by
  let P := fun t => Ready [] N B pid t ∧ RowFrame Y N pid s t
  have body (index : Nat) (t : State α) (hi : index<Config.fullTiles N B) (ht : P t) :
      ∃ u,run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
        (TypedSource.fullOutput Y X N B)
        (t.setReg "start_n" .int [] (fun _ => (index*B:Int)))=some u ∧ P u := by
    have bound : index*B+B≤ N := by
      have step := Nat.mul_le_mul_right B (by omega : index+1≤ Config.fullTiles N B)
      have tail := Config.tail_start_lt (B:=B) hN
      nlinarith
    obtain ⟨u,he,hr,hf⟩ := output_row A eq Y X N B pid Bool.false t ht.1 (index*B) (fun _ => bound)
    refine ⟨u,?_,hr,fun r i h => (hf r i h).trans (ht.2 r i h)⟩
    simpa only [Int.natCast_mul] using he
  obtain ⟨mid,hfull,hmid⟩ := block_range (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
    (TypedSource.fullOutput Y X N B) P B (Config.fullTiles N B) hB body 0 s
      ⟨ready,fun _ _ _ => rfl⟩
  have hp := previous_nat N B hN hB
  have hpn : previous N B=(Config.fullTiles N B:Int)*B := by
    rw [hp,Config.previous_eq_full_tiles hN hB,Int.natCast_mul]
  simp only [Nat.cast_zero,zero_mul,←hpn] at hfull
  obtain ⟨t,htail,hr,hf⟩ := output_row A eq Y X N B pid Bool.true mid hmid.1
    (Config.previous N B) (by simp)
  have bounds := Config.final_tail_nonempty_once hN hB
  have start : previous N B<(N:Int) := by rw [hp];exact_mod_cast bounds.1
  have stop : ¬previous N B+(B:Int)<N := by rw [hp];omega
  have htailrange : rangeInt (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
      "start_n" (previous N B) N B (TypedSource.tailOutput Y X N B) mid=some t := by
    rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_pos start,hp]
    simp only [↓reduceIte] at htail
    rw [htail]
    simp only [bind,Option.bind]
    rw [rangeInt,if_neg (Nat.ne_of_gt hB)]
    rw [hp] at stop
    exact if_neg stop
  refine ⟨t,?_,hr,fun r i h => (hf r i h).trans (hmid.2 r i h)⟩
  simp only [loopStatements,Bool.false_eq_true,↓reduceIte,run,step,evalOp_unfold,bind,Option.bind,
    ready.2.2.1,hmid.1.2.2.1,hmid.1.2.1,Option.bind_some,hfull,htailrange]

/-- Successful actual source execution with a raw-cell frame outside its row. -/
theorem complete_row (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (M N B : Nat) (hN : 0<N) (hB : 0<B) (s : State α) :
    ∃ t, FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
      (kernel Y X M N B) s=some t ∧ RowFrame Y N (s.pids 0) s t := by
  obtain ⟨a,ha,hra,hma⟩ := prelude A eq Y X N B hB s
  obtain ⟨b,hb,hrb,hmb⟩ := inputs A eq Y X N B (s.pids 0) hB a hra
  obtain ⟨c,hc,hrc,hmc⟩ := reduction A eq Y X N B (s.pids 0) hB b hrb
  obtain ⟨t,ht,hrt,hmt⟩ := outputs_framed A eq Y X N B (s.pids 0) hN hB c hrc
  refine ⟨t,?_,fun r i h => (hmt r i h).trans ?_⟩
  · change run _ (kernel Y X M N B).surfaceBody s=some t
    rw [TypedSource.source_body]
    change run _ (((TypedSource.body Y X N B).take 5 ++ loopStatements Y X N B Bool.true) ++
      ((TypedSource.body Y X N B).drop 7).take 4 ++ loopStatements Y X N B Bool.false) s=some t
    simp only [run_append,ha,hb,hc,ht,Option.bind_some]
  · rw [hmc,hmb,hma]

#axiomsClean complete_row

#axiomsClean output_row
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.FPExecution
