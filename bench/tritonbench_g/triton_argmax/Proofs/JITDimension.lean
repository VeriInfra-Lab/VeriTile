import bench.tritonbench_g.triton_argmax.Proofs.JITDimensionStep
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITDimension
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
open Typed
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem prefix_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) :
    ∃ t, run (backend rest) ((JITSource.dimension X O BM BN index64).surfaceBody.take 7) (arguments s M N K base)=some t ∧
      t.mem=s.mem ∧ Ready M N K BM (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K)
        (fun _ => (0xff800000,0)) t := by
  have nonnegative : ¬ ((s.pids 0 + base : Nat) : Int) < 0 := by omega
  have remainder := Nat.mod_add_div (s.pids 0 + base) K
  cases index64 <;>
    simp [JITSource.dimension,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      backend,arguments,Ready,ofFloat,toFloat,bop,numeric,numericLt,natLt,IntegerExpr.sameShape,integralFloorDiv,integralMod,ComputeDType.eraseDType,Option.bind, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, -Int.natCast_mod, ← Int.natCast_ediv, ← Int.natCast_emod]
  all_goals constructor <;> funext i
  all_goals simp [bop, Broadcast.leftIndex, Broadcast.rightIndex,
    -Nat.cast_add, -Nat.cast_mul, -Int.natCast_mod, ← Int.natCast_add, ← Int.natCast_mul,
    ← Int.natCast_ediv, ← Int.natCast_emod, Int.emod_def, nonnegative]
  all_goals
    have equation := congrArg (fun n : Nat => (n : Int)) remainder
    simp only [Nat.cast_add] at equation
    exact sub_eq_iff_eq_add.mpr equation.symm

theorem range_run (rest : Algebra Word) (initial s : State Word) (same : s.mem=initial.mem)
    (X : RegionName) (O : Region .int) (M N K BM BN base row inner cur : Nat)
    (index64 : Bool) (positive : 0<BN) (current : Fin BM → Word × Nat)
    (ready : Ready M N K BM row inner current s) :
    ∃ t, rangeInt (backend rest) "start_n" (cur : Int) (N : Int) BN (body X O BM BN index64) s=some t ∧
      t.mem=initial.mem ∧ Ready M N K BM row inner
        (vectorLoop initial X M N K BM BN row inner positive cur current) t := by
  by_cases active : cur<N
  · obtain ⟨middle,runStep,memStep,readyStep⟩ :=
      body_run rest s X O M N K BM BN base row inner cur index64 positive current ready
    rw [next_memory initial s same] at readyStep
    obtain ⟨t,runRest,memRest,readyRest⟩ := range_run rest initial middle (memStep.trans same)
      X O M N K BM BN base row inner (cur+BN) index64 positive _ readyStep
    refine ⟨t,?_,memRest,?_⟩
    · rw [rangeInt, if_neg (Nat.ne_of_gt positive), if_pos (show (cur : Int) < (N : Int) by omega),runStep]
      simp only [← Int.natCast_add]
      exact runRest
    · rw [vectorLoop,dif_pos active]
      exact readyRest
  · refine ⟨s,by rw [rangeInt, if_neg (Nat.ne_of_gt positive), if_neg (show ¬(cur : Int) < (N : Int) by omega)],same,?_⟩
    rw [vectorLoop,dif_neg active]
    exact ready
termination_by N-cur
decreasing_by omega


theorem tail_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner : Nat) (index64 : Bool) (current : Fin BM → Word × Nat)
    (ready : Ready M N K BM row inner current s) :
    ∃ t, run (backend rest) ((JITSource.dimension X O BM BN index64).surfaceBody.drop 8) s=some t ∧
      t.mem=(written s O M K BM row inner current).mem := by
  obtain ⟨hM,hN,hK,hk,hm,hv,hi⟩ := ready
  simp [JITSource.dimension,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    backend,hM,hN,hK,hk,hm,hv,hi,ofFloat,toFloat,bop,numeric,numericLt,natLt,ComputeDType.eraseDType,
    Option.bind,store,Region.cast,written,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals simp only [fold_memory]
  all_goals rfl



/-- Execute the complete quotation with its runtime scalar ABI. -/
theorem source_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0<BN) :
    ∃ t, FP.Structural.exec (backend rest) (JITSource.dimension X O BM BN index64)
      (arguments s M N K base)=some t ∧
      t.mem=(written s O M K BM (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K)
        (vectorLoop s X M N K BM BN (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) positive 0
          (fun _ => (0xff800000,0)))).mem := by
  obtain ⟨a,ha,ma,ra⟩ := prefix_run rest s X O M N K BM BN base index64
  obtain ⟨b,hb,mb,rb⟩ := range_run rest s a ma X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) 0 index64 positive _ ra
  obtain ⟨t,ht,mt⟩ := tail_run rest b X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) index64 _ rb
  refine ⟨t,?_,mt.trans (written_memory b s mb _ _ _ _ _ _ _)⟩
  let code := (JITSource.dimension X O BM BN index64).surfaceBody
  change run (backend rest) code (arguments s M N K base)=some t
  rw [←List.take_append_drop 7 code,run_append,ha,Option.bind_some]
  have splitCode : code.drop 7 =
      [.forRangeInt "start_n" (.constInt 0) (.ref .int [] "N") (.constNat BN)
        (body X O BM BN index64)] ++ code.drop 8 := rfl
  rw [splitCode,run_append]
  have bound := ra.2.1
  simp only [run,step,evalOp_unfold,bound,Option.bind_some]
  simp only [bind, pure, Option.bind, ← Int.natCast_zero, hb]
  exact ht

theorem source_reference (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0<BN) (nonempty : 0<N) (innerPositive : 0<K) :
    ∃ t, FP.Structural.exec (backend rest) (JITSource.dimension X O BM BN index64) (arguments s M N K base)=some t ∧
      (∀ i : Fin BM,((s.pids 0+base)/K)*BM+i.val<M →
        t.mem (Region.cast O) ((((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K)=
          .mk .int (Int.ofNat ((reference N nonempty (fun j : Fin N =>
            inputRow s X N K (((s.pids 0+base)/K)*BM+i.val) ((s.pids 0+base)%K) j.val)).val))) ∧
      (∀ region offset,(region≠Region.cast O ∨
        ∀ i : Fin BM,((s.pids 0+base)/K)*BM+i.val<M →
          offset≠((((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K)) →
        t.mem region offset=s.mem region offset) := by
  obtain ⟨t,run,mem⟩ := source_run rest s X O M N K BM BN base index64 positive
  refine ⟨t,run,?_,?_⟩
  · intro i active
    rw [mem]
    unfold written
    rw [State.masked_scatter_readback (Region.cast O)
      (fun j : TileIndex [BM] => (((s.pids 0+base)/K)*BM+j.1.val)*K+(s.pids 0+base)%K) _
      (fun j : TileIndex [BM] => ((s.pids 0+base)/K)*BM+j.1.val<M) s (i,PUnit.unit) active]
    · rw [vector_reference s X M N K BM BN _ _ positive nonempty i active]
    · rintro ⟨i,_⟩ ⟨j,_⟩ same
      have sameVal : i.val=j.val := by nlinarith
      have sameFin : i=j := Fin.ext sameVal
      cases sameFin
      rfl
  · intro region offset outside
    rw [mem]
    unfold written
    apply State.masked_scatter_frame
    rcases outside with region | offset
    · exact Or.inl region
    · exact Or.inr (fun i _ hi => offset i.1 hi)


#axiomsClean source_reference
#axiomsClean source_run

#axiomsClean tail_run
#axiomsClean prefix_run
#axiomsClean range_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITDimension
