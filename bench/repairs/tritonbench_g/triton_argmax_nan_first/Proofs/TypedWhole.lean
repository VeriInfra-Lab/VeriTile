import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedLoop

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r'=r ∧ o=off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons,if_pos hi,ih]
      congr 1
      funext r' o
      simp [hi,State.write]
    · simp only [List.foldl_cons,if_neg hi,ih]
      congr 1
      funext r' o
      simp [hi]

def written (s : State Word) (O : Region .int) (M K BM row inner : Nat)
    (current : Fin BM → Word × Nat) : State Word :=
  (TileShape.allIndices [BM]).foldl (fun t i =>
    if row+i.1.val<M then t.write (Region.cast O) ((row+i.1.val)*K+inner)
      (.mk .int (Int.ofNat (current i.1).2)) else t) s

theorem tail_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner : Nat) (index64 : Bool) (current : Fin BM → Word × Nat)
    (ready : Ready BM row inner current s) :
    ∃ t, run (backend rest) ((TypedSource.dimension X O M N K BM BN base index64).surfaceBody.drop 8) s=some t ∧
      t.mem=(written s O M K BM row inner current).mem := by
  obtain ⟨hk,hm,hv,hi⟩ := ready
  simp [TypedSource.dimension,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    backend,State.setReg,hk,hm,hv,hi,ofFloat,toFloat,bop,numeric,numericLt,natLt,ComputeDType.eraseDType,
    Option.bind,store,Region.cast,written,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals simp only [fold_memory]
  all_goals rfl

theorem written_memory (s initial : State Word) (same : s.mem=initial.mem) (O : Region .int)
    (M K BM row inner : Nat) (current : Fin BM → Word × Nat) :
    (written s O M K BM row inner current).mem=(written initial O M K BM row inner current).mem := by
  simp only [written,fold_memory,same]

/-- Successful execution of the entire shared readable quotation. Every
loop tile executes; only final active rows store int64 indices. -/
theorem source_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0<BN) :
    ∃ t, FP.Structural.exec (backend rest) (TypedSource.dimension X O M N K BM BN base index64) s=some t ∧
      t.mem=(written s O M K BM (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K)
        (vectorLoop s X M N K BM BN (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) positive 0
          (fun _ => (0xff800000,0)))).mem := by
  obtain ⟨a,ha,ma,ra⟩ := prefix_run rest s X O M N K BM BN base index64
  obtain ⟨b,hb,mb,rb⟩ := range_run rest s a ma X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) 0 index64 positive _ ra
  obtain ⟨t,ht,mt⟩ := tail_run rest b X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) index64 _ rb
  refine ⟨t,?_,mt.trans (written_memory b s mb _ _ _ _ _ _ _)⟩
  let code := (TypedSource.dimension X O M N K BM BN base index64).surfaceBody
  change run (backend rest) code s=some t
  rw [←List.take_append_drop 7 code,run_append,ha,Option.bind_some]
  have splitCode : code.drop 7=[.forRange "start_n" 0 N BN (body X O M N K BM BN base index64)] ++ code.drop 8 := rfl
  rw [splitCode,run_append]
  simp only [run,step,hb,Option.bind_some]
  exact ht

#axiomsClean tail_run
#axiomsClean source_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
