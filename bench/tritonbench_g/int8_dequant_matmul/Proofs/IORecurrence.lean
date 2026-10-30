import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOOperands
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteRecurrence

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IORecurrence
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def wrappedCell (s : Shape) (c : Config) (pid : Nat)
    (valid : pid<Geometry.count (geometry s c)) (i : TileIndex [c.BM,c.BN]) : MatrixCell s :=
  (⟨row s c pid i.1%s.M,Nat.mod_lt _ (IOGeometry.rows_pos s c pid valid)⟩,
   ⟨col s c pid i.2.1%s.N,Nat.mod_lt _ (IOGeometry.columns_pos s c pid valid)⟩)

theorem source_update (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals initial)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (hz : initial.pids 1<c.S)
    (t : Nat) (ht : t<steps s c) (old : TileIndex [c.BM,c.BN]→Int)
    (i : TileIndex [c.BM,c.BN]) :
    let base:=(preludeExecution (floating q) q s c hasBias initial).val
    let cell:=wrappedCell s c (initial.pids 0) valid i
    q.encode (update q s c base t old i)=
      (selected.plan cell.1.val cell.2.val (initial.pids 1) t).eval q.ops
        (Mathematical.left s c (IOLayout.data q.ops s vals) cell.1 t (initial.pids 1))
        (Mathematical.right s c (IOLayout.data q.ops s vals) cell.2 t (initial.pids 1))
        (q.encode (old i)) := by
  dsimp only
  let base:=(preludeExecution (floating q) q s c hasBias initial).val
  have pm:=prelude_pm (floating q) q s c hasBias initial valid
  have pn:=prelude_pn (floating q) q s c hasBias initial valid
  have ha:=fun l=>IOOperands.source_a q s c hasBias vals initial cells valid hz t ht i.1 l
  have hb:=fun l=>IOOperands.source_b q s c hasBias vals initial cells valid hz t ht i.2.1 l
  have pids : base.pids=initial.pids := by simp [base,preludeExecution]
  simp only [update,dotAdd,q.roundtrip]
  change (q.plans c.BK
    (((register base .int [] "pid_m" PUnit.unit).toNat*c.BM+i.1.val)%s.M)
    (((register base .int [] "pid_n" PUnit.unit).toNat*c.BN+i.2.1.val)%s.N)
    (base.pids 1) t).eval q.ops _ _ (q.encode (old i))=_
  rw [pm,pn,pids,plans]
  simp only [Int.toNat_natCast,selectedPlan_same]
  congr 1
  · exact funext ha
  · exact funext hb

theorem accumulation_from (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals initial)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (hz : initial.pids 1<c.S)
    (t : Nat) (bound : t ≤ steps s c) (old : TileIndex [c.BM,c.BN]→Int)
    (h : ∀ i,q.encode (old i)=Mathematical.accumulator q.ops s c selected
      (IOLayout.data q.ops s vals) (wrappedCell s c (initial.pids 0) valid i)
        ⟨initial.pids 1,hz⟩ t) :
    let base:=(preludeExecution (floating q) q s c hasBias initial).val
    ∀ i,q.encode (foldRange (update q s c base) t (steps s c) 1 old i)=
      Mathematical.accumulator q.ops s c selected (IOLayout.data q.ops s vals)
        (wrappedCell s c (initial.pids 0) valid i) ⟨initial.pids 1,hz⟩ (steps s c) := by
  dsimp only
  by_cases active : t<steps s c
  · let base:=(preludeExecution (floating q) q s c hasBias initial).val
    have next : ∀ i,q.encode (update q s c base t old i)=
        Mathematical.accumulator q.ops s c selected (IOLayout.data q.ops s vals)
          (wrappedCell s c (initial.pids 0) valid i) ⟨initial.pids 1,hz⟩ (t+1) := by
      intro i
      rw [source_update q s c hasBias vals initial cells selected plans valid hz t active old i,h i]
      rfl
    have later:=accumulation_from q s c hasBias vals initial cells selected plans valid hz
      (t+1) (by omega) (update q s c base t old) next
    intro i
    rw [foldRange]
    simpa only [show (1:Nat)≠0 by decide,active,if_false,if_true] using later i
  · have equal : t=steps s c := by omega
    subst t
    intro i
    rw [foldRange]
    simpa only [show (1:Nat)≠0 by decide,lt_self_iff_false,if_false] using h i
termination_by steps s c-t
decreasing_by omega

theorem source_accumulation (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals initial)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (valid : initial.pids 0<Geometry.count (geometry s c)) (hz : initial.pids 1<c.S)
    (i : TileIndex [c.BM,c.BN]) :
    let base:=(preludeExecution (floating q) q s c hasBias initial).val
    q.encode (Quote.accumulation q s c base i)=
      Mathematical.accumulator q.ops s c selected (IOLayout.data q.ops s vals)
        (wrappedCell s c (initial.pids 0) valid i) ⟨initial.pids 1,hz⟩ (steps s c) := by
  apply accumulation_from q s c hasBias vals initial cells selected plans valid hz
    0 (Nat.zero_le _) (fun _=>0) _ i
  intro j
  change q.encode 0=q.ops.izero
  rw [←q.zero,q.roundtrip]

#axiomsClean source_update
#axiomsClean accumulation_from
#axiomsClean source_accumulation
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IORecurrence
