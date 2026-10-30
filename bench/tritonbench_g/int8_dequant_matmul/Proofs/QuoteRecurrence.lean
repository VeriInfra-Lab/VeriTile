import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteOperands
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteExecution

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
open IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem source_update (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (m : Memory F H) (divisor : F)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (pid z t : Nat) (valid : pid<Geometry.count (geometry s c))
    (hz : z<c.S) (ht : t<steps s c) (old : TileIndex [c.BM,c.BN]→Int)
    (i : TileIndex [c.BM,c.BN]) :
    let base:=(preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    q.encode (update q s c base t old i)=
      (selected.plan (row s c pid i.1%s.M) (col s c pid i.2.1%s.N) z t).eval q.ops
        (operandA s c b m (row s c pid i.1%s.M) t z)
        (operandB s c b m (col s c pid i.2.1%s.N) t z) (q.encode (old i)) := by
  dsimp only
  let initial:=initialState q b m divisor pid z
  let base:=(preludeExecution (floating q) q s c bias initial).val
  have va : initial.pids 0<Geometry.count (geometry s c) := by simpa [initial,initialState] using valid
  have pm:=prelude_pm (floating q) q s c bias initial va
  have pn:=prelude_pn (floating q) q s c bias initial va
  have ha:=fun l=>source_a q s c bias b m divisor pid z t valid hz ht i.1 l
  have hb:=fun l=>source_b q s c bias b m divisor pid z t valid hz ht i.2.1 l
  simp only [update,dotAdd,q.roundtrip]
  change (q.plans c.BK
    (((register base .int [] "pid_m" PUnit.unit).toNat*c.BM+i.1.val)%s.M)
    (((register base .int [] "pid_n" PUnit.unit).toNat*c.BN+i.2.1.val)%s.N)
    z t).eval q.ops _ _ (q.encode (old i))=_
  rw [pm,pn,plans]
  simp only [Int.toNat_natCast,selectedPlan_same]
  congr 1
  · exact funext ha
  · exact funext hb

#axiomsClean source_update

theorem accumulation_from (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (m : Memory F H) (divisor : F)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (pid z : Nat) (valid : pid<Geometry.count (geometry s c)) (hz : z<c.S)
    (t : Nat) (bound : t ≤ steps s c) (old : TileIndex [c.BM,c.BN]→Int)
    (h : ∀ i,q.encode (old i)=Int8DequantMatmul.accumulation q.ops s c b selected m
      (row s c pid i.1%s.M) (col s c pid i.2.1%s.N) z t) :
    let base:=(preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    ∀ i,q.encode (foldRange (update q s c base) t (steps s c) 1 old i)=
      Int8DequantMatmul.accumulation q.ops s c b selected m
        (row s c pid i.1%s.M) (col s c pid i.2.1%s.N) z (steps s c) := by
  dsimp only
  by_cases active : t<steps s c
  · let base:=(preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    have next : ∀ i,q.encode (update q s c base t old i)=
        Int8DequantMatmul.accumulation q.ops s c b selected m
          (row s c pid i.1%s.M) (col s c pid i.2.1%s.N) z (t+1) := by
      intro i
      rw [source_update q s c bias b m divisor selected plans pid z t valid hz active old i,h i]
      rfl
    have later:=accumulation_from q s c bias b m divisor selected plans pid z valid hz
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
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (m : Memory F H) (divisor : F)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (pid z : Nat) (valid : pid<Geometry.count (geometry s c)) (hz : z<c.S)
    (i : TileIndex [c.BM,c.BN]) :
    let base:=(preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    q.encode (accumulation q s c base i)=Int8DequantMatmul.accumulation q.ops s c b selected m
      (row s c pid i.1%s.M) (col s c pid i.2.1%s.N) z (steps s c) := by
  apply accumulation_from q s c bias b m divisor selected plans pid z valid hz 0 (Nat.zero_le _) (fun _=>0) _ i
  intro j
  change q.encode 0=q.ops.izero
  rw [←q.zero,q.roundtrip]

#axiomsClean accumulation_from
#axiomsClean source_accumulation
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
