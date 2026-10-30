import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteRecurrence
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteFactors

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem prelude_pz {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α) :
    register (preludeExecution M q s c bias initial).val .int [] "pid_z"=
      (fun _=>(initial.pids 1:Int)) := by simp [register,preludeExecution]

theorem source_value (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (m : Memory F H) (divisor : F)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (pid z : Nat) (valid : pid<Geometry.count (geometry s c)) (hz : z<c.S)
    (i : TileIndex [c.BM,c.BN]) (hr : row s c pid i.1<s.M) (hc : col s c pid i.2.1<s.N) :
    let base:=(preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    value q s c bias base divisor i=partialValue q.ops s c b selected divisor bias m
      (row s c pid i.1) (col s c pid i.2.1) z := by
  let initial:=initialState q b m divisor pid z
  have valid' : initial.pids 0<Geometry.count (geometry s c) := by simpa [initial,initialState] using valid
  have hx:=factorX q s c bias initial valid'
  have hw:=factorW q s c bias initial valid'
  have rn:=prelude_rn (floating q) q s c bias initial valid'
  have pz:=prelude_pz (floating q) q s c bias initial
  have acc:=source_accumulation q s c bias b m divisor selected plans pid z valid hz i
  dsimp only
  unfold value
  rw [acc,hx,hw,rn,pz,prelude_memory]
  simp [initial,initialState,sourceMemory,asF,partialValue,Nat.mod_eq_of_lt hr,Nat.mod_eq_of_lt hc,
    hc,show ((col s c pid i.2.1:Nat):Int)<s.N by exact_mod_cast hc]

theorem integer_accumulation_real (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (m : Memory ℝ ℝ) (r n z t : Nat) :
    ((Int8DequantMatmul.accumulation integerRealOps s c b selected m r n z t : Int):ℝ)=
      Int8DequantMatmul.accumulation realOps s c b selected m r n z t := by
  induction t with
  | zero=>simp [Int8DequantMatmul.accumulation,integerRealOps,realOps]
  | succ t ih=>simp only [Int8DequantMatmul.accumulation,integer_plan_real,ih]

theorem integer_partial_real (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (m : Memory ℝ ℝ) (divisor : ℝ) (bias : Bool) (r n z : Nat) :
    partialValue integerRealOps s c b selected divisor bias m r n z=
      partialValue realOps s c b selected divisor bias m r n z := by
  unfold partialValue
  have h:=integer_accumulation_real s c b selected m r n z (steps s c)
  simp only [integerRealOps,realOps] at h
  simp only [integerRealOps,realOps,read32,id_eq]
  rw [h]

#axiomsClean source_value
#axiomsClean integer_accumulation_real
#axiomsClean integer_partial_real
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
