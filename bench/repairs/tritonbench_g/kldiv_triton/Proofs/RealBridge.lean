import bench.repairs.tritonbench_g.kldiv_triton.Proofs.BackSafety

namespace VeriTile.Bench.Repairs.KldivTriton.RealHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Host
open scoped BigOperators
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 8000000

def negation : Negation ℝ := ⟨Neg.neg,neg_neg⟩
def arithmetic : Arithmetic ℝ where
  zero := 0
  one := 1
  add := (·+·)
  mul := (·*·)
  equal := fun x y => decide (x=y)
  dividePythonNat := fun x n => x/(n:ℝ)

def row {M β : Type} (ops : Ops M ℝ β) (s : World M) (m : Matrix) (r c : Nat) : ℝ :=
  logical ops negation s.memory m.view (r*m.cols+c)

def total {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (s : World M) : ℝ :=
  Math.total call.prediction.rows call.prediction.cols call.flag (row ops s call.prediction) (row ops s call.target)

def reduced {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (s : World M) : ℝ :=
  Math.reduced call.mode call.prediction.rows call.prediction.cols call.flag
    (row ops s call.prediction) (row ops s call.target)

theorem point_real (flag : Bool) (y t : ℝ) : point Storage.real flag y t=Math.loss flag y t := RealModel.loss_real flag y t

theorem derivative_real (flag : Bool) (t : ℝ) : derivative Storage.real flag t=Math.gradient flag t := RealModel.gradient_real flag t

theorem reduced_flat {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (initial : World M)
    (output : Matrix) (rows : output.rows=call.prediction.rows) (cols : output.cols=outputCols call)
    (mode : call.mode≠.none) :
    (∑ i : Fin output.view.numel, forwardExpected ops negation Storage.real call initial i.val)=total ops call initial := by
  let count := VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds call.prediction.cols (blockSize call.prediction.cols)
  have width : outputCols call=count := by simp [outputCols,mode,count]
  have size : output.view.numel=call.prediction.rows*count := by rw [output.numel,rows,cols,width]
  rw [Fin.sum_univ_eq_sum_range,size,←Math.sum_ranges_mul]
  have values (r k : Nat) (hk : k<count) :
      forwardExpected ops negation Storage.real call initial (r*count+k)=
        Tile.chunkSum RealModel.model call.flag call.prediction.cols (blockSize call.prediction.cols) k
          (row ops initial call.prediction r) (row ops initial call.target r) := by
    have pos : 0<count := by omega
    have div : (r*count+k)/count=r := by rw [Nat.mul_comm r count,Nat.mul_add_div pos,Nat.div_eq_of_lt hk,Nat.add_zero]
    have rem : (r*count+k)%count=k := by simp [Nat.add_mod,Nat.mod_eq_of_lt hk]
    simp only [forwardExpected,width,div,rem,mode,if_false]
    rfl
  calc
    (∑ r ∈ Finset.range call.prediction.rows, ∑ k ∈ Finset.range count,
      forwardExpected ops negation Storage.real call initial (r*count+k)) =
        ∑ r ∈ Finset.range call.prediction.rows, ∑ k ∈ Finset.range count,
          Tile.chunkSum RealModel.model call.flag call.prediction.cols (blockSize call.prediction.cols) k
            (row ops initial call.prediction r) (row ops initial call.target r) := by
      apply Finset.sum_congr rfl
      intro r _
      apply Finset.sum_congr rfl
      intro k hk
      exact values r k (Finset.mem_range.mp hk)
    _ = total ops call initial := by
      have h := RealModel.all_chunks_real call.prediction.rows call.prediction.cols (blockSize call.prediction.cols)
        (block_positive _) call.flag (row ops initial call.prediction) (row ops initial call.target)
      have inner (r : Nat) :
          (∑ k : Fin count,Tile.chunkSum RealModel.model call.flag call.prediction.cols (blockSize call.prediction.cols) k.val
            (row ops initial call.prediction r) (row ops initial call.target r)) =
          ∑ k ∈ Finset.range count,Tile.chunkSum RealModel.model call.flag call.prediction.cols (blockSize call.prediction.cols) k
            (row ops initial call.prediction r) (row ops initial call.target r) :=
        Fin.sum_univ_eq_sum_range (fun k => Tile.chunkSum RealModel.model call.flag call.prediction.cols
          (blockSize call.prediction.cols) k (row ops initial call.prediction r) (row ops initial call.target r)) count
      change (∑ r : Fin call.prediction.rows, ∑ k : Fin count,
        Tile.chunkSum RealModel.model call.flag call.prediction.cols (blockSize call.prediction.cols) k.val
          (row ops initial call.prediction r.val) (row ops initial call.target r.val))=total ops call initial at h
      simp only [inner] at h
      rw [Fin.sum_univ_eq_sum_range (fun r => ∑ k ∈ Finset.range count,
        Tile.chunkSum RealModel.model call.flag call.prediction.cols (blockSize call.prediction.cols) k
          (row ops initial call.prediction r) (row ops initial call.target r))] at h
      exact h

theorem normalize_real {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (initial : World M) :
    scalarNormalize arithmetic (denominator call.mode call.prediction.rows call.prediction.cols) (total ops call initial)=
      reduced ops call initial := by
  cases hm : call.mode <;> simp [denominator,scalarNormalize,arithmetic,reduced,Math.reduced,total,hm]

/-- Independent Real API outcome: elementwise KL or mathematically normalized
full logical total, together with the supplied backward destination gradient. -/
structure ForwardTarget {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (initial : World M)
    (out : View) (after : World M) : Prop where
  result : if call.mode=.none then
    out.shape=[call.prediction.rows,call.prediction.cols] ∧ ∀ r c,r<call.prediction.rows → c<call.prediction.cols →
      logical ops negation after.memory out (r*call.prediction.cols+c)=
        Math.loss call.flag (row ops initial call.prediction r c) (row ops initial call.target r c)
    else out.shape=[] ∧ logical ops negation after.memory out 0=reduced ops call initial

theorem forward_target {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (initial after : World M)
    (output : Matrix) (out : View) (rows : output.rows=call.prediction.rows) (cols : output.cols=outputCols call)
    (result : ForwardReturned ops negation arithmetic call output (forwardExpected ops negation Storage.real call initial) out after) :
    ForwardTarget ops call initial out after := by
  constructor
  have h := result.value
  by_cases mode : call.mode=.none
  · simp only [mode,if_true] at h ⊢
    obtain ⟨rfl,values⟩ := h
    have width : output.cols=call.prediction.cols := by simpa [outputCols,mode] using cols
    refine ⟨by rw [output.shape,rows,width],?_⟩
    intro r c hr hc
    have active : r*call.prediction.cols+c<output.view.numel := by
      rw [←width]
      exact coord_bound output r c (by rw [rows];exact hr) (by rw [width];exact hc)
    rw [values _ active]
    have pos : 0<call.prediction.cols := by omega
    have div : (r*call.prediction.cols+c)/call.prediction.cols=r := by rw [Nat.mul_comm r call.prediction.cols,Nat.mul_add_div pos,Nat.div_eq_of_lt hc,Nat.add_zero]
    have rem : (r*call.prediction.cols+c)%call.prediction.cols=c := by simp [Nat.add_mod,Nat.mod_eq_of_lt hc]
    simp only [forwardExpected,outputCols,mode,if_true,div,rem,point_real]
    rfl
  · simp only [mode,if_false] at h ⊢
    refine ⟨h.shape,?_⟩
    obtain ⟨plan,value⟩ := h.value
    rw [value]
    change scalarNormalize arithmetic _ (HostReduction.eval (·+·) 0 _ plan.tree)=_
    rw [HostReduction.real_plan,reduced_flat ops call initial output rows cols mode,normalize_real]

#specNonCircular total avoiding [forwardExpected,Storage.forwardValue,Tile.chunkSum]
#specNonCircular reduced avoiding [forwardExpected,Storage.forwardValue,Tile.chunkSum]
#axiomsClean reduced_flat
#axiomsClean forward_target
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.RealHost
