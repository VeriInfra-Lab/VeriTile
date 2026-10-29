import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeControl
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPSuccess

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

/-- Counted iteration of the unchanged typed body at natural tile coordinates. -/
def unroll (M : Algebra α) (body : List ComputeStmt) (bm : Nat) : Nat → Nat → State α → Option (State α)
  | 0,_,s => some s
  | n+1,step,s => do
    let t ← run M body (s.setReg "start_n" .int [] (fun _ => ((step*bm : Nat) : Int)))
    unroll M body bm n (step+1) t

/-- Actual native signed64 loop test, unit backedge and unsigned64 start_n
conversion. Fuel accounts for every body step plus the terminating test. -/
def bitLoop (M : Algebra α) (body : List ComputeStmt) (bm : Nat)
    (count : BitVec 64) : Nat → BitVec 64 → State α → Option (State α)
  | 0,_,_ => none
  | n+1,counter,s =>
    if counter.slt count then do
      let t ← run M body (s.setReg "start_n" .int [] (fun _ => ((counter*scalar bm).toNat : Int)))
      bitLoop M body bm count n (counter+word 1) t
    else some s

theorem range_unroll (M : Algebra α) (body : List ComputeStmt) (stop bm left step : Nat)
    (hm : Tile bm) (hs : step+left = ceilTiles stop bm) (s : State α) :
    rangeInt M "start_n" ((step*bm : Nat) : Int) (stop : Int) bm body s = unroll M body bm left step s := by
  have hbm : 0 < bm := by rcases hm with rfl|rfl <;> decide
  induction left generalizing step s with
  | zero =>
    have ht : ¬ ((step*bm : Nat) : Int) < (stop : Int) := by
      have h := ceil_step stop bm step hm
      omega
    rw [rangeInt]
    simp only [Nat.ne_of_gt hbm,↓reduceIte,ht,unroll]
  | succ left ih =>
    have ht : ((step*bm : Nat) : Int) < (stop : Int) := by
      have h := (ceil_step stop bm step hm).mp (by omega)
      exact_mod_cast h
    have hn : ((step*bm : Nat) : Int)+(bm : Int) = (((step+1)*bm : Nat) : Int) := by
      push_cast
      ring
    rw [rangeInt]
    simp only [Nat.ne_of_gt hbm,↓reduceIte,ht,unroll]
    cases hb : run M body (s.setReg "start_n" .int [] (fun _ => ((step*bm : Nat) : Int))) with
    | none => simp [hb]
    | some t =>
      simp only [hb,Option.bind_some,hn]
      exact ih (step+1) (by omega) t

theorem bitLoop_unroll (M : Algebra α) (body : List ComputeStmt) (stop bm left step : Nat)
    (hm : Tile bm) (hstop : stop < 2^63) (hs : step+left = ceilTiles stop bm) (s : State α) :
    bitLoop M body bm (word (ceilTiles stop bm)) (left+1) (word step) s = unroll M body bm left step s := by
  have hc := ceil_small stop bm hstop hm
  have compare : ∀ j, j ≤ ceilTiles stop bm →
      ((word j).slt (word (ceilTiles stop bm)) = Bool.true ↔ j < ceilTiles stop bm) := by
    intro j hj
    exact slt_exact _ _ (by omega) hc
  induction left generalizing step s with
  | zero =>
    have htest := compare step (by omega)
    have hn : ¬ (word step).slt (word (ceilTiles stop bm)) = Bool.true := by
      intro h; have := htest.mp h; omega
    simp [bitLoop,unroll,hn]
  | succ left ih =>
    have htest := compare step (by omega)
    have hstep : step < ceilTiles stop bm := by omega
    have ht : (word step).slt (word (ceilTiles stop bm)) = Bool.true := htest.mpr hstep
    have hi : step*bm < 2^64 := lt_trans ((ceil_step stop bm step hm).mp hstep) (by omega)
    have hmul : (word step*scalar bm).toNat = step*bm := by
      rw [scalar_eq]
      rw [show word step*word bm = word (step*bm) by simp only [word,BitVec.ofNat_mul_ofNat]]
      exact unsigned _ hi
    have hnext : word step+word 1 = word (step+1) := by simp only [word,BitVec.ofNat_add_ofNat]
    rw [bitLoop]
    simp only [ht,↓reduceIte,hmul,unroll]
    cases hb : run M body (s.setReg "start_n" .int [] (fun _ => ((step*bm : Nat) : Int))) with
    | none => simp [hb]
    | some t =>
      simp only [hb,Option.bind_some,hnext]
      exact ih (step+1) (by omega) t

/-- Native counted control is equivalent to the source signed range for an
arbitrary typed body and state, including failure; the separate source-success
proof constructs success, while native event proofs cover the body's addresses. -/
theorem native_range_eq (M : Algebra α) (body : List ComputeStmt) (stop bm : Nat)
    (hm : Tile bm) (hstop : stop < 2^63) (s : State α) :
    bitLoop M body bm (word (ceilTiles stop bm)) (ceilTiles stop bm+1) (word 0) s =
      rangeInt M "start_n" 0 (stop : Int) bm body s := by
  rw [bitLoop_unroll M body stop bm (ceilTiles stop bm) 0 hm hstop (by omega)]
  simpa only [Nat.zero_mul,Nat.cast_zero] using (range_unroll M body stop bm (ceilTiles stop bm) 0 hm (by omega) s).symm

#print axioms native_range_eq
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
