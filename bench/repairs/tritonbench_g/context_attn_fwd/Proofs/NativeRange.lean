import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeControl
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FPSuccess

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
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

/-- Actual PTX signed64 start_n test and BLOCK_N backedge. -/
def bitLoop (M : Algebra α) (body : List ComputeStmt) (bm : Nat)
    (stop : BitVec 64) : Nat → BitVec 64 → State α → Option (State α)
  | 0,_,_ => none
  | n+1,index,s =>
    if index.slt stop then do
      let t ← run M body (s.setReg "start_n" .int [] (fun _ => index.toInt))
      bitLoop M body bm stop n (index+scalar bm) t
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
    (hm : Tile bm) (hstop : stop<2^31) (hs : step+left=ceilTiles stop bm) (s : State α) :
    bitLoop M body bm (word stop) (left+1) (word (step*bm)) s=unroll M body bm left step s := by
  have compare : ∀ j,j≤ceilTiles stop bm →
      ((word (j*bm)).slt (word stop)= Bool.true ↔ j<ceilTiles stop bm) := by
    intro j hj
    rw [slt_exact _ _ (final_index_bound stop bm j hm hstop hj) (by omega)]
    exact (ceil_step stop bm j hm).symm
  induction left generalizing step s with
  | zero =>
    have ht := compare step (by omega)
    have hn : ¬ (word (step*bm)).slt (word stop)= Bool.true := by intro h; have := ht.mp h; omega
    simp [bitLoop,unroll,hn]
  | succ left ih =>
    have ht := (compare step (by omega)).mpr (show step<ceilTiles stop bm by omega)
    have hi := signed (step*bm) (final_index_bound stop bm step hm hstop (by omega))
    have hn : word (step*bm)+scalar bm=word ((step+1)*bm) := by
      rw [scalar_eq]
      simp only [word,BitVec.ofNat_add_ofNat,Nat.succ_mul]
    rw [bitLoop]
    simp only [ht,↓reduceIte,hi,unroll]
    cases hb : run M body (s.setReg "start_n" .int [] (fun _ => ((step*bm : Nat) : Int))) with
    | none => simp [hb]
    | some t =>
      simp only [hb,Option.bind_some,hn]
      exact ih (step+1) (by omega) t

/-- Includes the final signed test and backedge, with no successful-execution
premise and no floating-value condition. -/
theorem native_range_eq (M : Algebra α) (body : List ComputeStmt) (stop bm : Nat)
    (hm : Tile bm) (hs : stop<2^31) (s : State α) :
    bitLoop M body bm (word stop) (ceilTiles stop bm+1) (word 0) s=
      rangeInt M "start_n" 0 (stop : Int) bm body s := by
  have h := bitLoop_unroll M body stop bm (ceilTiles stop bm) 0 hm hs (by omega) s
  simpa only [Nat.zero_mul,Nat.cast_zero] using
    h.trans (range_unroll M body stop bm (ceilTiles stop bm) 0 hm (by omega) s).symm

#print axioms native_range_eq
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native
