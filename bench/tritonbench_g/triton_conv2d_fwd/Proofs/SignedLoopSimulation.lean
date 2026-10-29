import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedBodySimulation

/-! The checked local masked-pointer simulation lifted through all three
source loops. Memory and every other register remain related. -/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedLoopSimulation
open VeriTile.Triton MaskedPointerSimulation SignedBodySimulation
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem singleton (R : RoundingModel) (st : Stmt) (s : BlockState) :
    stepStmtsR R [st] s = stepStmtR R st s := by
  rw [stepStmtsR]
  cases h : stepStmtR R st s <;> simp only [stepStmtsR]

theorem range_sim (R : RoundingModel) {a b : List Stmt}
    (body : Simulates R a b) (index : RegName) (first last stride : Nat) :
    Simulates R [.forRange index first last stride a]
      [.forRange index first last stride b] := by
  intro s t h t' run
  rw [singleton] at run ⊢
  rw [stepStmtR] at run ⊢
  change stepForRangeAuxR R index first last stride b t = some t' at run
  change ∃ s', stepForRangeAuxR R index first last stride a s = some s' ∧ Related s' t'
  exact MaskedPointerSimulation.range R Related index a b
    (fun s t h n => ⟨h.1.setReg index .nat [] (Tile.scalar n),h.2⟩)
    body first last stride s t t' h run

theorem channel_sim (R : RoundingModel) {a b : List Stmt}
    (body : Simulates R a b) (stride : Nat) :
    Simulates R [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat stride) a]
      [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat stride) b] := by
  intro s t h t' run
  rw [singleton] at run ⊢
  simp only [stepStmtR,evalOpR,Option.bind,Tile.scalar] at run ⊢
  rw [h.1.registers .nat [] "in_group_dim" (by decide)]
  cases hv : t.regs .nat [] "in_group_dim" with
  | none => simp [hv] at run
  | some v =>
    simp only [hv] at run ⊢
    exact MaskedPointerSimulation.range R Related "c" a b
      (fun s t h n => ⟨h.1.setReg "c" .nat [] (Tile.scalar n),h.2⟩)
      body 0 (v.data PUnit.unit) stride s t t' h run

theorem nest (R : RoundingModel)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW KH KW : Nat)
    (tf32 : Bool) :
    Simulates R
      [.forRange "h" 0 KH 1 [.forRange "w" 0 KW 1
        [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat BIN)
          (SignedPointerModel.inner BHW BIN OF IIFS IHS IWS WIFS WHS WWS
            batch_dim in_height in_width PH PW SH SW tf32)]]]
      [.forRange "h" 0 KH 1 [.forRange "w" 0 KW 1
        [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat BIN)
          (PositivePointerModel.convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS
            batch_dim in_height in_width PH PW SH SW tf32)]]] :=
  range_sim R (range_sim R (channel_sim R
    (SignedBodySimulation.inner R BHW BIN OF IIFS IHS IWS WIFS WHS WWS
      batch_dim in_height in_width PH PW SH SW tf32) BIN) "w" 0 KW 1) "h" 0 KH 1

#axiomsClean nest
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedLoopSimulation
