import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedPointerModel

/-! Local execution simulation for the actual signed-pointer body. The relation
allows only the masked input-pointer register to differ. -/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedBodySimulation
open VeriTile.Triton MaskedPointerSimulation
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def Related (s t : BlockState) : Prop :=
  AgreeExcept "curr_input_pointer" s t ∧ ∀ r o, t.undef r o = 0

def Simulates (R : RoundingModel) (source model : List Stmt) : Prop :=
  ∀ s t, Related s t → ∀ t', stepStmtsR R model t = some t' →
    ∃ s', stepStmtsR R source s = some s' ∧ Related s' t'

theorem append (R : RoundingModel) {a b c d : List Stmt}
    (first : Simulates R a b) (next : Simulates R c d) :
    Simulates R (a ++ c) (b ++ d) := by
  intro s t h t' run
  rw [stepStmtsR_append] at run ⊢
  cases ht : stepStmtsR R b t with
  | none => simp [ht] at run
  | some u =>
    obtain ⟨v,hv,hr⟩ := first s t h u ht
    obtain ⟨v',hv',hr'⟩ := next v u hr t' (by simpa [ht] using run)
    exact ⟨v',by simpa [hv] using hv',hr'⟩

theorem assign (R : RoundingModel) (d : TileDType) (sh : TileShape)
    (name : RegName) (a b : Op d sh)
    (same : ∀ s t, Related s t → evalOpR R a s = evalOpR R b t) :
    Simulates R [.assign d sh name a] [.assign d sh name b] := by
  intro s t h t' run
  simp only [stepStmtsR,stepStmtR] at run ⊢
  rw [same s t h]
  cases ht : evalOpR R b t with
  | none => simp [ht] at run
  | some v =>
    simp only [ht,Option.bind_some] at run ⊢
    cases run
    exact ⟨s.setReg name d sh v,rfl,h.1.setReg name d sh v,h.2⟩

theorem common_prefix (R : RoundingModel)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat) :
    let body := PositivePointerModel.convInnerBodyFactored BHW BIN OF IIFS IHS IWS
      WIFS WHS WWS batch_dim in_height in_width PH PW SH SW
    Simulates R (body.take 3) (body.take 3) := by
  dsimp only [PositivePointerModel.convInnerBodyFactored,List.take]
  change Simulates R ([_] ++ [_] ++ [_]) ([_] ++ [_] ++ [_])
  apply append R
  · apply append R
    · apply assign R
      intro s t h
      simp only [evalOpR,h.1.registers .nat [] "c" (by decide)]
    · apply assign R
      intro s t h
      simp only [evalOpR,h.1.registers .nat [] "h" (by decide),
        h.1.registers .nat [BHW] "output_height_offset" (by decide)]
  · apply assign R
    intro s t h
    simp only [evalOpR,h.1.registers .nat [] "w" (by decide),
      h.1.registers .nat [BHW] "output_width_offset" (by decide)]

theorem if_true (R : RoundingModel) {a b c d : List Stmt}
    (h : Simulates R a b) :
    Simulates R [.ifThenElse (.constBool Bool.true) a c]
      [.ifThenElse (.constBool Bool.true) b d] := by
  have one (xs ys : List Stmt) (s : BlockState) :
      stepStmtsR R [.ifThenElse (.constBool Bool.true) xs ys] s = stepStmtsR R xs s := by
    rw [stepStmtsR,stepStmtR]
    simp only [evalOpR,Tile.scalar,bind,Option.bind,ite_true]
    cases hx : stepStmtsR R xs s <;> simp only [stepStmtsR]
  intro s t related t' run
  rw [one] at run
  obtain ⟨s',hs,hr⟩ := h s t related t' run
  exact ⟨s',by rw [one]; exact hs,hr⟩

theorem common_suffix (R : RoundingModel)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat)
    (tf32 : Bool) :
    let body := PositivePointerModel.convInnerBody BHW BIN OF IIFS IHS IWS
      WIFS WHS WWS batch_dim in_height in_width PH PW SH SW tf32
    Simulates R (body.drop 8) (body.drop 8) := by
  change Simulates R ([_] ++ [PositivePointerModel.conv_newBranch BHW BIN OF tf32])
    ([_] ++ [PositivePointerModel.conv_newBranch BHW BIN OF tf32])
  apply append R
  · apply assign R
    intro s t h
    simp [evalOpR,h.1.registers .ptr [BIN,OF] "curr_weight_pointer" (by decide),
      h.1.registers .bool [BIN,OF] "weight_mask" (by decide),
      BlockState.readMemValue_real,BlockState.readMem,h.1.memory,h.1.undef]
  · dsimp only [PositivePointerModel.conv_newBranch]
    apply if_true R
    change Simulates R ([_] ++ [_] ++ [_]) ([_] ++ [_] ++ [_])
    apply append R
    · apply append R
      · apply assign R
        intro s t h
        simp only [evalOpR.eq_def,FloatDType.toTileDType,List.nil_append,h.1.registers .real [BHW,BIN] "input_block" (by decide)]
      · apply assign R
        intro s t h
        simp only [evalOpR.eq_def,FloatDType.toTileDType,List.nil_append,h.1.registers .real [BIN,OF] "weight_block" (by decide)]
    · apply assign R
      intro s t h
      simp only [evalOpR.eq_def,FloatDType.toTileDType,List.nil_append,h.1.registers .real [BHW,OF] "accum" (by decide),
        h.1.registers .fp16 [BHW,BIN] "input_block" (by decide),
        h.1.registers .fp16 [BIN,OF] "weight_block" (by decide)]

theorem pointer_chunk (R : RoundingModel)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat) :
    Simulates R
      ((SignedPointerModel.innerFactored BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW).drop 3 |>.take 5)
      ((PositivePointerModel.convInnerBodyFactored BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW).drop 3 |>.take 5) := by
  intro s t h t' run
  have shared := h.1.registers
  simp only [PositivePointerModel.convInnerBodyFactored,List.drop,List.take,
    stepStmtsR,stepStmtR,evalOpR.eq_def,BlockState.setReg] at run
  cases hv0 : t.regs .ptr [BHW,1] "Input" with
  | none => simp [hv0,Option.bind] at run
  | some v0 =>
    simp [hv0,Option.bind] at run
    cases hv1 : t.regs .nat [BIN] "input_feat_offset" with
    | none => simp [hv0,hv1,Option.bind] at run
    | some v1 =>
      simp [hv0,hv1,Option.bind] at run
      cases hv2 : t.regs .int [BHW] "input_height_offset" with
      | none => simp [hv0,hv1,hv2,Option.bind] at run
      | some v2 =>
        simp [hv0,hv1,hv2,Option.bind] at run
        cases hv3 : t.regs .int [BHW] "input_width_offset" with
        | none => simp [hv0,hv1,hv2,hv3,Option.bind] at run
        | some v3 =>
          simp [hv0,hv1,hv2,hv3,Option.bind] at run
          cases hv4 : t.regs .ptr [1,OF] "Weight" with
          | none => simp [hv0,hv1,hv2,hv3,hv4,Option.bind] at run
          | some v4 =>
            simp [hv0,hv1,hv2,hv3,hv4,Option.bind] at run
            cases hv5 : t.regs .nat [] "h" with
            | none => simp [hv0,hv1,hv2,hv3,hv4,hv5,Option.bind] at run
            | some v5 =>
              simp [hv0,hv1,hv2,hv3,hv4,hv5,Option.bind] at run
              cases hv6 : t.regs .nat [] "w" with
              | none => simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,Option.bind] at run
              | some v6 =>
                simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,Option.bind] at run
                cases hv7 : t.regs .nat [BHW] "batch_offset" with
                | none => simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,Option.bind] at run
                | some v7 =>
                  simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,Option.bind] at run
                  cases hv8 : t.regs .nat [] "in_group_dim" with
                  | none => simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8,Option.bind] at run
                  | some v8 =>
                    simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8,Option.bind] at run
                    cases hv9 : t.regs .nat [OF] "output_feat_offset" with
                    | none => simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8,hv9,Option.bind] at run
                    | some v9 =>
                      simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8,hv9,Option.bind] at run
                      cases hv10 : t.regs .nat [] "out_group_dim" with
                      | none => simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8,hv9,hv10,Option.bind] at run
                      | some v10 =>
                        simp [hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8,hv9,hv10,Option.bind] at run
                        subst t'
                        simp only [SignedPointerModel.innerFactored,List.drop,List.take,
                          stepStmtsR,stepStmtR,evalOpR.eq_def,BlockState.setReg]
                        simp [shared,hv0,hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8,hv9,hv10,Option.bind]
                        refine ⟨⟨h.1.memory,h.1.pids,h.1.numPids,h.1.undef,?_⟩,h.2⟩
                        intro dt sh name hn
                        by_cases hi : name = "input_block"
                        · subst name
                          by_cases hd : dt = .real
                          · subst dt
                            by_cases hs : sh = [BHW,BIN]
                            · subst sh
                              simp only [↓reduceIte,↓reduceDIte,castTile]
                              apply congrArg some
                              apply Tile.ext
                              intro i
                              simp only [BlockState.readMemValue_real,BlockState.readMem,
                                h.1.memory,h.1.undef,h.2]
                              split_ifs with hm
                              · have hh : 0 ≤ v2.data (i.1,PUnit.unit) := hm.1.1.1.2
                                have hw : 0 ≤ v3.data (i.1,PUnit.unit) := hm.1.2
                                have nh : (-((IHS : Int)*v2.data (i.1,PUnit.unit))).toNat = 0 :=
                                  Int.toNat_eq_zero.mpr (neg_nonpos.mpr (mul_nonneg (Int.natCast_nonneg _) hh))
                                have nw : (-((IWS : Int)*v3.data (i.1,PUnit.unit))).toNat = 0 :=
                                  Int.toNat_eq_zero.mpr (neg_nonpos.mpr (mul_nonneg (Int.natCast_nonneg _) hw))
                                simp [Tile.ptrAdd,Tile.ptrSub,Tile.bop,Tile.expandDim,Tile.scalar,
                                  Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul,
                                  NumericDType.sub,nh,nw]
                              · rfl
                            · simp [hs]
                          · simp [hd]
                        · simp [hi,hn,shared]

theorem inner (R : RoundingModel)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat)
    (tf32 : Bool) :
    Simulates R
      (SignedPointerModel.inner BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW tf32)
      (PositivePointerModel.convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW tf32) := by
  have hp := common_prefix R BHW BIN OF IIFS IHS IWS WIFS WHS WWS
    batch_dim in_height in_width PH PW SH SW
  have hm := pointer_chunk R BHW BIN OF IIFS IHS IWS WIFS WHS WWS
    batch_dim in_height in_width PH PW SH SW
  have hs := common_suffix R BHW BIN OF IIFS IHS IWS WIFS WHS WWS
    batch_dim in_height in_width PH PW SH SW tf32
  have whole := append R (append R hp hm) hs
  exact whole

#axiomsClean pointer_chunk
#axiomsClean common_suffix
#axiomsClean inner
#axiomsClean common_prefix
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedBodySimulation
