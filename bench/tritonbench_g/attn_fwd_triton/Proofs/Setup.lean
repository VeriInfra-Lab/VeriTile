import bench.tritonbench_g.attn_fwd_triton.Proofs.AttnFwdTriton_Verified_Loop_Loop

/-! Execute the repaired diagonal setup: context-bounded upper endpoint,
independent key/value row strides, and pointer reset from caller bases. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Setup
open VeriTile.Triton Memory Preparation
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

abbrev lowerName := RealSetup.lowerName
abbrev upperName := RealSetup.upperName
abbrev keyBase := RealSetup.keyBase
abbrev scaleBase := RealSetup.scaleBase
abbrev valueBase := RealSetup.valueBase
abbrev lowerExpr := RealSetup.lowerExpr

def upperExpr (BM : Nat) : Op .nat [] :=
  .where (.lt .nat .nil (RealSetup.upperExpr BM) (.ref .nat [] "n_ctx"))
    (RealSetup.upperExpr BM) (.ref .nat [] "n_ctx")

def code (BM BN BD : Nat) (second : Bool) : List Stmt :=
  [.assign .nat [] (lowerName second) (lowerExpr BM),
   .assign .nat [] (upperName second) (upperExpr BM),
   .assign .nat [] (lowerName second) (.ref .nat [] (lowerName second)),
   .assign .ptr [] (scaleName second) (.ptrAdd .nil (.ref .ptr [] (scaleBase second))
     (.floorDiv .nat .nil (.ref .nat [] (lowerName second)) (.constNat BN))),
   .assign .ptr [BD,BN] (keyName second) (.ptrAdd .scalarR (.ref .ptr [BD,BN] (keyBase second))
     (.mul .nat .nil (.ref .nat [] "stride_kn64") (.ref .nat [] (lowerName second)))),
   .assign .ptr [BN,BD] (valueName second) (.ptrAdd .scalarR (.ref .ptr [BN,BD] (valueBase second))
     (.mul .nat .nil (.ref .nat [] "stride_vk64") (.ref .nat [] (lowerName second))))]

def writes (second : Bool) : List RegName :=
  [lowerName second,upperName second,scaleName second,keyName second,valueName second]

theorem key_shift (s : BlockState) (c : Config) (BN BD start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [BD,BN] ptr = some (keyPointer c BD BN 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start))
    (hr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BD,BN] ptr)
      (.mul .nat .nil (.ref .nat [] "stride_kn64") (.ref .nat [] low))) s = some (keyPointer c BD BN start) := by
  simp only [evalOp.eq_def, hp, hl, hr, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyPointer,Tile.scalar,NumericDType.mul,-Tile.scalar_eta]
  ring

theorem value_shift (s : BlockState) (c : Config) (BN BD start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [BN,BD] ptr = some (valuePointer c BN BD 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start))
    (hr : s.regs .nat [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BN,BD] ptr)
      (.mul .nat .nil (.ref .nat [] "stride_vk64") (.ref .nat [] low))) s = some (valuePointer c BN BD start) := by
  simp only [evalOp.eq_def, hp, hl, hr, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp [valuePointer,Tile.scalar,NumericDType.mul,-Tile.scalar_eta]
  ring

theorem scale_shift (s : BlockState) (c : Config) (BN start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [] ptr = some (keyScalePointer c BN 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start)) :
    evalOp (.ptrAdd .nil (.ref .ptr [] ptr)
      (.floorDiv .nat .nil (.ref .nat [] low) (.constNat BN))) s = some (keyScalePointer c BN start) := by
  simp only [evalOp.eq_def, hp, hl, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [keyScalePointer, Tile.scalar, IntegralDType.floorDiv, -Tile.scalar_eta]

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat) (second : Bool)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM))
    (hk : s.regs .ptr [BD,BN] (keyBase second) = some (keyPointer c BD BN 0))
    (hs : s.regs .ptr [] (scaleBase second) = some (keyScalePointer c BN 0))
    (hv : s.regs .ptr [BN,BD] (valueBase second) = some (valuePointer c BN BD 0))
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context))
    (hkr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow))
    (hvr : s.regs .nat [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    ∃ t, stepStmts (code BM BN BD second) s = some t ∧
      t.regs .nat [] (lowerName second) = some (Tile.scalar (c.startM*BM)) ∧
      t.regs .nat [] (upperName second) = some (Tile.scalar (min ((c.startM+1)*BM) c.p.context)) ∧
      Pointers t c BN BD (c.startM*BM) second ∧ Frame (writes second) s t := by
  let sl := s.setReg (lowerName second) .nat [] (Tile.scalar (c.startM*BM))
  let sh := sl.setReg (upperName second) .nat [] (Tile.scalar (min ((c.startM+1)*BM) c.p.context))
  let sl' := sh.setReg (lowerName second) .nat [] (Tile.scalar (c.startM*BM))
  let ss := sl'.setReg (scaleName second) .ptr [] (keyScalePointer c BN (c.startM*BM))
  let sk := ss.setReg (keyName second) .ptr [BD,BN] (keyPointer c BD BN (c.startM*BM))
  have hlow : evalOp (lowerExpr BM) s = some (Tile.scalar (c.startM*BM)) := by
    simp [lowerExpr,RealSetup.lowerExpr, evalOp.eq_def, hm]; rfl
  have hhigh : evalOp (upperExpr BM) sl = some (Tile.scalar (min ((c.startM+1)*BM) c.p.context)) := by
    cases second <;> simp [upperExpr, RealSetup.upperExpr, evalOp.eq_def, sl, lowerName,RealSetup.lowerName, RealSetup.lowerName,RealSetup.lowerName, hm, hctx]
    all_goals
      congr 1
      apply Tile.ext
      rintro ⟨⟩
      simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,ComparableDType.lt,
        NumericDType.add,NumericDType.mul,min_def,-Tile.scalar_eta]
      change (if (c.startM+1)*BM < c.p.context then (c.startM+1)*BM else c.p.context) =
        (if (c.startM+1)*BM ≤ c.p.context then (c.startM+1)*BM else c.p.context)
      split_ifs <;> omega
  have hcopy : evalOp (.ref .nat [] (lowerName second)) sh = some (Tile.scalar (c.startM*BM)) := by
    cases second <;> simp [sh, sl, lowerName,RealSetup.lowerName, upperName,RealSetup.upperName]
  have hscale := scale_shift sl' c BN (c.startM*BM) (scaleBase second) (lowerName second)
    (by cases second <;> simpa [sl', sh, sl, scaleBase,RealSetup.scaleBase, lowerName,RealSetup.lowerName, upperName,RealSetup.upperName] using hs)
    (by simp [sl'])
  have hkey := key_shift ss c BN BD (c.startM*BM) (keyBase second) (lowerName second)
    (by cases second <;> simpa [ss, sl', sh, sl, keyBase,RealSetup.keyBase, scaleName,RealMemory.scaleName, lowerName,RealSetup.lowerName, upperName,RealSetup.upperName] using hk)
    (by cases second <;> simp [ss, sl', scaleName,RealMemory.scaleName, lowerName,RealSetup.lowerName,RealMemory.scaleName,RealMemory.scaleName,RealSetup.lowerName,RealSetup.lowerName])
    (by cases second <;> simp [ss,sl',sh,sl,scaleName,RealMemory.scaleName,lowerName,RealSetup.lowerName,upperName,RealMemory.scaleName,RealMemory.scaleName,RealSetup.lowerName,RealSetup.lowerName,RealSetup.upperName,hkr])
  have hvalue := value_shift sk c BN BD (c.startM*BM) (valueBase second) (lowerName second)
    (by cases second <;> simpa [sk, ss, sl', sh, sl, valueBase,RealSetup.valueBase, keyName,RealMemory.keyName, scaleName,RealMemory.scaleName, lowerName,RealSetup.lowerName, upperName,RealSetup.upperName] using hv)
    (by cases second <;> simp [sk, ss, sl', keyName,RealMemory.keyName, scaleName,RealMemory.scaleName, lowerName,RealSetup.lowerName,RealMemory.keyName,RealMemory.keyName,RealMemory.scaleName,RealMemory.scaleName,RealSetup.lowerName,RealSetup.lowerName])
    (by cases second <;> simp [sk,ss,sl',sh,sl,keyName,RealMemory.keyName,scaleName,RealMemory.scaleName,lowerName,RealSetup.lowerName,upperName,RealMemory.keyName,RealMemory.keyName,RealMemory.scaleName,RealMemory.scaleName,RealSetup.lowerName,RealSetup.lowerName,RealSetup.upperName,hvr])
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hlow),
    stepStmts.cons_some (stepStmt_assign_eq_some hhigh),
    stepStmts.cons_some (stepStmt_assign_eq_some hcopy),
    stepStmts.cons_some (stepStmt_assign_eq_some hscale),
    stepStmts.cons_some (stepStmt_assign_eq_some hkey),
    stepStmts.cons_some (stepStmt_assign_eq_some hvalue), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ⟨?_, ?_, ?_⟩, rfl, rfl, rfl, ?_⟩
  · cases second <;> simp [sk, ss, sl', valueName,RealMemory.valueName, keyName,RealMemory.keyName, scaleName,RealMemory.scaleName, lowerName,RealSetup.lowerName]
  · cases second <;> simp [sk, ss, sl', sh, valueName,RealMemory.valueName, keyName,RealMemory.keyName, scaleName,RealMemory.scaleName, lowerName,RealSetup.lowerName, upperName,RealSetup.upperName]
  · cases second <;> simp [sk, valueName,RealMemory.valueName, keyName,RealMemory.keyName]
  · cases second <;> simp [sk, ss, valueName,RealMemory.valueName, keyName,RealMemory.keyName, scaleName,RealMemory.scaleName]
  · simp
  · intro dt shape name hn
    simp only [writes, List.mem_cons, not_or] at hn
    simp [sk, ss, sl', sh, sl, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1, hn.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Setup
