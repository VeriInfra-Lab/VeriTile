import bench.tritonbench_g.attn_fwd_triton.Proofs.Memory
import bench.tritonbench_g.attn_fwd_triton.Proofs.Masks

/-! Execute the candidate's changed helper loads, masks and pointer advances.
All register equations below are intermediate invariants, discharged by the
prologue and loop composition; none is a numerical applicability condition. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Preparation
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles (finite)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def score (s : BlockState) (c : Config) (BM BN BD start : Nat) : Fin BM → Fin BN → ℝ :=
  RealNumeric.scores BM BN BD (qvalue s c BM BD) (kvalue s c BD BN start)
    (s.readMem c.p.QS (qsbase c BM+c.startM)) (s.readMem c.p.KS (ksbase c BN+start/BN))

theorem q_same_memory (s t : BlockState) (c : Config) (BM BD : Nat) (hm : s.mem = t.mem) :
    qvalue s c BM BD = qvalue t c BM BD := by
  funext i
  simp only [qvalue, BlockState.readMem, hm]
theorem k_same_memory (s t : BlockState) (c : Config) (BD BN start : Nat) (hm : s.mem = t.mem) :
    kvalue s c BD BN start = kvalue t c BD BN start := by
  funext i
  simp only [kvalue, BlockState.readMem, hm]
theorem v_same_memory (s t : BlockState) (c : Config) (BN BD start : Nat) (hm : s.mem = t.mem) :
    vvalue s c BN BD start = vvalue t c BN BD start := by
  funext i
  simp only [vvalue, BlockState.readMem, hm]
theorem score_same_memory (s t : BlockState) (c : Config) (BM BN BD start : Nat) (hm : s.mem = t.mem) :
    score s c BM BN BD start = score t c BM BN BD start := by
  simp only [score, q_same_memory s t c BM BD hm, k_same_memory s t c BD BN start hm,
    BlockState.readMem, hm]

structure Pointers (s : BlockState) (c : Config) (BN BD start : Nat) (second : Bool) : Prop where
  key : s.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN start)
  scale : s.regs .ptr [] (scaleName second) = some (keyScalePointer c BN start)
  value : s.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start)

theorem loads_execute (s : BlockState) (c : Config) (BN BD start : Nat) (second : Bool)
    (hp : Pointers s c BN BD start second)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context)) :
    ∃ t, stepStmts (loadPrefix BD BN c.p.headDim second) s = some t ∧
      t.regs .real [BD,BN] "k" = some (finite (kvalue s c BD BN start)) ∧
      t.regs .real [] "k_scale" = some (Tile.scalar (some (s.readMem c.p.KS (ksbase c BN+start/BN)))) ∧
      t.regs .nat [] "start_n" = some (Tile.scalar start) ∧
      Frame ["start_n","k_mask","k","k_scale"] s t := by
  let sn := s.setReg "start_n" .nat [] (Tile.scalar start)
  let sm := sn.setReg "k_mask" .bool [BD,BN] (keyMaskTile BD BN c.p.context c.p.headDim start)
  have hm := eval_key_mask sn BD BN c.p.context c.p.headDim start
    (by simpa [sn] using hn) (by simp [sn]) (by simpa [sn] using hctx)
  have hk := key_load sm c BD BN start (keyName second)
    (by cases second <;> simpa [sm,sn,keyName,RealMemory.keyName] using hp.key) (by simp [sm])
  rw [k_same_memory sm s c BD BN start rfl] at hk
  let sk := sm.setReg "k" .real [BD,BN] (finite (kvalue s c BD BN start))
  have hscale := scale_load sk c.p.KS (ksbase c BN+start/BN) (scaleName second)
    (by cases second <;> simpa [sk,sm,sn,scaleName,RealMemory.scaleName,keyScalePointer] using hp.scale)
  change evalOp (.load .real (.ptr (.ref .ptr [] (scaleName second))) .none) sk =
    some (Tile.scalar (some (s.readMem c.p.KS (ksbase c BN+start/BN)))) at hscale
  simp only [loadPrefix]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.ref .nat [] "start_n") s =
    some (Tile.scalar start) from by simpa using hs)),
    stepStmts.cons_some (stepStmt_assign_eq_some hm),
    stepStmts.cons_some (stepStmt_assign_eq_some hk),
    stepStmts.cons_some (stepStmt_assign_eq_some hscale), stepStmts.nil]
  refine ⟨_, rfl, by simp [sk], by simp, by simp [sk,sm,sn], rfl,rfl,rfl,?_⟩
  intro dt sh name hname
  simp only [List.mem_cons,not_or] at hname
  simp [sk,sm,sn,hname.1,hname.2.1,hname.2.2.1,hname.2.2.2.1]

theorem value_executes (s : BlockState) (c : Config) (BN BD start : Nat) (second : Bool)
    (hp : s.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context)) :
    ∃ t, stepStmt (valueLoad BN BD c.p.headDim second) s = some t ∧
      t.regs .real [BN,BD] "v" = some (finite (vvalue s c BN BD start)) ∧ Frame ["v"] s t := by
  refine ⟨_, stepStmt_assign_eq_some (value_load s c BN BD start (valueName second) hp hn hs hctx), by simp, ?_⟩
  exact Frame.setReg _ s "v" _ _ _ (by simp)

theorem advances_execute (s : BlockState) (c : Config) (BN BD start : Nat) (hN : 0 < BN) (second : Bool)
    (hp : Pointers s c BN BD start second)
    (hkr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow))
    (hvr : s.regs .nat [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    ∃ t, stepStmts (advance BN BD second) s = some t ∧
      Pointers t c BN BD (start+BN) second ∧ Frame [keyName second,scaleName second,valueName second] s t := by
  have hsk : scaleName second ≠ keyName second := by cases second <;> decide
  have hvk : valueName second ≠ keyName second := by cases second <;> decide
  have hvs : valueName second ≠ scaleName second := by cases second <;> decide
  let sk := s.setReg (keyName second) .ptr [BD,BN] (keyPointer c BD BN (start+BN))
  let ss := sk.setReg (scaleName second) .ptr [] (keyScalePointer c BN (start+BN))
  have h1 := key_advance s c BN BD start (keyName second) hp.key hkr
  have h2 := scale_advance sk c BN start hN (scaleName second) (by simpa [sk,hsk] using hp.scale)
  have h3 := value_advance ss c BN BD start (valueName second) (by simpa [ss,sk,hvk,hvs] using hp.value)
    (by cases second <;> simpa [ss,sk,keyName,scaleName,RealMemory.keyName,RealMemory.scaleName] using hvr)
  simp only [advance]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1),stepStmts.cons_some (stepStmt_assign_eq_some h2),
    stepStmts.cons_some (stepStmt_assign_eq_some h3),stepStmts.nil]
  refine ⟨_,rfl,⟨by simp [ss,sk,Ne.symm hvk,Ne.symm hsk],by simp [ss,Ne.symm hvs],by simp⟩,rfl,rfl,rfl,?_⟩
  intro dt sh name hn
  simp only [List.mem_cons,not_or] at hn
  simp [ss,sk,hn.1,hn.2.1,hn.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Preparation
