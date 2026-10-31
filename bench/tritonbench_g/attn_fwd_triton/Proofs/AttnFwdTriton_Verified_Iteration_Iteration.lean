import bench.tritonbench_g.attn_fwd_triton.Proofs.Preparation

/-! Complete changed helper iteration. The score, row reduction, rescaling,
FP16-erased value dot and accumulation use the original arithmetic theorems. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Iteration
open VeriTile.Triton Memory Preparation
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles (finite center probability dot_finite)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry update)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

abbrev Carries := RealIteration.Carries

def prefixWrites (second : Bool) : List RegName :=
  ["start_n","k_mask","k","k_scale","qk",maskName second,"m_ij","p"]
def writes (second : Bool) : List RegName :=
  prefixWrites second ++ ["l_ij","alpha","l_i","acc","v","m_i",keyName second,scaleName second,valueName second]

structure Fixed (mem s : BlockState) (c : Config) (BM BN BD : Nat) : Prop where
  memory : s.mem = mem.mem
  query : s.regs .real [BM,BD] "q" = some (finite (qvalue mem c BM BD))
  queryScale : s.regs .real [] "q_scale" = some (Tile.scalar (some (mem.readMem c.p.QS (qsbase c BM+c.startM))))
  rows : s.regs .int [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩
  keys : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  context : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context)
  keyStride : s.regs .int [] "stride_kn64" = some (Tile.scalar c.p.keyRow)
  valueStride : s.regs .int [] "stride_vk64" = some (Tile.scalar c.p.valueRow)

theorem Fixed.transport {mem s t : BlockState} {c : Config} {BM BN BD : Nat} {second : Bool}
    (f : Fixed mem s c BM BN BD) (hf : Frame (writes second) s t) : Fixed mem t c BM BN BD := by
  refine ⟨hf.mem.trans f.memory,?_,?_,?_,?_,?_,?_,?_⟩
  · exact (hf.regs _ _ "q" (by cases second <;>
      simp [writes,prefixWrites,maskName,RealPreparation.maskName,keyName,scaleName,valueName,
        RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName])).trans f.query
  · exact (hf.regs _ _ "q_scale" (by cases second <;>
      simp [writes,prefixWrites,maskName,RealPreparation.maskName,keyName,scaleName,valueName,
        RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName])).trans f.queryScale
  · exact (hf.regs _ _ "offs_m" (by cases second <;>
      simp [writes,prefixWrites,maskName,RealPreparation.maskName,keyName,scaleName,valueName,
        RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName])).trans f.rows
  · exact (hf.regs _ _ "offs_n" (by cases second <;>
      simp [writes,prefixWrites,maskName,RealPreparation.maskName,keyName,scaleName,valueName,
        RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName])).trans f.keys
  · exact (hf.regs _ _ "n_ctx" (by cases second <;>
      simp [writes,prefixWrites,maskName,RealPreparation.maskName,keyName,scaleName,valueName,
        RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName])).trans f.context
  · exact (hf.regs _ _ "stride_kn64" (by cases second <;>
      simp [writes,prefixWrites,maskName,RealPreparation.maskName,keyName,scaleName,valueName,
        RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName])).trans f.keyStride
  · exact (hf.regs _ _ "stride_vk64" (by cases second <;>
      simp [writes,prefixWrites,maskName,RealPreparation.maskName,keyName,scaleName,valueName,
        RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName])).trans f.valueStride

def prepared (BM BN BD D : Nat) (second diagonal : Bool) : List Stmt :=
  loadPrefix BD BN D second ++ [RealBody.score BM BN BD] ++ guard BM BN second diagonal ++
    RealBody.center BM BN ++ [RealBody.probability BM BN]

theorem prepares (mem s : BlockState) (c : Config) (BM BN BD start : Nat) (hN : 0 < BN)
    (second diagonal : Bool) (old : Fin BM → WithBot ℝ)
    (f : Fixed mem s c BM BN BD) (hp : Pointers s c BN BD start second)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .real [BM] "m_i" = some (⟨fun i => old i.1⟩ : Tile .real [BM])) :
    let sc := score mem c BM BN BD start
    let active := Masks.keep BM BN c.p.context c.startM start diagonal
    ∃ t, stepStmts (prepared BM BN BD c.p.headDim second diagonal) s = some t ∧
      t.regs .real [BM] "m_ij" = some (⟨fun i => center sc active old i.1⟩ : Tile .real [BM]) ∧
      t.regs .real [BM,BN] "p" = some (finite (fun i : TileIndex [BM,BN] => probability sc active old i.1 i.2.1)) ∧
      t.regs .int [] "start_n" = some (Tile.scalar start) ∧ Frame (prefixWrites second) s t := by
  dsimp only
  obtain ⟨sl,hl,hk,hks,hstart,hlf⟩ := loads_execute s c BN BD start second hp f.keys hs f.context
  rw [k_same_memory s mem c BD BN start f.memory] at hk
  simp only [BlockState.readMem,f.memory] at hks
  have fl := f.transport (second := second) (hlf.mono (by simp [writes,prefixWrites]))
  obtain ⟨ss,hss,hscore,hsf⟩ := RealNumeric.score_executes sl BM BN BD
    (qvalue mem c BM BD) (kvalue mem c BD BN start)
    (mem.readMem c.p.QS (qsbase c BM+c.startM)) (mem.readMem c.p.KS (ksbase c BN+start/BN))
    fl.query hk fl.queryScale hks
  have fs := fl.transport (second := second) (hsf.mono (by simp [writes,prefixWrites]))
  obtain ⟨sg,hg,hguard,hgf⟩ := Masks.guard_executes ss BM BN c.p.context c.startM start second diagonal
    (score mem c BM BN BD start) hscore fs.rows fs.keys
    ((hsf.regs _ _ "start_n" (by simp)).trans hstart) fs.context
  have hbase : Frame ["start_n","k_mask","k","k_scale","qk",maskName second] s sg :=
    ((hlf.mono (by simp)).trans (hsf.mono (by simp))).trans (hgf.mono (by simp))
  obtain ⟨t,ht,hcenter,hprob,htf⟩ := Masks.probabilities_execute sg BM BN hN
    (score mem c BM BN BD start) (Masks.keep BM BN c.p.context c.startM start diagonal) old
    ((hbase.regs _ _ "m_i" (by cases second <;> simp [maskName,RealPreparation.maskName])).trans hm) hguard
  have hslist : stepStmts [RealBody.score BM BN BD] sl = some ss := by
    rw [stepStmts.cons_some hss,stepStmts.nil]
  refine ⟨t,?_,hcenter,hprob,?_,(hbase.mono (by simp [prefixWrites])).trans (htf.mono (by simp [prefixWrites]))⟩
  · simpa only [prepared,List.append_assoc] using
      (stepStmts.append_some ((stepStmts.append_some ((stepStmts.append_some hl).trans hslist)).trans hg)).trans ht
  · exact (htf.regs _ _ "start_n" (by simp)).trans
      ((hgf.regs _ _ "start_n" (by cases second <;> simp [maskName,RealPreparation.maskName])).trans
        ((hsf.regs _ _ "start_n" (by simp)).trans hstart))

def updated (mem : BlockState) (c : Config) (BM BN BD start : Nat) (diagonal : Bool)
    (carry : Carry BM BD) : Carry BM BD :=
  update carry (score mem c BM BN BD start) (Masks.keep BM BN c.p.context c.startM start diagonal)
    (vvalue mem c BN BD start)

theorem executes (mem s : BlockState) (c : Config) (BM BN BD start : Nat) (hN : 0 < BN)
    (second diagonal : Bool) (carry : Carry BM BD)
    (f : Fixed mem s c BM BN BD) (hc : Carries s carry) (hp : Pointers s c BN BD start second)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start)) :
    ∃ t, stepStmts (flatIteration BM BN BD c.p.headDim second diagonal) s = some t ∧
      Carries t (updated mem c BM BN BD start diagonal carry) ∧
      Pointers t c BN BD (start+BN) second ∧ Fixed mem t c BM BN BD ∧ Frame (writes second) s t := by
  let sc := score mem c BM BN BD start
  let active := Masks.keep BM BN c.p.context c.startM start diagonal
  obtain ⟨sp,hpstep,hcenter,hprob,hstart,hpf⟩ := prepares mem s c BM BN BD start hN second diagonal
    carry.center f hp hs hc.center
  obtain ⟨sr,hrstep,hrden,hracc,hrf⟩ := RealNumeric.rescale_executes sp BM BN BD
    (⟨fun i => carry.center i.1⟩ : Tile .real [BM])
    (⟨fun i => center sc active carry.center i.1⟩ : Tile .real [BM])
    (finite (fun i : TileIndex [BM] => carry.denominator i.1))
    (finite (fun i : TileIndex [BM,BN] => probability sc active carry.center i.1 i.2.1))
    (finite carry.numerator)
    ((hpf.regs _ _ "m_i" (by cases second <;> simp [prefixWrites,maskName,RealPreparation.maskName])).trans hc.center)
    hcenter
    ((hpf.regs _ _ "l_i" (by cases second <;> simp [prefixWrites,maskName,RealPreparation.maskName])).trans hc.denominator)
    hprob
    ((hpf.regs _ _ "acc" (by cases second <;> simp [prefixWrites,maskName,RealPreparation.maskName])).trans hc.numerator)
  rw [VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles.alpha_matches BM BN sc active carry.center] at hrden hracc
  have hden : sr.regs .real [BM] "l_i" =
      some (finite (fun i : TileIndex [BM] => (updated mem c BM BN BD start diagonal carry).denominator i.1)) := by
    refine hrden.trans (congrArg some ?_)
    apply Tile.ext
    rintro ⟨row,⟨⟩⟩
    exact VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles.denominator_matches BM BN sc active
      carry.center carry.denominator row (fun _ => 0) 0
  let scaled := fun i : TileIndex [BM,BD] => carry.numerator i *
    VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFormula.weight (carry.center i.1) (center sc active carry.center i.1)
  have hacc : sr.regs .real [BM,BD] "acc" = some (finite scaled) := by
    exact hracc.trans (congrArg some (by apply Tile.ext; rintro ⟨row,d,⟨⟩⟩; rfl))
  have hrbase : Frame (prefixWrites second ++ ["l_ij","alpha","l_i","acc"]) s sr :=
    (hpf.mono (by simp)).trans (hrf.mono (by simp))
  have fr := f.transport (second := second) (hrbase.mono (by simp [writes]))
  obtain ⟨sv,hvstep,hvval,hvf⟩ := value_executes sr c BN BD start second
    ((hrbase.regs _ _ (valueName second)
      (by cases second <;> simp [prefixWrites,maskName,RealPreparation.maskName,valueName,RealMemory.valueName])).trans hp.value)
    fr.keys ((hrf.regs _ _ "start_n" (by simp)).trans hstart) fr.context
  rw [v_same_memory sr mem c BN BD start fr.memory] at hvval
  obtain ⟨sa,hastep,hanum,ham,haf⟩ := RealNumeric.accumulate_executes sv BM BN BD
    (fun i : TileIndex [BM,BN] => probability sc active carry.center i.1 i.2.1)
    (vvalue mem c BN BD start) scaled (⟨fun i => center sc active carry.center i.1⟩ : Tile .real [BM])
    ((hvf.regs _ _ "p" (by simp)).trans ((hrf.regs _ _ "p" (by simp)).trans hprob)) hvval
    ((hvf.regs _ _ "acc" (by simp)).trans hacc)
    ((hvf.regs _ _ "m_ij" (by simp)).trans ((hrf.regs _ _ "m_ij" (by simp)).trans hcenter))
  have hnum : sa.regs .real [BM,BD] "acc" = some (finite (updated mem c BM BN BD start diagonal carry).numerator) := by
    refine hanum.trans (congrArg some ?_)
    apply Tile.ext
    rintro ⟨row,d,⟨⟩⟩
    have h := VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles.numerator_matches BM BN BD sc active
      carry.center carry.numerator (vvalue mem c BN BD start) row d (carry.denominator row)
    rw [dot_finite] at h
    exact h
  have habase : Frame (prefixWrites second ++ ["l_ij","alpha","l_i","acc","v","p","m_i"]) s sa :=
    ((hrbase.mono (by simp)).trans (hvf.mono (by simp))).trans (haf.mono (by simp))
  have fa := f.transport (second := second) (habase.mono (by simp [writes,prefixWrites]))
  have hptr : Pointers sa c BN BD start second := by
    constructor
    · exact (habase.regs _ _ (keyName second) (by cases second <;>
        simp [prefixWrites,maskName,RealPreparation.maskName,keyName,RealMemory.keyName])).trans hp.key
    · exact (habase.regs _ _ (scaleName second) (by cases second <;>
        simp [prefixWrites,maskName,RealPreparation.maskName,scaleName,RealMemory.scaleName])).trans hp.scale
    · exact (habase.regs _ _ (valueName second) (by cases second <;>
        simp [prefixWrites,maskName,RealPreparation.maskName,valueName,RealMemory.valueName])).trans hp.value
  obtain ⟨t,htstep,htptr,htf⟩ := advances_execute sa c BN BD start hN second hptr fa.keyStride fa.valueStride
  have hframe : Frame (writes second) s t :=
    (habase.mono (by simp [writes,prefixWrites])).trans (htf.mono (by simp [writes]))
  refine ⟨t,?_,⟨?_,?_,?_⟩,htptr,f.transport hframe,hframe⟩
  · have hvlist : stepStmts [valueLoad BN BD c.p.headDim second] sr = some sv := by
      rw [stepStmts.cons_some hvstep,stepStmts.nil]
    have hrescale := (stepStmts.append_some hpstep).trans hrstep
    have hvalue := (stepStmts.append_some hrescale).trans hvlist
    have haccum := (stepStmts.append_some hvalue).trans hastep
    simpa only [flatIteration,prepared,suffix,List.append_assoc,List.cons_append,List.nil_append] using
      (stepStmts.append_some haccum).trans htstep
  · exact (htf.regs _ _ "m_i" (by cases second <;> decide)).trans ham
  · exact (htf.regs _ _ "l_i" (by cases second <;> decide)).trans
      ((haf.regs _ _ "l_i" (by simp)).trans ((hvf.regs _ _ "l_i" (by simp)).trans hden))
  · exact (htf.regs _ _ "acc" (by cases second <;> decide)).trans hnum

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Iteration
