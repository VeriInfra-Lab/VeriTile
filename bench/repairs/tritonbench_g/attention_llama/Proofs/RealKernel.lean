import bench.repairs.tritonbench_g.attention_llama.Proofs.LoopInvariant
import bench.repairs.tritonbench_g.attention_llama.Proofs.Postamble

namespace VeriTile.Bench.Repairs.AttentionLlama.RealKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open SourceExecution LoopInvariant MaskCoverage
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def Invariant.setEnd (c : Context) (n : Nat) (st : BlockState)
    (h : Invariant c n st) (v : Nat) :
    Invariant c n (st.setReg "block_n_end" .nat [] (Tile.scalar v)) where
  m := h.m
  l := h.l
  a := h.a
  pids := h.pids
  mem := h.mem
  undef := h.undef
  maximum := by simpa using h.maximum
  denominator := by simpa using h.denominator
  accumulator := by simpa using h.accumulator
  query := by simpa using h.query
  rows := by simpa using h.rows
  columns := by simpa using h.columns
  batch := by simpa using h.batch
  head := by simpa using h.head
  keyPtr := by simpa using h.keyPtr
  valuePtr := by simpa using h.valuePtr
  numerical := h.numerical

theorem loop (c : Context) (st : BlockState) (bound : Nat)
    (h : Invariant c 0 st)
    (hb : st.regs .nat [] "block_n_end" = some (Tile.scalar bound))
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64) :
    ∃ sf, stepStmt (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end")
      (.constNat 64) (sourceLoop c.scale c.skn c.svk c.N c.SP 64 64 c.D c.causal)) st = some sf ∧
      Nonempty (Invariant c (count bound) sf) := by
  obtain ⟨final,sf,hrun,hfinal,hmod,hle,hinv⟩ := forRangeDyn_inv
    (idx := "start_n") (startOp := .constNat 0) (stopOp := .ref .nat [] "block_n_end")
    (stepOp := .constNat 64)
    (body := sourceLoop c.scale c.skn c.svk c.N c.SP 64 64 c.D c.causal)
    (P := fun i st => i % 64 = 0 ∧ i ≤ count bound * 64 ∧ Nonempty (Invariant c (i/64) st))
    (s_init := st) (by rw [evalOp_constNat]) (by rw [evalOp_ref,hb])
    (by rw [evalOp_constNat]) (by decide)
    ⟨by simp,by omega,⟨h⟩⟩ (by
      intro i s hi hp
      obtain ⟨hmod,hle,⟨hinv⟩⟩ := hp
      have hieq : i / 64 * 64 = i := by omega
      obtain ⟨s',hr,h'⟩ := iteration c (i/64) s hinv hD hN hSP
      refine ⟨s',?_,?_,?_,?_⟩
      · simpa only [hieq] using hr
      · omega
      · have hc := count_covers bound
        omega
      · simpa only [show (i+64)/64 = i/64+1 by omega] using h')
  have hc : final / 64 = count bound := by
    simp only [count] at *
    omega
  exact ⟨sf,hrun,by simpa only [hc] using hinv⟩

noncomputable def target (c : Context) (idx : TileIndex [64,c.D]) : ℝ :=
  TailMath.expected (c.score idx.1) (fun k => c.value k idx.2.1)
    (logicalKeys c.N c.causal c.SP (c.row idx.1))

/-- Complete repaired FP16 real source: preamble, both stop branches, every
masked key block, and actual final stores. All other memory cells are framed. -/
theorem execution (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64)
    (hout : Postamble.ActiveInjective (c.initial.pids 0 * 64) c.H 64 c.D som son)
    (hundef : ∀ rg o, c.initial.undef rg o = 0) :
    ∃ sf, exec (TypedSource.repaired c.Q c.K c.V Out c.scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
      soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal).eraseDType.toAlgKernel c.initial = some sf ∧
      (∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        sf.readMem Out (alOutOffset c.initial c.NH soz soh som son 64 idx) = target c idx) ∧
      (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        o ≠ alOutOffset c.initial c.NH soz soh som son 64 idx) → sf.mem r o = c.initial.mem r o) := by
  simp only [exec,repaired_body]
  obtain ⟨sp,hpre,hp,hm,hu,hsm,hbi,hoh,hom,hon,hmp,hlp,ha,hq,hkp,hvp,hend⟩ :=
    alPreLoop_evalG c.initial c.Q c.K c.V c.sqz c.sqh c.sqm c.sqk
      c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn c.NH c.H c.N 64 64 c.D hundef
  rw [stepStmts.append_some hpre]
  have hinv : Invariant c 0 sp := {
    m := fun _ => 0
    l := fun _ => 0
    a := fun _ => 0
    pids := hp
    mem := hm
    undef := hu
    maximum := hmp
    denominator := hlp
    accumulator := ha
    query := hq
    rows := hom
    columns := hon
    batch := hbi
    head := hoh
    keyPtr := hkp
    valuePtr := hvp
    numerical := by intro r d; exact TailMath.initial_consistent _ _
  }
  let bound := stop c.N c.causal c.SP (c.initial.pids 0)
  let sc := if c.causal then sp.setReg "block_n_end" .nat [] (Tile.scalar bound) else sp
  have hsc : Nonempty (Invariant c 0 sc) := by
    cases hc : c.causal
    · exact ⟨by simpa [sc,hc] using hinv⟩
    · exact ⟨by simpa [sc,hc] using Invariant.setEnd c 0 sp hinv bound⟩
  obtain ⟨hinvc⟩ := hsc
  have hb : sc.regs .nat [] "block_n_end" = some (Tile.scalar bound) := by
    cases hc : c.causal <;> simp [sc,hc,bound,stop,hend]
  obtain ⟨sl,hl,⟨hinvl⟩⟩ := loop c sc bound hinvc hb hD hN hSP
  have hacc : sl.regs .real [64,c.D] "acc" = some ⟨fun idx => some (target c idx)⟩ := by
    rw [hinvl.accumulator]
    apply congrArg some
    ext idx
    have hn := (hinvl.numerical idx.1 idx.2.1).2
    dsimp only at hn
    rw [show TailMath.processed (c.keys idx.1) (count bound) =
      logicalKeys c.N c.causal c.SP (c.row idx.1) from
      all_keys_covered c.N c.causal c.SP (c.initial.pids 0) idx.1.val idx.1.isLt] at hn
    exact congrArg some hn
  obtain ⟨sf,hpost,hread,hframe⟩ := Postamble.run Out c.initial soz soh som son c.NH c.H 64 c.D
    sl (target c) hout hinvl.pids hacc hinvl.batch hinvl.head
  refine ⟨sf,?_,hread,?_⟩
  · rw [ProgramRewrite.constant_branch]
    cases hc : c.causal
    · simp only [Bool.false_eq_true,↓reduceIte,List.nil_append]
      rw [stepStmts.cons_some (by simpa [sc,hc] using hl)]
      exact hpost
    · simp only [↓reduceIte,List.cons_append,List.nil_append]
      have he : evalOp (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat 64))
          (.constNat c.SP)) sp = some (Tile.scalar bound) := by
        rw [evalOp_add,evalOp_mul,evalOp_add]
        simp only [evalOp_ref,hsm,evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
        apply congrArg some
        ext u
        simp [bound,stop,hc,Tile.bop_data,Tile.scalar_data,NumericDType.add,NumericDType.mul,
          Broadcast.leftIndex,Broadcast.rightIndex]
      unfold alBnEndCausalStmt
      rw [stepStmts.cons_some (stepStmt_assign_eq_some he)]
      rw [stepStmts.cons_some (by simpa [sc,hc] using hl)]
      exact hpost
  · intro r o ho
    exact (hframe r o ho).trans (congrFun (congrFun hinvl.mem r) o)

end VeriTile.Bench.Repairs.AttentionLlama.RealKernel
