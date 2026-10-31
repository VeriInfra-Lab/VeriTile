import bench.tritonbench_g.attn_fwd_triton.Proofs.Memory

/-! Candidate normalization and store, with readback for every active output
lane and a frame for every other memory cell. Generic original store lemmas
are reused through an output-only view configuration. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Output
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles (finite)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def outputConfig (c : Config) : RealLoads.Config where
  Q := c.p.Q
  K := c.p.K
  V := c.p.V
  QS := c.p.QS
  KS := c.p.KS
  base := obase c
  queryScaleBase := 0
  keyScaleBase := 0
  strideM := c.p.outputRow
  strideD := c.p.outputDim
  strideKN := 0
  headDim := c.p.headDim
  context := c.p.context
  activeHead := c.p.headDim
  startM := c.startM

def active (c : Config) (BM BD : Nat) := RealEpilogue.active (outputConfig c) BM BD
def address (c : Config) (BM BD : Nat) := RealEpilogue.address (outputConfig c) BM BD

def writeOutput (s : BlockState) (c : Config) (BM BD : Nat) (values : TileIndex [BM,BD] → ℝ) :=
  RealEpilogue.writeOutput s (outputConfig c) c.p.Out BM BD values

def store (BM BD D : Nat) : Stmt :=
  .store .real [BM,BD] (.ptr (.ref .ptr [BM,BD] "O_block_ptr")) (.ref .real [BM,BD] "acc")
    (.mask (queryMask BM BD D))

theorem store_executes (s : BlockState) (c : Config) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ)
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD))
    (hr : s.regs .int [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context))
    (hv : s.regs .real [BM,BD] "acc" = some (finite values)) :
    stepStmt (store BM BD c.p.headDim) s = some (writeOutput s c BM BD values) := by
  have hm := eval_query_mask s BM BD c.p.context c.p.headDim c.startM hr hctx
  simp only [store]
  conv_lhs => unfold stepStmt
  simp only [evalOp_ref,hp,hv,hm,bind,Option.bind,Option.map_some]
  simp [writeOutput,RealEpilogue.writeOutput,RealEpilogue.active,RealEpilogue.address,outputConfig,
    outputPointer,queryMaskTile,finite,BlockState.writeMemTyped_real,FloatDType.storeValue,
    FloatDType.real_toWithBot,WithBot.unbotD_some,Region.cast]

theorem executes (s : BlockState) (c : Config) (BM BD : Nat) (reciprocal : Bool)
    (a : TileIndex [BM,BD] → ℝ) (l : Fin BM → ℝ)
    (ha : s.regs .real [BM,BD] "acc" = some (finite a))
    (hl : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => l i.1)))
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD))
    (hr : s.regs .int [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context)) :
    ∃ t, stepStmts (epilogue BM BD c.p.headDim reciprocal) s = some t ∧
      (∀ i, active c BM BD i = Bool.true →
        (∀ x y, active c BM BD x = Bool.true → active c BM BD y = Bool.true →
          address c BM BD x = address c BM BD y → x = y) →
        t.readMem c.p.Out (address c BM BD i) = a i/l i.1) ∧
      (∀ region offset, (region ≠ c.p.Out ∨ ∀ i, active c BM BD i = Bool.true → offset ≠ address c BM BD i) →
        t.mem region offset = s.mem region offset) := by
  let normalized := fun i : TileIndex [BM,BD] => a i/l i.1
  let sn := s.setReg "acc" .real [BM,BD] (finite normalized)
  have hn := RealEpilogue.normalize_executes s BM BD Bool.false a l ha hl
  have hstore := store_executes sn c BM BD normalized (by simp [sn,hp])
    (by simp [sn,hr]) (by simp [sn,hctx]) (by simp [sn])
  refine ⟨writeOutput sn c BM BD normalized,?_,?_,?_⟩
  · change stepStmts [RealSource.normalization BM BD Bool.false,store BM BD c.p.headDim] s = _
    rw [stepStmts.cons_some hn,stepStmts.cons_some hstore,stepStmts.nil]
  · intro i hi hinj
    exact RealEpilogue.readback sn (outputConfig c) c.p.Out BM BD normalized i hi hinj
  · intro region offset h
    exact RealEpilogue.frame sn (outputConfig c) c.p.Out BM BD normalized region offset h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Output
