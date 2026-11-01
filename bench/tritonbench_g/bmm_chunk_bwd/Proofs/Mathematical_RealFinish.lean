import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealLoop
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealStore

/-! Assemble the exact post-loop source and then the full kernel execution.
The loop state is constructed from initial memory by RealLoop; the residual
branch and final masked store execute in the current canonical source. These execution
lemmas are internal preparation for the independent formula and flat IO proof. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealFinish
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def rows (BM pm : Nat) : Fin BM → Nat := fun i => pm * BM + i.val
def columns (BN pn : Nat) : Fin BN → Nat := fun j => pn * BN + j.val

def refreshCode (BM BN : Nat) : List Stmt :=
  [.assign .int [BM] "offs_m" (RealResidual.rowOffsets BM),
   .assign .int [BN] "offs_n" (RealResidual.columnOffsets BN)]

def refreshed (s : BlockState) (BM BN pm pn : Nat) : BlockState :=
  (s.setReg "offs_m" .int [BM] (Tile.vec fun i => (rows BM pm i : Int))).setReg
    "offs_n" .int [BN] (Tile.vec fun i => (columns BN pn i : Int))

theorem refresh_executes (s : BlockState) (BM BN pm pn : Nat)
    (hm : s.regs .int [] "pid_m" = some (Tile.scalar (pm : Int)))
    (hn : s.regs .int [] "pid_n" = some (Tile.scalar (pn : Int))) :
    stepStmts (refreshCode BM BN) s = some (refreshed s BM BN pm pn) := by
  simp [refreshCode, RealResidual.rowOffsets, RealResidual.columnOffsets,
    stepStmts, stepStmt, evalOp.eq_def, hm, hn, bind, Option.bind]
  rfl

def afterResidual (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (length : Int) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  if p.hasResidual then RealResidual.residualResult s BM BN p pb pc ph length rs ns carry else s

theorem residual_branch (swapped : Bool) (s : BlockState) (BM BN : Nat)
    (p : RealProgram.Parameters) (pb pc ph : Nat) (length : Int)
    (rs : Fin BM → Nat) (ns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ)
    (hb : s.regs .int [] "pid_b" = some (Tile.scalar (pb : Int)))
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hh : s.regs .int [] "pid_h" = some (Tile.scalar (ph : Int)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rs i : Int)))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun i => (ns i : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmt (.ifThen (.constBool p.hasResidual) (RealResidual.residual swapped BM BN p)) s =
      some (afterResidual s BM BN p pb pc ph length rs ns carry) := by
  have he := RealResidual.residual_executes swapped s BM BN p pb pc ph length rs ns carry hb hc hh hr hn hl ha
  rw [stepStmt]
  simp only [evalOp.eq_def, Tile.scalar, bind, Option.bind_some]
  cases hf : p.hasResidual <;> simp [afterResidual, hf, he]

def outputCode (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .real [BM, BN] "db" (.ref .real [BM, BN] "acc"),
   .assign .ptr [] "db_ptr" (RealResidual.basePointer p.Db p.chunk p.oBatch p.oRow p.oHead),
   .assign .ptr [BM, BN] "db_ptrs" (RealResidual.pointers BM BN "db_ptr" p.oRow p.oColumn)]

theorem source_split (swapped : Bool) (BM BN : Nat) (p : RealProgram.Parameters) :
    RealResidual.code swapped BM BN p = refreshCode BM BN ++
      [.ifThen (.constBool p.hasResidual) (RealResidual.residual swapped BM BN p)] ++
      outputCode BM BN p ++ [RealStore.code BM BN p.K] := by
  rfl

def outputPrepared (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (values : Fin BM → Fin BN → ℝ) : BlockState :=
  let base := RealResidual.baseValue p.chunk p.oBatch p.oRow p.oHead pb pc ph
  let ptrs := RealResidual.pointerValues p.Db BM BN base p.oRow p.oColumn rs ns
  let t := s.setReg "db" .real [BM, BN] (Tile.mat fun i j => some (values i j))
  let t := t.setReg "db_ptr" .ptr [] (Tile.scalar (p.Db, base))
  t.setReg "db_ptrs" .ptr [BM, BN] ptrs

theorem output_preparation (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (hb : s.regs .int [] "pid_b" = some (Tile.scalar (pb : Int)))
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hh : s.regs .int [] "pid_h" = some (Tile.scalar (ph : Int)))
    (hr : s.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rs i : Int)))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec fun i => (ns i : Int)))
    (hv : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (values i j))) :
    stepStmts (outputCode BM BN p) s = some (outputPrepared s BM BN p pb pc ph rs ns values) := by
  let sd := s.setReg "db" .real [BM, BN] (Tile.mat fun i j => some (values i j))
  let base := RealResidual.baseValue p.chunk p.oBatch p.oRow p.oHead pb pc ph
  let sb := sd.setReg "db_ptr" .ptr [] (Tile.scalar (p.Db, base))
  have hbval : evalOp (RealResidual.basePointer p.Db p.chunk p.oBatch p.oRow p.oHead) sd =
      some (Tile.scalar (p.Db, base)) := by
    apply RealResidual.base_eval sd p.Db p.chunk p.oBatch p.oRow p.oHead pb pc ph
    all_goals simpa [sd] using (by assumption)
  have hpval : evalOp (RealResidual.pointers BM BN "db_ptr" p.oRow p.oColumn) sb =
      some (RealResidual.pointerValues p.Db BM BN base p.oRow p.oColumn rs ns) := by
    apply RealResidual.pointers_eval sb p.Db BM BN base p.oRow p.oColumn "db_ptr" rs ns
    · simp [sb]
    · simpa [sb, sd] using hr
    · simpa [sb, sd] using hn
  dsimp only [sd, sb, base] at hbval hpval
  simp only [outputCode, stepStmts, stepStmt, evalOp_ref, hv, hbval, hpval,
    bind, Option.bind_some]
  rfl


def values (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (length : Int) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) (i : Fin BM) (j : Fin BN) : ℝ :=
  if p.hasResidual then carry i j + RealResidual.residualValue s
    (RealResidual.pointerValues p.Res BM BN
      (RealResidual.baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph) p.rRow p.rColumn rs ns)
    rs ns length.toNat p.K i j else carry i j

theorem after_residual_acc (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (length : Int) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ)
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    (afterResidual s BM BN p pb pc ph length rs ns carry).regs .real [BM, BN] "acc" =
      some (Tile.mat fun i j => some (values s BM BN p pb pc ph length rs ns carry i j)) := by
  cases hf : p.hasResidual <;> simp [afterResidual, values, hf, RealResidual.residualResult, ha]

theorem after_residual_fixed (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (length : Int) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["res_ptr", "res_ptrs", "__veritile_load_2", "res", "acc"]) :
    (afterResidual s BM BN p pb pc ph length rs ns carry).regs dtype shape name =
      s.regs dtype shape name := by
  cases hf : p.hasResidual
  · simp [afterResidual, hf]
  · simpa [afterResidual, hf] using
      RealResidual.residual_register_frame s BM BN p pb pc ph length rs ns carry dtype shape name hn

def result (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph pm pn : Nat) (length : Int) (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let r := refreshed s BM BN pm pn
  let v := values r BM BN p pb pc ph length (rows BM pm) (columns BN pn) carry
  let t := afterResidual r BM BN p pb pc ph length (rows BM pm) (columns BN pn) carry
  let u := outputPrepared t BM BN p pb pc ph (rows BM pm) (columns BN pn) v
  RealStore.write u p.Db BM BN (RealResidual.baseValue p.chunk p.oBatch p.oRow p.oHead pb pc ph)
    p.oRow p.oColumn length.toNat p.K (rows BM pm) (columns BN pn) v

theorem executes (swapped : Bool) (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph pm pn : Nat) (length : Int) (carry : Fin BM → Fin BN → ℝ)
    (hb : s.regs .int [] "pid_b" = some (Tile.scalar (pb : Int)))
    (hc : s.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)))
    (hh : s.regs .int [] "pid_h" = some (Tile.scalar (ph : Int)))
    (hm : s.regs .int [] "pid_m" = some (Tile.scalar (pm : Int)))
    (hn : s.regs .int [] "pid_n" = some (Tile.scalar (pn : Int)))
    (hl : s.regs .int [] "chunk_size_limit" = some (Tile.scalar length))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (RealResidual.code swapped BM BN p) s =
      some (result s BM BN p pb pc ph pm pn length carry) := by
  let r := refreshed s BM BN pm pn
  let v := values r BM BN p pb pc ph length (rows BM pm) (columns BN pn) carry
  let t := afterResidual r BM BN p pb pc ph length (rows BM pm) (columns BN pn) carry
  let u := outputPrepared t BM BN p pb pc ph (rows BM pm) (columns BN pn) v
  have hb' : r.regs .int [] "pid_b" = some (Tile.scalar (pb : Int)) := by simpa [r, refreshed] using hb
  have hc' : r.regs .int [] "pid_c" = some (Tile.scalar (pc : Int)) := by simpa [r, refreshed] using hc
  have hh' : r.regs .int [] "pid_h" = some (Tile.scalar (ph : Int)) := by simpa [r, refreshed] using hh
  have hl' : r.regs .int [] "chunk_size_limit" = some (Tile.scalar length) := by simpa [r, refreshed] using hl
  have ha' : r.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)) := by
    simpa [r, refreshed] using ha
  have hr' : r.regs .int [BM] "offs_m" = some (Tile.vec fun i => (rows BM pm i : Int)) := by simp [r, refreshed]
  have hn' : r.regs .int [BN] "offs_n" = some (Tile.vec fun i => (columns BN pn i : Int)) := by simp [r, refreshed]
  have hrefresh := refresh_executes s BM BN pm pn hm hn
  have hbranch := residual_branch swapped r BM BN p pb pc ph length (rows BM pm) (columns BN pn) carry
    hb' hc' hh' hr' hn' hl' ha'
  have hfixed (dtype : TileDType) (shape : TileShape) (name : RegName)
      (hnm : name ∉ ["res_ptr", "res_ptrs", "__veritile_load_2", "res", "acc"]) :
      t.regs dtype shape name = r.regs dtype shape name :=
    after_residual_fixed r BM BN p pb pc ph length (rows BM pm) (columns BN pn) carry dtype shape name hnm
  have hv : t.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (v i j)) :=
    after_residual_acc r BM BN p pb pc ph length (rows BM pm) (columns BN pn) carry ha'
  have htb := (hfixed .int [] "pid_b" (by decide)).trans hb'
  have htc := (hfixed .int [] "pid_c" (by decide)).trans hc'
  have hth := (hfixed .int [] "pid_h" (by decide)).trans hh'
  have htr := (hfixed .int [BM] "offs_m" (by decide)).trans hr'
  have htn := (hfixed .int [BN] "offs_n" (by decide)).trans hn'
  have htl := (hfixed .int [] "chunk_size_limit" (by decide)).trans hl'
  have hout := output_preparation t BM BN p pb pc ph (rows BM pm) (columns BN pn) v htb htc hth htr htn hv
  have hstore := RealStore.executes u p.Db BM BN
    (RealResidual.baseValue p.chunk p.oBatch p.oRow p.oHead pb pc ph) p.oRow p.oColumn length p.K
    (rows BM pm) (columns BN pn) v (by simp [u, outputPrepared]) (by simp [u, outputPrepared])
    (by simpa [u, outputPrepared] using htr) (by simpa [u, outputPrepared] using htn)
    (by simpa [u, outputPrepared] using htl)
  rw [source_split]
  simp only [List.append_assoc, List.cons_append, List.nil_append]
  rw [stepStmts.append_some hrefresh, stepStmts.cons_some hbranch,
    stepStmts.append_some hout, stepStmts.cons_some hstore]
  exact stepStmts.nil


/-- Assemble setup, the entire dynamic contraction, the optional residual
and the actual terminal store. The intermediate loop state is constructed by
the proof and is not an execution premise of this theorem. -/
theorem kernel_executes (swapped : Bool) (s : BlockState) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) :
    ∃ t, RealLoop.Holds s BM BN BCS p (RealMath.blocks (RealSetup.limit s p) BCS) t ∧
      stepStmts (RealProgram.kernel swapped BM BN BCS p).toAlgKernel.body s =
        some (result t BM BN p (s.pids 1) (RealSetup.chunk s p) (RealSetup.head s p)
          (RealSetup.tileM s BN p) (RealSetup.tileN s BN p) (RealSetup.limitInt s p)
          (RealLoop.carry s BM BN BCS p (RealMath.blocks (RealSetup.limit s p) BCS))) := by
  obtain ⟨t, hp, h⟩ := RealLoop.prefix_executes swapped s BM BN BCS p
  have hb : t.regs .int [] "pid_b" = some (Tile.scalar (s.pids 1 : Int)) := by
    have hv := h.fixed .int [] "pid_b" (by decide)
    simpa [RealSetup.result] using hv
  have hc : t.regs .int [] "pid_c" = some (Tile.scalar (RealSetup.chunk s p : Int)) := by
    have hv := h.fixed .int [] "pid_c" (by decide)
    simpa [RealSetup.result] using hv
  have hh : t.regs .int [] "pid_h" = some (Tile.scalar (RealSetup.head s p : Int)) := by
    have hv := h.fixed .int [] "pid_h" (by decide)
    simpa [RealSetup.result] using hv
  have hm : t.regs .int [] "pid_m" = some (Tile.scalar (RealSetup.tileM s BN p : Int)) := by
    have hv := h.fixed .int [] "pid_m" (by decide)
    simpa [RealSetup.result] using hv
  have hn : t.regs .int [] "pid_n" = some (Tile.scalar (RealSetup.tileN s BN p : Int)) := by
    have hv := h.fixed .int [] "pid_n" (by decide)
    simpa [RealSetup.result] using hv
  have hl : t.regs .int [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limitInt s p)) := by
    have hv := h.fixed .int [] "chunk_size_limit" (by decide)
    simpa [RealSetup.result] using hv
  refine ⟨t, h, ?_⟩
  rw [RealProgram.body_split, stepStmts.append_some hp, RealResidual.matches_source]
  exact executes swapped t BM BN p (s.pids 1) (RealSetup.chunk s p) (RealSetup.head s p)
    (RealSetup.tileM s BN p) (RealSetup.tileN s BN p) (RealSetup.limitInt s p)
    (RealLoop.carry s BM BN BCS p (RealMath.blocks (RealSetup.limit s p) BCS)) hb hc hh hm hn hl h.acc

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealFinish
