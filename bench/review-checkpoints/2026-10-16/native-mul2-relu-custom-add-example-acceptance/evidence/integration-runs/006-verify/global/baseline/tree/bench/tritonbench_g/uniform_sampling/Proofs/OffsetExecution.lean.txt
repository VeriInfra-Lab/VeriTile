import bench.tritonbench_g.uniform_sampling.Proofs.IndexExecution

/-! Exact execution of all four output-offset assignments, independent of
the random/value prefix. SourceDomain and WordArithmetic separately establish
that these natural-number indices agree with the source machine indices on
the launched signed-i32 domain.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.OffsetExecution
open VeriTile.Triton IndexExecution
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

def outputFn (pid B k : Nat) : TileIndex [B] → Nat :=
  fun i => (pid * 4 + k) * B + i.1.val

def firstOp (B : Nat) : Op .nat [B] :=
  .add .nat .scalarL
    (.mul .nat .nil (.mul .nat .nil (.programId 0) (.ref .nat [] "BLOCK_arg")) (.constNat 4))
    (.arange B)

def bumpOp (B : Nat) (name : RegName) : Op .nat [B] :=
  .add .nat .scalarR (.ref .nat [B] name) (.ref .nat [] "BLOCK_arg")

def body (B : Nat) : List Stmt := [
  .assign .nat [B] "off_0" (firstOp B), .assign .nat [B] "off_1" (bumpOp B "off_0"),
  .assign .nat [B] "off_2" (bumpOp B "off_1"), .assign .nat [B] "off_3" (bumpOp B "off_2")]

def offsetState (pid B : Nat) (s : BlockState) : BlockState :=
  let s := s.setReg "off_0" .nat [B] ⟨outputFn pid B 0⟩
  let s := s.setReg "off_1" .nat [B] ⟨outputFn pid B 1⟩
  let s := s.setReg "off_2" .nat [B] ⟨outputFn pid B 2⟩
  s.setReg "off_3" .nat [B] ⟨outputFn pid B 3⟩

theorem source_slice (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) :
    ((Algorithm.kernel swapped y N seed offset lower upper B).body.drop 15).take 4 = body B := rfl

theorem first_eval (B : Nat) (s : BlockState)
    (hb : s.regs .nat [] "BLOCK_arg" = some (Tile.scalar B)) :
    evalOp (firstOp B) s = some ⟨outputFn (s.pids 0) B 0⟩ := by
  simp [-Tile.scalar_eta, firstOp, evalOp.eq_def, hb, Tile.bop, Tile.scalar, Tile.vec,
    NumericDType.add, NumericDType.mul]
  funext i
  simp only [outputFn]
  ring

theorem bump_eval (pid B k : Nat) (name : RegName) (s : BlockState)
    (hb : s.regs .nat [] "BLOCK_arg" = some (Tile.scalar B))
    (ho : s.regs .nat [B] name = some ⟨outputFn pid B k⟩) :
    evalOp (bumpOp B name) s = some ⟨outputFn pid B (k + 1)⟩ := by
  simp [-Tile.scalar_eta, bumpOp, evalOp.eq_def, hb, ho, Tile.bop, Tile.scalar, NumericDType.add]
  funext i
  simp only [outputFn]
  ring

private theorem four_steps {a b c d : Stmt} {s0 s1 s2 s3 s4 : BlockState}
    (ha : stepStmt a s0 = some s1) (hb : stepStmt b s1 = some s2)
    (hc : stepStmt c s2 = some s3) (hd : stepStmt d s3 = some s4) :
    stepStmts [a, b, c, d] s0 = some s4 := by
  simp only [stepStmts, ha, hb, hc, hd]

theorem offsets_exec (B : Nat) (s : BlockState)
    (hb : s.regs .nat [] "BLOCK_arg" = some (Tile.scalar B)) :
    stepStmts (body B) s = some (offsetState (s.pids 0) B s) := by
  let t0 := s.setReg "off_0" .nat [B] ⟨outputFn (s.pids 0) B 0⟩
  let t1 := t0.setReg "off_1" .nat [B] ⟨outputFn (s.pids 0) B 1⟩
  let t2 := t1.setReg "off_2" .nat [B] ⟨outputFn (s.pids 0) B 2⟩
  have h0 := assign_step_of_eval "off_0" _ _ _ (first_eval B s hb)
  have h1 := assign_step_of_eval "off_1" _ _ _
    (bump_eval (s.pids 0) B 0 "off_0" t0
      (by simpa [t0] using hb) (by simp [t0]))
  have h2 := assign_step_of_eval "off_2" _ _ _
    (bump_eval (s.pids 0) B 1 "off_1" t1
      (by simpa [t1, t0] using hb) (by simp [t1]))
  have h3 := assign_step_of_eval "off_3" _ _ _
    (bump_eval (s.pids 0) B 2 "off_2" t2
      (by simpa [t2, t1, t0] using hb) (by simp [t2]))
  exact four_steps h0 h1 h2 h3

theorem offsetState_frame (pid B : Nat) (s : BlockState) :
    (offsetState pid B s).mem = s.mem ∧ (offsetState pid B s).pids = s.pids ∧
    (offsetState pid B s).undef = s.undef ∧ (offsetState pid B s).numPids = s.numPids :=
  ⟨rfl, rfl, rfl, rfl⟩

theorem outputFn_spec (pid B : Nat) (f : Fin 4) (i : TileIndex [B]) :
    outputFn pid B f.val i = SourceDomain.outputIndex pid B f i.1 := rfl

#axiomsClean source_slice
#axiomsClean first_eval
#axiomsClean bump_eval
#axiomsClean offsets_exec
#axiomsClean offsetState_frame
#axiomsClean outputFn_spec

end
end VeriTile.Bench.TritonBenchG.UniformSampling.OffsetExecution
