import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.Geometry
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealIndexBridge
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkCumsumKernel
open VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def sourceFor (v : Bool) (input output : RegionName) (T BT : Nat) : ComputeKernel :=
  if v then sumLeftKernel .fp32 .fp32 input output T BT else repairedKernel .fp32 .fp32 input output T BT

def projected (v : Bool) (input output : RegionName) (T BT : Nat) : Kernel :=
  (sourceFor v input output T BT).eraseDType.toAlgKernel

def bodyFor (v : Bool) (input output : RegionName) (T BT : Nat) : List Stmt :=
  match (projected v input output T BT).body[2]? with
  | some (Stmt.forRangeInt _ _ _ _ code) => code
  | _ => []

theorem program_body (v : Bool) (input output : RegionName) (T BT : Nat) :
    (projected v input output T BT).body=prelude ++
      [.forRangeInt "i_t" (.constInt 0) (.constInt ((T+BT-1)/BT : Nat)) (.constNat 1)
        (bodyFor v input output T BT)] := by
  cases v
  all_goals
    simp only [prelude,bodyFor,projected,sourceFor,repairedKernel,sumLeftKernel,Bool.false_eq_true,↓reduceIte,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
      MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
      ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
      List.map_cons,List.map_nil,eraseDType_float]
  all_goals
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComparableDType.eraseDType]
    rfl


private def carry (v : Bool) : Stmt :=
  .assign .real [] "b_z" (if v then .add .real .nil (.ref .real [] "b_zz") (.ref .real [] "b_z")
    else .add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz"))

private theorem carry_equal (s : BlockState) :
    stepStmts [carry Bool.true] s=stepStmts [carry Bool.false] s := by
  have commute (a b : WithBot ℝ) : WithBot.realAdd a b=WithBot.realAdd b a := by
    cases a <;> cases b
    · rfl
    · rfl
    · rfl
    · simp [WithBot.realAdd,add_comm]
  simp only [carry,Bool.false_eq_true,↓reduceIte,stepStmts,stepStmt,evalOp_add]
  cases ha : evalOp (.ref .real [] "b_z") s <;> cases hb : evalOp (.ref .real [] "b_zz") s
  all_goals simp only [bind,Option.bind_none,Option.bind_some]
  rename_i a b
  have eq : Tile.bop WithBot.realAdd .nil a b=Tile.bop WithBot.realAdd .nil b a := by
    ext i
    exact commute _ _
  simpa only [NumericDType.add] using congrArg (fun value => some (s.setReg "b_z" .real [] value)) eq.symm

private theorem body_view (v : Bool) (input output : RegionName) (T BT : Nat) :
    bodyFor v input output T BT=
      (actualBody input output T BT).take 6 ++ carry v :: (actualBody input output T BT).drop 7 := by
  cases v
  all_goals
    simp only [bodyFor,projected,sourceFor,repairedKernel,sumLeftKernel,Bool.false_eq_true,↓reduceIte,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
      MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
      ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
      List.map_cons,List.map_nil,eraseDType_float]
  all_goals
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,ComparableDType.eraseDType]
    rfl

theorem body_equal (v : Bool) (input output : RegionName) (T BT : Nat) (s : BlockState) :
    stepStmts (bodyFor v input output T BT) s=stepStmts (actualBody input output T BT) s := by
  rw [body_view]
  have original : actualBody input output T BT =
      (actualBody input output T BT).take 6 ++ carry Bool.false :: (actualBody input output T BT).drop 7 := rfl
  conv_rhs => rw [original]
  cases v
  · rfl
  · simpa only [List.singleton_append,List.append_assoc] using
      ProgramRewrite.context ((actualBody input output T BT).take 6)
        ((actualBody input output T BT).drop 7) [carry Bool.true] [carry Bool.false] carry_equal s

def Outside (s : BlockState) (output : RegionName) (T : Nat) (r : RegionName) (o : Nat) :=
  r≠output ∨ ∀ i,i<T → o≠s.pids 0*T+i

/-- Ghost state carries only the independent chunk invariant and one-step
arithmetic proof; the sole complete execution here is the repaired signed loop. -/
theorem loop_actual (v : Bool) (input output : RegionName) (T BT : Nat) (s : BlockState)
    (separate : output≠input) (cur stop : Nat) (within : cur≤stop) (a b : BlockState)
    (related : Rel a b) (invariant : surfaceInv s input output T BT cur b)
    (frame : ∀ r o,Outside s output T r o → b.mem r o=s.mem r o) :
    ∃ x y,stepForRangeIntAux "i_t" cur stop 1 (bodyFor v input output T BT) a=some x ∧
      Rel x y ∧ surfaceInv s input output T BT stop y ∧
      (∀ r o,Outside s output T r o → y.mem r o=s.mem r o) := by
  by_cases hlt : cur<stop
  · have ready : Ready (s.pids 0) b := ⟨invariant.2.2.1,by rw [invariant.2.2.2.1];rfl⟩
    obtain ⟨midA,midB,ha,hb,hr,_⟩ := RealIndexBridge.body input output T BT (s.pids 0) cur a b related ready
    obtain ⟨ghost,hg,hi⟩ := surface_step input output T BT s separate cur b invariant
    change stepStmts (ccLoopBody input output T BT)
      (b.setReg "i_t" .nat [] (Tile.scalar cur))=some ghost at hg
    have same : ghost=midB := Option.some.inj (hg.symm.trans hb)
    subst ghost
    have nextFrame : ∀ r o,Outside s output T r o → midB.mem r o=s.mem r o := by
      intro r o outside
      exact (chunk_body_frame input output T BT s cur b midB invariant.2.2.1 hb r o outside).trans (frame r o outside)
    obtain ⟨x,y,hx,hrel,hy,hframe⟩  := loop_actual v input output T BT s separate
      (cur+1) stop (by omega) midA midB hr hi nextFrame
    have actualStep : stepStmts (bodyFor v input output T BT)
        (a.setReg "i_t" .int [] (Tile.scalar (cur:Int)))=some midA := by
      rw [body_equal];exact ha
    refine ⟨x,y,?_,hrel,hy,hframe⟩
    rw [stepForRangeIntAux,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hlt),actualStep]
    simpa using hx
  · have done : cur=stop := by omega
    subst cur
    refine ⟨a,b,?_,related,invariant,frame⟩
    rw [stepForRangeIntAux,if_neg (by decide : ¬(1:Nat)=0),if_neg (by simp)]
termination_by stop-cur
 decreasing_by omega

theorem complete_blocks (v : Bool) (input output : RegionName) (T BT : Nat) (s : BlockState)
    (separate : output≠input) (positive : 0<BT) :
    ∃ t,exec (sourceFor v input output T BT).eraseDType.toAlgKernel s=some t ∧
      (∀ i,i<T → t.readMem output (s.pids 0*T+i)=
        ∑ j ∈ (Finset.range T).filter (fun j => j ≤ i), s.readMem input (s.pids 0*T+j)) ∧
      (∀ r o,Outside s output T r o → t.mem r o=s.mem r o) := by
  let initial := (s.setReg "i_bh" .nat [] (Tile.scalar (s.pids 0))).setReg "b_z" .real []
    (Tile.scalar (some (0:ℝ)))
  have pre : stepStmts prelude s=some initial := by
    simp [prelude,initial,stepStmts,stepStmt,evalOp,Tile.scalar,-Tile.scalar_eta]
  have inv : surfaceInv s input output T BT 0 initial := by
    refine ⟨rfl,fun _=>rfl,?_,?_,?_,rfl⟩
    · simp [initial]
    · simp [initial,surfaceCarry]
    · intro i bound _;omega
  obtain ⟨x,y,hx,hr,hy,hf⟩  := loop_actual v input output T BT s separate 0 ((T+BT-1)/BT)
    (Nat.zero_le _) initial initial ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩ inv (fun _ _ _=>rfl)
  simp only [Nat.cast_zero] at hx
  have covered : T≤((T+BT-1)/BT)*BT := by
    have div := Nat.div_add_mod (T+BT-1) BT
    have rem := Nat.mod_lt (T+BT-1) positive
    rw [Nat.mul_comm] at div
    omega
  refine ⟨x,?_,?_,?_⟩
  · change stepStmts (projected v input output T BT).body s=some x
    rw [program_body,stepStmts.append_some pre]
    simp only [stepStmts,stepStmt,evalOp.eq_def,bind,pure,Option.bind,Tile.scalar,hx]
  · intro i hi
    have read := hy.2.2.2.2.1 i (lt_of_lt_of_le hi covered) hi
    simpa only [BlockState.readMem,hr.1,surfaceClosed] using read
  · intro r o outside
    exact (congrFun (congrFun hr.1 r) o).trans (hf r o outside)

/-- The public Real result executes the repaired signed source directly.
The proof uses only chunk-level arithmetic and frame facts as ghost invariants;
there is no second whole-kernel execution in its hypotheses or conclusion. -/
theorem complete (input output : RegionName) (T : Nat) (cfg : Config)
    (s : BlockState) (separate : output≠input) :
    ∃ t,exec (repairedKernel .fp32 .fp32 input output T cfg.block).eraseDType.toAlgKernel s=some t ∧
      (∀ i,i<T → t.readMem output (s.pids 0*T+i)=
        ∑ j ∈ (Finset.range T).filter (fun j => j ≤ i), s.readMem input (s.pids 0*T+j)) ∧
      (∀ r o,(r≠output ∨ ∀ i,i<T → o≠s.pids 0*T+i) → t.mem r o=s.mem r o) :=
  complete_blocks Bool.false input output T cfg.block s separate cfg.positive

#axiomsClean complete
#axiomsClean complete_blocks
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
