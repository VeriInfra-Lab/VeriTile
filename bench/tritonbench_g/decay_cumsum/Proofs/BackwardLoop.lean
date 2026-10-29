import bench.tritonbench_g.decay_cumsum.Proofs.RegisterFrame
import bench.tritonbench_g.decay_cumsum.Proofs.RealReference
open VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardLoop

def dead (n : RegName) : Prop := n = "j" ∨ n = "__rev_j" ∨ n = "offs"
instance (n : RegName) : Decidable (dead n) := by unfold dead; infer_instance

def norm (s : BlockState) : BlockState :=
  { s with regs := fun d sh n => if dead n then none else s.regs d sh n }

@[simp] theorem norm_idem (s : BlockState) : norm (norm s) = norm s := by
  apply BlockState.ext
  · intros; rfl
  · intro d sh n
    by_cases h : dead n <;> simp [norm,h]
  · intros; rfl
  · intros; rfl
  · intros; rfl

@[simp] theorem norm_set_dead (s : BlockState) (n : RegName) (d : TileDType)
    (sh : TileShape) (v : Tile d sh) (h : dead n) :
    norm (s.setReg n d sh v) = norm s := by
  apply BlockState.ext
  · intros; rfl
  · intro d' sh' n'
    by_cases hn : n'=n
    · subst n'; simp [norm,h]
    · simp [norm,BlockState.setReg,hn]
  · intros; rfl
  · intros; rfl
  · intros; rfl

theorem norm_set_live (s : BlockState) (n : RegName) (d : TileDType)
    (sh : TileShape) (v : Tile d sh) (h : ¬dead n) :
    (norm s).setReg n d sh v = norm (s.setReg n d sh v) := by
  apply BlockState.ext
  · intros; rfl
  · intro d' sh' n'
    by_cases hn : n'=n
    · subst n'; simp [norm,BlockState.setReg,h]
    · simp [norm,BlockState.setReg,hn]
  · intros; rfl
  · intros; rfl
  · intros; rfl

 theorem norm_write (s : BlockState) (d : TileDType) (r : RegionName) (o : Nat) (x : TileCarrier d) :
    (norm s).writeMemTyped d r o x = norm (s.writeMemTyped d r o x) := by
  cases d <;> rfl



set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def sourceBody (DK BT BK : Nat) : List Stmt :=
  [.assign .int [] "j" (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_j")),
   .assign .real [BK] "_g"
      (.load .real (.ptr (.ref .ptr [BK] "p_g"))
        (.maskOther (.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
   .ifThen (.eq .int .nil (.ref .int [] "j")
      (.castNatToInt (.sub .nat .nil (.constNat BT) (.constNat 1))))
      [.assign .real [BK] "last_g" (.ref .real [BK] "_g")]] ++
    (RealReference.bwdIterBodyG DK BT BK).drop 3

def tail (DK BT BK : Nat) : List Stmt := (RealReference.bwdIterBodyG DK BT BK).drop 3

theorem tail_transports (DK BT BK : Nat) (s : BlockState) :
    stepStmts (tail DK BT BK) (norm s) = (stepStmts (tail DK BT BK) s).map norm := by
  apply RegisterFrame.list
  intro st hs
  simp only [tail,RealReference.bwdIterBodyG,List.drop,List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first
    | apply RegisterFrame.assign
      · intro u; simp [norm,dead,evalOp,BlockState.readMem]
      · intro u v; exact norm_set_live _ _ _ _ _ (by decide)
    | apply RegisterFrame.store_ptr_mask
      · intro u; simp [norm,dead,evalOp,BlockState.readMem]
      · intro u; simp [norm,dead,evalOp,BlockState.readMem]
      · intro u; simp [norm,dead,evalOp,BlockState.readMem]
      · intro u r o x; exact norm_write _ _ _ _ _

theorem map_bind (f : BlockState → Option BlockState)
    (hf : ∀ s, f (norm s) = (f s).map norm) (x : Option BlockState) :
    (x.bind f).map norm = (x.map norm).bind f := by
  cases x with
  | none => rfl
  | some s => exact (hf s).symm

@[simp] theorem norm_set_live' (s : BlockState) (n : RegName) (d : TileDType)
    (sh : TileShape) (v : Tile d sh) (h : ¬dead n) :
    norm (s.setReg n d sh v) = (norm s).setReg n d sh v := (norm_set_live s n d sh v h).symm


def loadG (BK : Nat) : Op .real [BK] :=
  .load .real (.ptr (.ref .ptr [BK] "p_g"))
    (.maskOther (.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))

theorem load_norm (BK : Nat) (s : BlockState) : evalOp (loadG BK) (norm s) = evalOp (loadG BK) s := by
  simp [loadG,evalOp,norm,dead,BlockState.readMem]

theorem load_dead (BK : Nat) (s : BlockState) (n : RegName) (d : TileDType)
    (sh : TileShape) (v : Tile d sh) (h : dead n) :
    evalOp (loadG BK) (s.setReg n d sh v) = evalOp (loadG BK) s := by
  rcases h with rfl | rfl | rfl <;> simp [loadG,evalOp]

theorem head_nat (DK BT BK m : Nat) (hm : m<BT) (s : BlockState) :
    (stepStmts ((RealReference.bwdIterBodyG DK BT BK).take 3)
      (s.setReg "__rev_j" .nat [] (Tile.scalar m))).map norm =
    (evalOp (loadG BK) s).map (fun g =>
      let v := (norm s).setReg "_g" .real [BK] g
      if m=0 then v.setReg "last_g" .real [BK] g else v) := by
  change (stepStmts [
    .assign .nat [] "j" (.sub .nat .nil
      (.sub .nat .nil (.constNat BT) (.constNat 1))
      (.mul .nat .nil (.ref .nat [] "__rev_j") (.constNat 1))),
    .assign .real [BK] "_g" (loadG BK),
    .ifThen (.eq .nat .nil (.ref .nat [] "j") (.sub .nat .nil (.constNat BT) (.constNat 1)))
      [.assign .real [BK] "last_g" (.ref .real [BK] "_g")]]
    (s.setReg "__rev_j" .nat [] (Tile.scalar m))).map norm = _
  simp only [stepStmts,stepStmt,evalOp,BlockState.setReg_same,Option.bind_some,
    Tile.bop,Tile.cop,Tile.scalar_data,NumericDType.sub,NumericDType.mul,
    ComparableDType.eq,Nat.mul_one,bind,pure,Option.bind_some]
  simp only [load_dead (n := "j") (h := by decide),
    load_dead (n := "__rev_j") (h := by decide)]
  cases hg : evalOp (loadG BK) s with
  | none => rfl
  | some g =>
    have hc : BT - 1 - m = BT - 1 ↔ m = 0 := by omega
    by_cases hz : m=0
    all_goals simp [-Tile.scalar_eta,hz,BlockState.setReg_ne_name,Tile.scalar,Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
      hc,norm_set_live',norm_set_dead,dead]


theorem head_int (DK BT BK m : Nat) (hm : m<BT) (s : BlockState) :
    (stepStmts ((sourceBody DK BT BK).take 3)
      (s.setReg "__rev_j" .int [] (Tile.scalar (1-(BT:Int)+m)))).map norm =
    (evalOp (loadG BK) s).map (fun g =>
      let v := (norm s).setReg "_g" .real [BK] g
      if m=0 then v.setReg "last_g" .real [BK] g else v) := by
  change (stepStmts [
    .assign .int [] "j" (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_j")),
    .assign .real [BK] "_g" (loadG BK),
    .ifThen (.eq .int .nil (.ref .int [] "j")
      (.castNatToInt (.sub .nat .nil (.constNat BT) (.constNat 1))))
      [.assign .real [BK] "last_g" (.ref .real [BK] "_g")]]
    (s.setReg "__rev_j" .int [] (Tile.scalar (1-(BT:Int)+m)))).map norm = _
  simp only [stepStmts,stepStmt,evalOp,BlockState.setReg_same,Option.bind_some,
    Tile.bop,Tile.cop,Tile.uop,Tile.scalar_data,NumericDType.sub,NumericDType.mul,
    ComparableDType.eq,Nat.mul_one,bind,pure,Option.bind_some]
  simp only [load_dead (n := "j") (h := by decide),
    load_dead (n := "__rev_j") (h := by decide)]
  cases hg : evalOp (loadG BK) s with
  | none => rfl
  | some g =>
    have hn : ((BT-1:Nat):Int) = (BT:Int)-1 := by omega
    have hc : (0: Int)-(1-(BT:Int)+m) = (BT-1:Nat) ↔ m = 0 := by omega
    by_cases hz : m=0
    all_goals simp [-Tile.scalar_eta,hz,BlockState.setReg_ne_name,Tile.scalar,
      Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
      hc,hn,norm_set_live',norm_set_dead,dead]


theorem body (DK BT BK m : Nat) (hm : m<BT) (s t : BlockState) (he : norm s = norm t) :
    (stepStmts (RealReference.bwdIterBodyG DK BT BK)
      (s.setReg "__rev_j" .nat [] (Tile.scalar m))).map norm =
    (stepStmts (sourceBody DK BT BK)
      (t.setReg "__rev_j" .int [] (Tile.scalar (1-(BT:Int)+m)))).map norm := by
  have hg : evalOp (loadG BK) s = evalOp (loadG BK) t := by
    rw [← load_norm BK s,he,load_norm]
  have hh : (stepStmts ((RealReference.bwdIterBodyG DK BT BK).take 3)
      (s.setReg "__rev_j" .nat [] (Tile.scalar m))).map norm =
    (stepStmts ((sourceBody DK BT BK).take 3)
      (t.setReg "__rev_j" .int [] (Tile.scalar (1-(BT:Int)+m)))).map norm := by
    rw [head_nat _ _ _ _ hm,head_int _ _ _ _ hm,he,hg]
  change (stepStmts ((RealReference.bwdIterBodyG DK BT BK).take 3 ++ tail DK BT BK) _).map norm =
    (stepStmts ((sourceBody DK BT BK).take 3 ++ tail DK BT BK) _).map norm
  rw [ProgramRewrite.append,ProgramRewrite.append,
    map_bind _ (tail_transports DK BT BK),map_bind _ (tail_transports DK BT BK),hh]

/-- Exact Nat/Int iteration bridge after removing only dead loop/cache registers.
The source visits BT-1,...,0 and keeps every final pointer decrement. -/
theorem range (DK BT BK m : Nat) (s t : BlockState) (he : norm s = norm t) :
    (stepForRangeAux "__rev_j" m BT 1 (RealReference.bwdIterBodyG DK BT BK) s).map norm =
    (stepForRangeIntAux "__rev_j" (1-(BT:Int)+m) 1 1 (sourceBody DK BT BK) t).map norm := by
  by_cases hm : m<BT
  · rw [stepForRangeAux.step_lt (by decide) hm,
      stepForRangeIntAux.step_lt (by decide) (show 1-(BT:Int)+m<1 by omega)]
    have hb := body DK BT BK m hm s t he
    cases hs : stepStmts (RealReference.bwdIterBodyG DK BT BK)
        (s.setReg "__rev_j" .nat [] (Tile.scalar m)) with
    | none =>
      cases ht : stepStmts (sourceBody DK BT BK)
          (t.setReg "__rev_j" .int [] (Tile.scalar (1-(BT:Int)+m))) <;> simp_all
    | some s' =>
      cases ht : stepStmts (sourceBody DK BT BK)
          (t.setReg "__rev_j" .int [] (Tile.scalar (1-(BT:Int)+m))) with
      | none => simp [hs,ht] at hb
      | some t' =>
        have he' : norm s' = norm t' := by simpa [hs,ht] using hb
        simpa only [Option.bind_some,Nat.cast_add,Nat.cast_one,Int.add_assoc] using
          range DK BT BK (m+1) s' t' he'
  · rw [stepForRangeAux.step_ge (by decide) (by omega),
      stepForRangeIntAux.step_ge (by decide) (show 1≤1-(BT:Int)+m by omega)]
    exact congrArg some he
termination_by BT-m

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardLoop
