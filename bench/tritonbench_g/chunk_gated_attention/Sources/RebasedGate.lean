import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedAccess
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHGate

/-! Actual repaired gate execution. Changed pointer setup is proved here;
all scale, exp, broadcast, subtraction and pre-dot arithmetic use the existing
RealHAlgebra lemmas unchanged. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedGate
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHTiles RealHAlgebra RebasedAccess
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def keyState (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := RebasedAccess.loaded p l a c
  let s1 := u.setReg "p_g" .blockPtr [p.BK,p.BT] (keyPointer p l r.gate c)
  let s2 := s1.setReg "p_gn" .blockPtr [p.BK]
    (endpointPointer p l r.gate c p.keyHead p.keyTime p.keyFeature p.BK l.key)
  let s3 := s2.setReg "b_gn" .real [p.BK] (Tile.vec (fun i => some (x.keyEnd i)))
  let s4 := s3.setReg "b_h" .real [p.BK,p.BV] (stateTile p (scaled x Bool.true f))
  let s5 := s4.setReg "b_g" .real [p.BK,p.BT] (Tile.mat (fun i j => some (x.keyGate i j)))
  s5.setReg "b_k" .real [p.BK,p.BT] (Tile.mat (fun i j => some (gatedKey x Bool.true i j)))

theorem key_executes (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p) (u : BlockState)
    (hi : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hdata : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun i j => some ((RebasedAccess.loaded p l a c).key i j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) :
    stepStmts (RebasedProgram.gateKey p r) u = some (keyState p l r a c f u) := by
  let x := RebasedAccess.loaded p l a c
  simp only [RebasedProgram.gateKey,RealHProgram.gateKey,List.drop,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.gate c u hh hi ht))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.keyHead p.keyTime p.keyFeature p.BK l.key "i_k" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_load p l r.gate a.gate c
    p.keyHead p.keyTime p.keyFeature p.BK l.key "p_gn" _ (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (scale_key p x f _
    (by simp [hcarry]) (by simp [x,RebasedAccess.loaded,keyIndex])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.gate a.gate c "p_g" _
    (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_key p x _
    (by simpa using hdata) (by simp [x,RebasedAccess.loaded])
    (by simp [x,RebasedAccess.loaded,keyIndex]))),stepStmts.nil]
  rfl

def valueState (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := RebasedAccess.loaded p l a c
  let s1 := u.setReg "p_g" .blockPtr [p.BT,p.BV] (valuePointer p l r.gate c)
  let s2 := s1.setReg "p_gn" .blockPtr [p.BV]
    (endpointPointer p l r.gate c p.valueHead p.valueTime p.valueFeature p.BV l.value)
  let s3 := s2.setReg "b_gn" .real [p.BV] (Tile.vec (fun i => some (x.valueEnd i)))
  let s4 := s3.setReg "b_h" .real [p.BK,p.BV] (stateTile p (scaled x Bool.false f))
  let s5 := s4.setReg "b_g" .real [p.BT,p.BV] (Tile.mat (fun i j => some (x.valueGate i j)))
  s5.setReg "b_v" .real [p.BT,p.BV] (Tile.mat (fun i j => some (gatedValue x Bool.false i j)))

theorem value_executes (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p) (u : BlockState)
    (hi : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hdata : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun i j => some ((RebasedAccess.loaded p l a c).value i j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) :
    stepStmts (RebasedProgram.gateValue p r) u = some (valueState p l r a c f u) := by
  let x := RebasedAccess.loaded p l a c
  simp only [RebasedProgram.gateValue,RealHProgram.gateValue,List.drop,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.gate c u hh hi ht))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.valueHead p.valueTime p.valueFeature p.BV l.value "i_v" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_load p l r.gate a.gate c
    p.valueHead p.valueTime p.valueFeature p.BV l.value "p_gn" _ (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (scale_value p x f _
    (by simp [hcarry]) (by simp [x,RebasedAccess.loaded,valueIndex])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p l r.gate a.gate c "p_g" _
    (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_value p x _
    (by simpa using hdata) (by simp [x,RebasedAccess.loaded])
    (by simp [x,RebasedAccess.loaded,valueIndex]))),stepStmts.nil]
  rfl

def output (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (f : StateValue p) (u : BlockState) : BlockState :=
  if gateK then keyState p l r a c f u else valueState p l r a c f u

theorem result (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (f : StateValue p) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun e t => some ((RebasedAccess.loaded p l a c).key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun t j => some ((RebasedAccess.loaded p l a c).value t j)))) :
    RealHGate.Result p (RebasedAccess.loaded p l a c) gateK f u (output p l r a c gateK f u) := by
  cases gateK <;> constructor
  all_goals first | rfl | simp [output,keyState,valueState,gatedKey,gatedValue,hk,hv]

theorem executes (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (f : StateValue p) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hk : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun e t => some ((RebasedAccess.loaded p l a c).key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun t j => some ((RebasedAccess.loaded p l a c).value t j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) :
    stepStmt (.ifThenElse (.constBool gateK) (RebasedProgram.gateKey p r) (RebasedProgram.gateValue p r)) u =
      some (output p l r a c gateK f u) := by
  cases gateK
  · simpa only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,
      Tile.scalar,Bool.false_eq_true,↓reduceIte,output] using
      value_executes p l r a c f u hiv hh ht hcarry hv hm
  · simpa only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,
      Tile.scalar,↓reduceIte,output] using
      key_executes p l r a c f u hik hh ht hcarry hk hm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedGate
