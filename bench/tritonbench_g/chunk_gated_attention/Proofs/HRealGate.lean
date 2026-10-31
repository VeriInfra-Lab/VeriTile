import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealIterationParts
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHGate
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.Gate
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHTiles RealHAlgebra RebasedAccess
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def keyState (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u : BlockState) : BlockState :=
  let x := RebasedAccess.loaded p.toLayout l a c
  let s1 := u.setReg "p_g" .blockPtr [p.BK,p.BT] (keyPointer p.toLayout l r.gate c)
  let s2 := s1.setReg "p_gn" .blockPtr [p.BK]
    (endpointPointer p.toLayout l r.gate c p.keyHead p.keyTime p.keyFeature p.BK l.key)
  let s3 := s2.setReg "b_gn" .real [p.BK] (Tile.vec (fun i => some (x.keyEnd i)))
  let s4 := s3.setReg "b_h" .real [p.BK,p.BV] (stateTile p.toLayout (scaled x Bool.true f))
  let s5 := s4.setReg "b_g" .real [p.BK,p.BT] (Tile.mat (fun i j => some (x.keyGate i j)))
  s5.setReg "b_k" .real [p.BK,p.BT] (Tile.mat (fun i j => some (gatedKey x Bool.true i j)))

theorem key_executes (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u : BlockState)
    (hi : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hh : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (ht : u.regs .int [] "i_t" = some (Tile.scalar (c : Int)))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p.toLayout f))
    (hdata : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun i j => some ((RebasedAccess.loaded p.toLayout l a c).key i j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) (hstr : StrideRegs p u) (hBT : 0<p.BT) :
    stepStmts (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateKey p r) u = some (keyState p l r a c f u) := by
  let x := RebasedAccess.loaded p.toLayout l a c
  rw [gate_key_ast]
  simp only [RealHProgram.gateKey,List.drop,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_pointer_eval p l r.gate c u hh hi ht hstr))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_pointer_eval p l r.gate c
    p.keyHead p.keyTime p.keyFeature p.BK l.key
    "s_k_h" "s_k_t" "s_k_d" "i_k" _ hBT
    (by simp [hh]) (by simp [ht]) (by simp [hi])
    (by simpa using hstr.1)
    (by simpa using hstr.2.1)
    (by simpa using hstr.2.2.1)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_load p.toLayout l r.gate a.gate c
    p.keyHead p.keyTime p.keyFeature p.BK l.key "p_gn" _ (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (scale_key p.toLayout x f _
    (by simp [hcarry]) (by simp [x,RebasedAccess.loaded,RealHMath.keyIndex])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p.toLayout l r.gate a.gate c "p_g" _
    (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_key p.toLayout x _
    (by simpa using hdata) (by simp [x,RebasedAccess.loaded])
    (by simp [x,RebasedAccess.loaded,RealHMath.keyIndex]))),stepStmts.nil]
  rfl

def valueState (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u : BlockState) : BlockState :=
  let x := RebasedAccess.loaded p.toLayout l a c
  let s1 := u.setReg "p_g" .blockPtr [p.BT,p.BV] (valuePointer p.toLayout l r.gate c)
  let s2 := s1.setReg "p_gn" .blockPtr [p.BV]
    (endpointPointer p.toLayout l r.gate c p.valueHead p.valueTime p.valueFeature p.BV l.value)
  let s3 := s2.setReg "b_gn" .real [p.BV] (Tile.vec (fun i => some (x.valueEnd i)))
  let s4 := s3.setReg "b_h" .real [p.BK,p.BV] (stateTile p.toLayout (scaled x Bool.false f))
  let s5 := s4.setReg "b_g" .real [p.BT,p.BV] (Tile.mat (fun i j => some (x.valueGate i j)))
  s5.setReg "b_v" .real [p.BT,p.BV] (Tile.mat (fun i j => some (gatedValue x Bool.false i j)))

theorem value_executes (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u : BlockState)
    (hi : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hh : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (ht : u.regs .int [] "i_t" = some (Tile.scalar (c : Int)))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p.toLayout f))
    (hdata : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun i j => some ((RebasedAccess.loaded p.toLayout l a c).value i j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) (hstr : StrideRegs p u) (hBT : 0<p.BT) :
    stepStmts (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateValue p r) u = some (valueState p l r a c f u) := by
  let x := RebasedAccess.loaded p.toLayout l a c
  rw [gate_value_ast]
  simp only [RealHProgram.gateValue,List.drop,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_pointer_eval p l r.gate c u hh hi ht hstr))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_pointer_eval p l r.gate c
    p.valueHead p.valueTime p.valueFeature p.BV l.value
    "s_v_h" "s_v_t" "s_v_d" "i_v" _ hBT
    (by simp [hh]) (by simp [ht]) (by simp [hi])
    (by simpa using hstr.2.2.2.1)
    (by simpa using hstr.2.2.2.2.1)
    (by simpa using hstr.2.2.2.2.2.1)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (endpoint_load p.toLayout l r.gate a.gate c
    p.valueHead p.valueTime p.valueFeature p.BV l.value "p_gn" _ (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (scale_value p.toLayout x f _
    (by simp [hcarry]) (by simp [x,RebasedAccess.loaded,RealHMath.valueIndex])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p.toLayout l r.gate a.gate c "p_g" _
    (by simpa using hm) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (gate_value p.toLayout x _
    (by simpa using hdata) (by simp [x,RebasedAccess.loaded])
    (by simp [x,RebasedAccess.loaded,RealHMath.valueIndex]))),stepStmts.nil]
  rfl

def output (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (f : StateValue p.toLayout) (u : BlockState) : BlockState :=
  if gateK then keyState p l r a c f u else valueState p l r a c f u

theorem result (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (f : StateValue p.toLayout) (u : BlockState)
    (hk : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun e t => some ((RebasedAccess.loaded p.toLayout l a c).key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun t j => some ((RebasedAccess.loaded p.toLayout l a c).value t j)))) :
    RealHGate.Result p.toLayout (RebasedAccess.loaded p.toLayout l a c) gateK f u (output p l r a c gateK f u) := by
  cases gateK <;> constructor
  all_goals first | rfl | simp [output,keyState,valueState,gatedKey,gatedValue,hk,hv]

theorem executes (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (f : StateValue p.toLayout) (u : BlockState)
    (hik : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hh : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (ht : u.regs .int [] "i_t" = some (Tile.scalar (c : Int)))
    (hcarry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p.toLayout f))
    (hk : u.regs .real [p.BK,p.BT] "b_k" =
      some (Tile.mat (fun e t => some ((RebasedAccess.loaded p.toLayout l a c).key e t))))
    (hv : u.regs .real [p.BT,p.BV] "b_v" =
      some (Tile.mat (fun t j => some ((RebasedAccess.loaded p.toLayout l a c).value t j))))
    (hm : ∀ off, u.readMem r.gate off = a.gate off) (hstr : StrideRegs p u) (hBT : 0<p.BT) :
    stepStmt (.ifThenElse (.constBool gateK) (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateKey p r) (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateValue p r)) u =
      some (output p l r a c gateK f u) := by
  cases gateK
  · simpa only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,
      Tile.scalar,Bool.false_eq_true,↓reduceIte,output] using
      value_executes p l r a c f u hiv hh ht hcarry hv hm hstr hBT
  · simpa only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,
      Tile.scalar,↓reduceIte,output] using
      key_executes p l r a c f u hik hh ht hcarry hk hm hstr hBT

theorem integers (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK : Bool) (f : StateValue p.toLayout) (u : BlockState) (name : RegName)
    (hn : name∉["b_k","b_v","b_g","b_h","b_gn","p_g","p_gn"]) :
    (output p l r a c gateK f u).regs .int [] name=u.regs .int [] name := by
  simp only [List.mem_cons,List.mem_singleton,not_or] at hn
  rcases hn with ⟨h1,h2,h3,h4,h5,h6,h7⟩
  cases gateK <;> simp only [output,Bool.false_eq_true,↓reduceIte,keyState,valueState,
    BlockState.setReg,h1,h2,h3,h4,h5,h6,h7.1,↓reduceDIte]

#axiomsClean integers
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.Gate
