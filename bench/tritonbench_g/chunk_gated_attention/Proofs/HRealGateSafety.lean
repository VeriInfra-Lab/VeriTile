import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealGate
import bench.tritonbench_g.chunk_gated_attention.Proofs.HBounds
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.GateSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram
open RealHAccessSafety (load1_safe load2_safe)
open RebasedAccess VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Bounds
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem key_safe (p : Parameters) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) (u : BlockState) (hb : GateBounds p.toLayout l r c Bool.true bounds)
    (hi : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hh : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (ht : u.regs .int [] "i_t" = some (Tile.scalar (c : Int))) (hstr : StrideRegs p u) (hBT : 0<p.BT) :
    Stmt.TraceSafeList bounds (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateKey p r) u := by
  rw [gate_key_ast]
  simp only [RealHProgram.gateKey,List.drop,List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_pointer_eval p l r.gate c u hh hi ht hstr))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,integer,reg,natReg,add,mul,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (endpoint_pointer_eval p l r.gate c
    p.keyHead p.keyTime p.keyFeature p.BK l.key
    "s_k_h" "s_k_t" "s_k_d" "i_k" _ hBT
    (by simp [hh]) (by simp [ht]) (by simp [hi])
    (by simpa using hstr.1)
    (by simpa using hstr.2.1)
    (by simpa using hstr.2.2.1)))]
  refine ⟨by simp [Stmt.TraceSafe,endpointPtr,integer,reg,natReg,add,mul,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load1_safe r.gate (l.head*p.keyHead+(c*p.BT+p.BT-1)*p.keyTime+l.key*p.BK*p.keyFeature) p.BK p.BK
    p.keyFeature 0 "p_gn" bounds _ (hb.keyEnd rfl) (by simp [endpointPointer,vector]))
  intro endpoint
  apply TraceProofs.assign_cons (by simp [carry,Op.SafeAt,Op.SafeAt.eq_def])
  intro scaled
  apply TraceProofs.assign_cons (load2_safe r.gate (l.head*p.keyHead+l.key*p.BK*p.keyFeature+c*p.BT*p.keyTime) p.BK p.BT p.BK p.BT p.keyFeature p.keyTime 0 0
    "p_g" bounds _ (hb.key rfl) (by simp [keyPointer,matrix]))
  intro gate
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem value_safe (p : Parameters) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) (u : BlockState) (hb : GateBounds p.toLayout l r c Bool.false bounds)
    (hi : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hh : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (ht : u.regs .int [] "i_t" = some (Tile.scalar (c : Int))) (hstr : StrideRegs p u) (hBT : 0<p.BT) :
    Stmt.TraceSafeList bounds (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateValue p r) u := by
  rw [gate_value_ast]
  simp only [RealHProgram.gateValue,List.drop,List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_pointer_eval p l r.gate c u hh hi ht hstr))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,integer,reg,natReg,add,mul,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (endpoint_pointer_eval p l r.gate c
    p.valueHead p.valueTime p.valueFeature p.BV l.value
    "s_v_h" "s_v_t" "s_v_d" "i_v" _ hBT
    (by simp [hh]) (by simp [ht]) (by simp [hi])
    (by simpa using hstr.2.2.2.1)
    (by simpa using hstr.2.2.2.2.1)
    (by simpa using hstr.2.2.2.2.2.1)))]
  refine ⟨by simp [Stmt.TraceSafe,endpointPtr,integer,reg,natReg,add,mul,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load1_safe r.gate (l.head*p.valueHead+(c*p.BT+p.BT-1)*p.valueTime+l.value*p.BV*p.valueFeature) p.BV p.BV
    p.valueFeature 0 "p_gn" bounds _ (hb.valueEnd rfl) (by simp [endpointPointer,vector]))
  intro endpoint
  apply TraceProofs.assign_cons (by simp [carry,Op.SafeAt,Op.SafeAt.eq_def])
  intro scaled
  apply TraceProofs.assign_cons (load2_safe r.gate (l.head*p.valueHead+c*p.BT*p.valueTime+l.value*p.BV*p.valueFeature) p.BT p.BV p.BT p.BV p.valueTime p.valueFeature 0 0
    "p_g" bounds _ (hb.value rfl) (by simp [valuePointer,matrix]))
  intro gate
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem branch_safe (p : Parameters) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) (u : BlockState) (hb : GateBounds p.toLayout l r c gateK bounds)
    (hik : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (hit : u.regs .int [] "i_t" = some (Tile.scalar (c : Int))) (hstr : StrideRegs p u) (hBT : 0<p.BT) :
    Stmt.TraceSafe bounds (.ifThenElse (.constBool gateK) (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateKey p r) (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateValue p r)) u := by
  cases gateK
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,Bool.false_eq_true,↓reduceIte,true_and]
    exact value_safe p l r c bounds u hb hiv hih hit hstr hBT
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    exact key_safe p l r c bounds u hb hik hih hit hstr hBT

theorem update_safe (p : Parameters) (l : Launch) (r : Regions) (c : Nat) (gateK swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) (hb : GateBounds p.toLayout l r c gateK bounds)
    (hik : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (hit : u.regs .int [] "i_t" = some (Tile.scalar (c : Int))) (hstr : StrideRegs p u) (hBT : 0<p.BT) :
    Stmt.TraceSafe bounds (.ifThenElse (.constBool gateK)
      (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateKey p r ++ [updateStmt p.toLayout swapped])
      (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.gateValue p r ++ [updateStmt p.toLayout swapped])) u := by
  have hnext (v : BlockState) : Stmt.TraceSafeList bounds [updateStmt p.toLayout swapped] v := by
    apply TraceProofs.all_safe
    cases swapped <;> simp [updateStmt,Stmt.TraceSafe,carry,dot,Op.SafeAt,Op.SafeAt.eq_def]
  cases gateK
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,Bool.false_eq_true,↓reduceIte,true_and]
    exact TraceProofs.append_safe _ u (value_safe p l r c bounds u hb hiv hih hit hstr hBT) (fun v _ => hnext v)
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    exact TraceProofs.append_safe _ u (key_safe p l r c bounds u hb hik hih hit hstr hBT) (fun v _ => hnext v)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.GateSafety
