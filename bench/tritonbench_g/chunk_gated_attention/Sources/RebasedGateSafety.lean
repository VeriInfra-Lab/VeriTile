import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedAccessSafety

/-! Safety of both complete gate branches. Only the selected branch requires
its two-dimensional gate tile and one-dimensional full-chunk endpoint bounds.
Numerical values remain abstract; safety does not require zero padding. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedGateSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
open RealHAccessSafety (load1_safe load2_safe)
open RebasedAccess RebasedAccessSafety
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) : Prop where
  key : gateK = Bool.true → KeyBounds p l r.gate c bounds
  keyEnd : gateK = Bool.true → EndpointBounds p l r.gate c p.keyHead p.keyTime p.keyFeature p.BK l.key bounds
  value : gateK = Bool.false → ValueBounds p l r.gate c bounds
  valueEnd : gateK = Bool.false → EndpointBounds p l r.gate c p.valueHead p.valueTime p.valueFeature p.BV l.value bounds

theorem key_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) (u : BlockState) (hb : Bounds p l r c Bool.true bounds)
    (hi : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (RebasedProgram.gateKey p r) u := by
  simp only [RebasedProgram.gateKey,RealHProgram.gateKey,List.drop,List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.gate c u hh hi ht))]
  refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.keyPtr,RebasedProgram.local2,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.keyHead p.keyTime p.keyFeature p.BK l.key "i_k" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.endpointPtr,RebasedProgram.local1,times,Op.SafeAt],?_⟩
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

theorem value_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) (u : BlockState) (hb : Bounds p l r c Bool.false bounds)
    (hi : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (RebasedProgram.gateValue p r) u := by
  simp only [RebasedProgram.gateValue,RealHProgram.gateValue,List.drop,List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.gate c u hh hi ht))]
  refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.valuePtr,RebasedProgram.local2,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (endpoint_eval p l r.gate c
    p.valueHead p.valueTime p.valueFeature p.BV l.value "i_v" _
    (by simp [hh]) (by simp [ht]) (by simp [hi])))]
  refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.endpointPtr,RebasedProgram.local1,times,Op.SafeAt],?_⟩
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

theorem branch_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) (u : BlockState) (hb : Bounds p l r c gateK bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafe bounds (.ifThenElse (.constBool gateK) (RebasedProgram.gateKey p r) (RebasedProgram.gateValue p r)) u := by
  cases gateK
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,Bool.false_eq_true,↓reduceIte,true_and]
    exact value_safe p l r c bounds u hb hiv hih hit
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    exact key_safe p l r c bounds u hb hik hih hit

theorem update_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) (hb : Bounds p l r c gateK bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafe bounds (.ifThenElse (.constBool gateK)
      (RebasedProgram.gateKey p r ++ [updateStmt p swapped])
      (RebasedProgram.gateValue p r ++ [updateStmt p swapped])) u := by
  have hnext (v : BlockState) : Stmt.TraceSafeList bounds [updateStmt p swapped] v := by
    apply TraceProofs.all_safe
    cases swapped <;> simp [updateStmt,Stmt.TraceSafe,carry,dot,Op.SafeAt,Op.SafeAt.eq_def]
  cases gateK
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,Bool.false_eq_true,↓reduceIte,true_and]
    exact TraceProofs.append_safe _ u (value_safe p l r c bounds u hb hiv hih hit) (fun v _ => hnext v)
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    exact TraceProofs.append_safe _ u (key_safe p l r c bounds u hb hik hih hit) (fun v _ => hnext v)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedGateSafety
