import bench.tritonbench_g.attn_fwd_triton.Proofs.Execution
import VeriTile.Triton.Float.StructuralSafety

/-! Actual Structural source accesses, including both causal loops. These
address obligations are later obtained from the public logical IO windows;
they are not numerical laws or execution assumptions. -/
namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph Model
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure InitialBounds (p : Config) (BD pm ph : Nat) (bounds : RegionName → Nat) : Prop where
  query : ∀ row : Fin 128, ∀ dim : Fin BD,
    pm*128+row.val<p.context → dim.val<p.headDim →
      ph/p.heads*p.queryBatch+ph%p.heads*p.queryHead+
        (pm*128+row.val)*p.queryRow+dim.val*p.queryDim < bounds p.Q
  scale : ph*((p.context+128-1)/128)+pm < bounds p.QS

structure IterationBounds (p : Config) (f : Fixed BD) (cur : Nat)
    (ptr : Cursor BD) (bounds : RegionName → Nat) : Prop where
  key : ∀ i : TileIndex [BD,64],
    f.keys (i.2.1,PUnit.unit)<f.context-cur → i.1.val<p.headDim →
      (ptr.key i).2 < bounds (ptr.key i).1
  scale : ptr.scale.2 < bounds ptr.scale.1
  value : ∀ i : TileIndex [64,BD],
    f.keys (i.1,PUnit.unit)<f.context-cur → i.2.1.val<p.headDim →
      (ptr.value i).2 < bounds (ptr.value i).1

def OutputBounds (p : Config) (f : Fixed BD) (bounds : RegionName → Nat) : Prop :=
  ∀ i : TileIndex [128,BD], f.rows (i.1,PUnit.unit)<f.context → i.2.1.val<p.headDim →
    (f.output i).2 < bounds (f.output i).1

theorem initial_safe (p : Config) (BD : Nat) (s : State Expr)
    (bounds : RegionName → Nat) (hb : InitialBounds p BD (s.pids 0) (s.pids 1) bounds) :
    Safety.traceSafeList (engine backend) bounds ((code p BD).take 27) s := by
  simp only [code,source,Bool.false_eq_true,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
  simp [Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,ofFloat,toFloat,Region.cast,State.setReg,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend]
  exact ⟨hb.query,hb.scale⟩

theorem setup_safe (p : Config) (BD : Nat) (stage : p.stage=3) (second : Bool)
    (f : Fixed BD) (old : Carry BD) (s : State Expr) (hb : Bound f old s)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine backend) bounds (setupCode p BD second) s := by
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  cases second
  all_goals simp only [setupCode,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero,
    List.take_succ_cons,List.take_zero]
  all_goals simp [Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    stage,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14]

theorem iteration_safe (p : Config) (BD : Nat) (stage : p.stage=3)
    (second : Bool) (f : Fixed BD) (old : Carry BD) (ptr : Cursor BD)
    (lo hi cur : Nat) (s : State Expr) (hb : Bound f old s)
    (hl : LoopBound second ptr lo hi s) (bounds : RegionName → Nat)
    (backed : IterationBounds p f cur ptr bounds) :
    Safety.traceSafeList (engine backend) bounds (body p BD second)
      (s.setReg "start_n" .nat [] (fun _=>cur)) := by
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases hl with ⟨hkp,hks,hvp,hlo,hhi⟩
  cases second
  all_goals simp only [keyName,scaleName,valueName,lowerName,upperName,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte] at hkp hks hvp hlo hhi
  all_goals simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true]
  all_goals simp [Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    stage,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,
    keyName,scaleName,valueName,lowerName,upperName,TileShape.axisDim,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,hkp,hks,hvp,hlo,hhi,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend]
  all_goals exact ⟨fun a b c => backed.key (a,b,c),backed.scale,
    fun a b c => backed.value (a,b,c)⟩

theorem tail_safe (p : Config) (BD : Nat) (f : Fixed BD) (old : Carry BD)
    (s : State Expr) (hb : Bound f old s) (bounds : RegionName → Nat)
    (backed : OutputBounds p f bounds) :
    Safety.traceSafeList (engine backend) bounds (tail p BD Bool.false) s := by
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp only [tail,source,Bool.false_eq_true,↓reduceIte,AttnFwdTriton.TypedSource.repaired,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero]
  simp [Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,natLt,numericLt,bop,
    Cell.read,ofFloat,toFloat,Region.cast,State.setReg,h0,h4,h9,h13,h14,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend]
  exact fun a b c => backed (a,b,c)

end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.SourceSafety
