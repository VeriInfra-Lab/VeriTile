import bench.tritonbench_g.chunk_gla_simple.Kernels
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

/-- The three actual FP32 dot sites need their native backend to be available.
The result functions are opaque: no value, reduction order or numerical law is
specified. These local capabilities cannot assert whole-kernel success. -/
structure Dots {α : Type} (M : Algebra α) (c : Shape) where
  qh : Values α .real [c.BT,c.BK] → Values α .real [c.BK,c.BV] → Values α .real [c.BT,c.BV]
  qk : Values α .real [c.BT,c.BK] → Values α .real [c.BK,c.BT] → Values α .real [c.BT,c.BT]
  sv : Values α .real [c.BT,c.BT] → Values α .real [c.BT,c.BV] → Values α .real [c.BT,c.BV]
  qh_eq : ∀ a b, M.dotWithOptions (some .fp32) {allowTF32 := some Bool.false} (batch := []) a b = some (qh a b)
  qk_eq : ∀ a b, M.dotWithOptions (some .fp32) {allowTF32 := some Bool.false} (batch := []) a b = some (qk a b)
  sv_eq : ∀ a b, M.dotWithOptions (some .fp32) {allowTF32 := some Bool.false} (batch := []) a b = some (sv a b)

def parts (c : Shape) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel c scale).surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def Present {α : Type} (s : State α) (d : TileDType) (sh : TileShape) (n : RegName) : Prop :=
  ∃ v, s.regs d sh n = some v

def Holds {α : Type} (c : Shape) (s : State α) : Prop :=
  Present s .nat [] "i_bh" ∧ Present s .nat [] "i_t" ∧
  Present s .nat [c.BT] "rows" ∧ Present s .nat [c.BV] "cols" ∧
  Present s .bool [c.BT,c.BT] "m_s" ∧ Present s .real [c.BT,c.BV] "b_o" ∧
  Present s .real [c.BT,c.BT] "b_s"

theorem entry {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α) :
    ∃ t, run (engine M) (parts c scale).before s = some t ∧ Holds c t := by
  simp only [parts,kernel,fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map,Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  dsimp [fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Holds,Present]

theorem iteration {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Dots M c) (n : Nat) (s : State α) (hs : Holds c s) :
    ∃ t, run (engine M) (parts c scale).body (s.setReg "i_k" .nat [] (fun _ => n)) = some t ∧ Holds c t := by
  rcases hs with ⟨⟨bh,hbh⟩,⟨it,hit⟩,⟨rows,hr⟩,⟨cols,hc⟩,⟨mask,hm⟩,⟨bo,ho⟩,⟨bs,hbs⟩⟩
  simp only [parts,kernel,fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map,Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  dsimp [fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    hbh,hit,hr,hc,hm,ho,hbs,d.qh_eq,d.qk_eq,Holds,Present]

theorem range_executes {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Dots M c) (cur stop : Nat) (s : State α) (hs : Holds c s) :
    ∃ t, range (engine M) "i_k" cur stop 1 (parts c scale).body s = some t ∧ Holds c t := by
  by_cases h : cur < stop
  · obtain ⟨u,he,hu⟩ := iteration M c scale d cur s hs
    obtain ⟨t,ht,hi⟩ := range_executes M c scale d (cur+1) stop u hu
    refine ⟨t,?_,hi⟩
    rw [range]
    simp only [show (1 : Nat) ≠ 0 from by decide,h,↓reduceIte,he,bind,Option.bind_some,ht]
  · refine ⟨s,?_,hs⟩
    rw [range]
    simp [h]
termination_by stop-cur

theorem finish {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Dots M c) (s : State α) (hs : Holds c s) :
    ∃ t, run (engine M) (parts c scale).after s = some t := by
  rcases hs with ⟨⟨bh,hbh⟩,⟨it,hit⟩,⟨rows,hr⟩,⟨cols,hc⟩,⟨mask,hm⟩,⟨bo,ho⟩,⟨bs,hbs⟩⟩
  simp only [parts,kernel,fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map,Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  dsimp [fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp (config := {maxSteps := 150000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    hbh,hit,hr,hc,hm,ho,hbs,d.sv_eq,FP.Structural.store]

theorem source_parts (c : Shape) (scale : ℝ) :
    (kernel c scale).surfaceBody = (parts c scale).before ++
    [.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop (parts c scale).stride (parts c scale).body] ++
    (parts c scale).after := rfl

/-- Every default-FP32 typed program executes successfully under the three
opaque, available native dot operations. All memory, casts, address/control
operations, separate add/multiply and gate arithmetic are executed here. -/
theorem executes {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Dots M c) (s : State α) :
    ∃ t, FP.Structural.exec (engine M) (kernel c scale) s = some t := by
  obtain ⟨u,hu,hi⟩ := entry M c scale s
  obtain ⟨v,hv,hiv⟩ := range_executes M c scale d 0 ((c.K+c.BK-1)/c.BK) u hi
  obtain ⟨t,ht⟩ := finish M c scale d v hiv
  have hl : step (engine M) (.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop
      (parts c scale).stride (parts c scale).body) u = some v := by
    have ha : (parts c scale).start = .constNat 0 := rfl
    have hb : (parts c scale).stop = .div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.constNat c.K) (.constNat c.BK)) (.constNat 1)) (.constNat c.BK) := rfl
    have hd : (parts c scale).stride = .constNat 1 := rfl
    simp only [step,ha,hb,hd,evalOp_unfold,numeric,bop,Option.bind,Option.bind_some]
    exact hv
  refine ⟨t,?_⟩
  change run (engine M) (kernel c scale).surfaceBody s = _
  rw [source_parts]
  simp only [List.append_assoc,run_append,hu,Option.bind_some,List.singleton_append]
  rw [run]
  simp only [hl,bind,Option.bind_some,ht]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Typed
