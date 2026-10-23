import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPIndices

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPValues
open VeriTile.Triton FP.Structural FPProgram FPIndices
noncomputable section
set_option maxHeartbeats 4800000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- A total opaque dot operation, preserving precision and every option. -/
abbrev DotBackend (α : Type) := Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
  Values α .real (batch++[m,k]) → Values α .real (batch++[k,n]) → Values α .real (batch++[m,n])

def machine (M : Algebra α) (dot : DotBackend α) : Algebra α :=
  ({M with dotWithOptions := fun p opts {batch} {m k n} a b => some (dot p opts a b)}).withDefaultPrecision .fp32

def options : DotOptions := {allowTF32:=some Bool.false}

def triangular (M : Algebra α) (c : Config) : Values α .real [c.BT,c.BT] :=
  fun i => if i.1.val ≤ i.2.1.val then M.literal (some .fp32) .real 1.0 else M.literal (some .fp32) .real 0.0

def loaded (M : Algebra α) (h : Call) (c : Config) (f : Nat) (input : Nat → Nat → α) (q : Nat) : Values α .real [c.BT,BS] :=
  fun i => M.cast (some .fp32) h.inputDType .real
    (if q*c.BT+i.1.val<h.T ∧ f*BS+i.2.1.val<h.S then input (q*c.BT+i.1.val) (f*BS+i.2.1.val)
     else M.cast (some .fp32) .real h.inputDType (M.literal (some .fp32) .real 0))

def updated (M : Algebra α) (h : Call) (c : Config) (f : Nat) (input : Nat → Nat → α)
    (q : Nat) (acc : Values α .real [BS]) : Values α .real [BS] :=
  fun j => M.binary (some .fp32) .real .add (acc j)
    (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false (loaded M h c f input q) j)

def output (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (f : Nat)
    (input : Nat → Nat → α) (q : Nat) (acc : Values α .real [BS]) : Values α .real [c.BT,BS] :=
  fun i => M.binary (some .fp32) .real .add (acc i.2)
    (dot (some .fp32) options (batch:=[]) (m:=c.BT) (k:=c.BT) (n:=BS) (triangular M c) (loaded M h c f input q) i)

theorem triangular_eval (M : Algebra α) (dot : DotBackend α) (c : Config) (u : State α)
    (hi : u.regs .nat [c.BT] "o_i"=some (lanes c.BT)) :
    FP.Structural.evalOp (machine M dot) none (triOp c) u=some (triangular M c) := by
  simp only [triOp,FP.Structural.evalOp,hi]
  apply congrArg some
  funext i
  simp [triangular,lanes,bop,numericLe,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex,machine,Algebra.withDefaultPrecision,resolvePrecision]

theorem load_eval (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (row q : Nat)
    (f : Geometry.FeatureBlock h) (input : Nat → Nat → α) (u : State α) (start : q*c.BT<h.T)
    (hp : u.regs .ptr [c.BT,BS] "p_s"=some (pointer h c row q f.val h.input))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c q f.val))
    (read : ∀ t s, t<h.T → s<h.S → toFloat h.inputDType
      ((u.mem h.input (Geometry.address h row t s)).read h.inputDType.toTileDType)=input t s) :
    FP.Structural.evalOp (machine M dot) (some .fp32) (loadOp h c) u=some (loaded M h c f.val input q) := by
  simp only [loadOp,FP.Structural.evalOp,hp,hm]
  apply congrArg some
  funext i
  by_cases ha : Geometry.active h c q f.val i.1 i.2.1
  · have hv := (Geometry.active_iff h c q f i.1 i.2.1 start).mp ha
    simp [pointer,mask,ha,Geometry.time,Geometry.feature,ha.1,ha.2,read _ _ hv.1 hv.2,
      loaded,hv.1,hv.2,machine,Algebra.withDefaultPrecision,resolvePrecision]
    rfl
  · have hv := (Geometry.active_iff h c q f i.1 i.2.1 start).not.mp ha
    simp [pointer,mask,ha,loaded,hv,machine,Algebra.withDefaultPrecision,resolvePrecision]
    cases h.inputDType <;> rfl

theorem output_eval (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (f : Nat)
    (input : Nat → Nat → α) (q : Nat) (acc : Values α .real [BS]) (u : State α)
    (ha : u.regs .real [BS] "b_z"=some acc)
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (triangular M c))
    (hb : u.regs .real [c.BT,BS] "b_s"=some (loaded M h c f input q)) :
    FP.Structural.evalOp (machine M dot) (some .fp32) (outputOp c) u=some (output M dot h c f input q acc) := by
  simp only [outputOp,FP.Structural.evalOp,List.nil_append,ha,hm,hb]
  rfl

theorem carry_eval (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (f : Nat)
    (input : Nat → Nat → α) (q : Nat) (acc : Values α .real [BS]) (u : State α)
    (ha : u.regs .real [BS] "b_z"=some acc)
    (hb : u.regs .real [c.BT,BS] "b_s"=some (loaded M h c f input q)) :
    FP.Structural.evalOp (machine M dot) (some .fp32) (carryOp c) u=some (updated M h c f input q acc) := by
  simp only [carryOp,FP.Structural.evalOp,ha,hb]
  rfl

#print axioms output_eval
#print axioms load_eval
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPValues
