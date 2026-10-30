import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPProgram

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPValues
open VeriTile.Triton FP.Structural IndexProgram
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Total opaque dot implementation. Precision and TF32 options stay explicit. -/
abbrev DotBackend (α : Type) := Option ComputeDType → DotOptions →
  {batch : TileShape} → {m k n : Nat} → Values α .real (batch++[m,k]) →
    Values α .real (batch++[k,n]) → Values α .real (batch++[m,n])
def machine (M : Algebra α) (dot : DotBackend α) : Algebra α :=
  ({M with dotWithOptions := fun p options {batch} {m k n} a b => some (dot p options a b)}).withDefaultPrecision .fp32

def lanes (width : Nat) : Values α .nat [width] := fun i => i.1.val
def axisMask (length width start : Nat) : Values α .bool [width] := fun i => decide (i.1.val<length-start)
def coordinate (length width start : Nat) : Values α .nat [width] :=
  fun i => start+if i.1.val<length-start then i.1.val else 0
def mask (h : Call) (c : Config) (feature q : Nat) : Values α .bool [c.BT,BS] :=
  fun i => decide (Geometry.active h c feature q i.1.val i.2.1.val)
def pointer (h : Call) (c : Config) (feature row q : Nat) (r : RegionName) (strides : Strides) : Values α .ptr [c.BT,BS] :=
  fun i => (r,strides.address h.H row (Geometry.time h c q i.1.val) (Geometry.column h feature i.2.1.val))
def triangular (M : Algebra α) (c : Config) : Values α .real [c.BT,c.BT] :=
  fun i => if i.2.1.val ≤ i.1.val then M.literal (some .fp32) .real 1.0 else M.literal (some .fp32) .real 0.0

def loaded (M : Algebra α) (h : Call) (c : Config) (feature row : Nat) (initial : State α)
    (q : Nat) : Values α .real [c.BT,BS] :=
  fun i => M.cast (some .fp32) h.inputDType .real
    (if Geometry.active h c feature q i.1.val i.2.1.val then
      toFloat h.inputDType ((initial.mem h.input
        (h.inputStrides.address h.H row (Geometry.time h c q i.1.val)
          (Geometry.column h feature i.2.1.val))).read h.inputDType.toTileDType)
    else M.cast (some .fp32) .real h.inputDType (M.literal (some .fp32) .real 0))

def updated (M : Algebra α) (c : Config) (input : Values α .real [c.BT,BS])
    (acc : Values α .real [BS]) : Values α .real [BS] :=
  fun j => M.binary (some .fp32) .real .add (acc j)
    (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false input j)
def output (M : Algebra α) (dot : DotBackend α) (c : Config)
    (input : Values α .real [c.BT,BS]) (acc : Values α .real [BS]) : Values α .real [c.BT,BS] :=
  fun i => M.binary (some .fp32) .real .add (acc (i.2.1,PUnit.unit))
    (dot (some .fp32) {allowTF32:=some Bool.false} (batch:=[]) (triangular M c) input i)

theorem start_eval (M : Algebra α) (name : RegName) (width value : Nat) (u : State α)
    (hi : u.regs .nat [] name=some (fun _ => value)) :
    FP.Structural.evalOp M none (startOp name width) u=some (fun _ => value*width) := by
  simp only [startOp,FP.Structural.evalOp,hi]
  rfl

theorem active_eval (M : Algebra α) (length width start : Nat) (ln sn : RegName) (u : State α)
    (hs : u.regs .nat [] sn=some (fun _ => start))
    (hl : u.regs .nat [width] ln=some (lanes width)) :
    FP.Structural.evalOp M none (activeOp length width ln sn) u=some (axisMask length width start) := by
  simp only [activeOp,FP.Structural.evalOp,hs,hl]
  rfl

theorem coordinate_eval (M : Algebra α) (length width start : Nat) (sn mn ln : RegName) (u : State α)
    (hs : u.regs .nat [] sn=some (fun _ => start))
    (hm : u.regs .bool [width] mn=some (axisMask length width start))
    (hl : u.regs .nat [width] ln=some (lanes width)) :
    FP.Structural.evalOp M none (coordinateOp width sn mn ln) u=some (coordinate length width start) := by
  simp only [coordinateOp,FP.Structural.evalOp,hs,hm,hl]
  apply congrArg some
  funext i
  simp [coordinate,axisMask,lanes,bop,numeric,Broadcast.leftIndex,Broadcast.rightIndex]

theorem mask_eval (M : Algebra α) (h : Call) (c : Config) (feature q : Nat) (u : State α)
    (ht : u.regs .bool [c.BT] "active_t"=some (axisMask h.T c.BT (q*c.BT)))
    (hs : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (feature*BS))) :
    FP.Structural.evalOp M none (maskOp c) u=some (mask h c feature q) := by
  simp only [maskOp,FP.Structural.evalOp,ht,hs]
  apply congrArg some
  funext i
  simp [mask,axisMask,Geometry.active,Geometry.activeTime,Geometry.activeColumn,
    bop,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem pointer_eval (M : Algebra α) (h : Call) (c : Config) (feature row q : Nat)
    (r : RegionName) (strides : Strides) (u : State α)
    (hb : u.regs .nat [] "i_b"=some (fun _ => row/h.H))
    (hh : u.regs .nat [] "i_h"=some (fun _ => row%h.H))
    (ht : u.regs .nat [c.BT] "times"=some (coordinate h.T c.BT (q*c.BT)))
    (hs : u.regs .nat [BS] "columns"=some (coordinate h.S BS (feature*BS))) :
    FP.Structural.evalOp M none (pointerOp c r strides) u=some (pointer h c feature row q r strides) := by
  simp only [pointerOp,FP.Structural.evalOp,hb,hh,ht,hs]
  apply congrArg some
  funext i
  simp [pointer,coordinate,Geometry.time,Geometry.column,Geometry.activeTime,Geometry.activeColumn,
    Strides.address,Strides.base,bop,numeric,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex]

theorem load_eval (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (feature row q : Nat) (initial u : State α)
    (hp : u.regs .ptr [c.BT,BS] "p_s"=some (pointer h c feature row q h.input h.inputStrides))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c feature q))
    (read : ∀ off, u.mem h.input off=initial.mem h.input off) (padding : u.undef=initial.undef) :
    FP.Structural.evalOp (machine M dot) (some .fp32) (FPProgram.loadOp h c) u=
      some (loaded M h c feature row initial q) := by
  simp only [FPProgram.loadOp,FP.Structural.evalOp,hp,hm]
  apply congrArg some
  funext i
  by_cases ha : Geometry.active h c feature q i.1.val i.2.1.val
  all_goals simp [pointer,mask,ha,loaded,read,padding,machine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals cases h.inputDType <;> rfl

theorem output_eval (M : Algebra α) (dot : DotBackend α) (c : Config)
    (input : Values α .real [c.BT,BS]) (acc : Values α .real [BS]) (u : State α)
    (ha : u.regs .real [BS] "b_z"=some acc)
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (triangular M c))
    (hb : u.regs .real [c.BT,BS] "b_s"=some input) :
    FP.Structural.evalOp (machine M dot) (some .fp32) (outputOp c) u=some (output M dot c input acc) := by
  simp only [outputOp,FP.Structural.evalOp,List.nil_append,ha,hm,hb]
  rfl

theorem carry_eval (M : Algebra α) (dot : DotBackend α) (c : Config)
    (input : Values α .real [c.BT,BS]) (acc : Values α .real [BS]) (u : State α)
    (ha : u.regs .real [BS] "b_z"=some acc)
    (hb : u.regs .real [c.BT,BS] "b_s"=some input) :
    FP.Structural.evalOp (machine M dot) (some .fp32) (carryOp c) u=some (updated M c input acc) := by
  simp only [carryOp,FP.Structural.evalOp,ha,hb]
  rfl

#axiomsClean pointer_eval
#axiomsClean load_eval
#axiomsClean output_eval
#axiomsClean carry_eval
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPValues
