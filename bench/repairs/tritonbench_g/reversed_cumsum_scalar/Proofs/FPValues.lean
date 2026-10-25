import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPProgram

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.FPValues
open VeriTile.Triton FP.Structural FPProgram
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- A total opaque scan backend, retaining precision, operation, axis and direction. -/
abbrev ScanBackend (α : Type) := Option ComputeDType → {shape : TileShape} → ScanOp →
  Fin shape.length → ScanDirection → Values α .real shape → Values α .real shape

def machine (M : Algebra α) (scan : ScanBackend α) : Algebra α :=
  ({M with scan := fun p {shape} scanOp scanAxis scanDirection values => some (scan p scanOp scanAxis scanDirection values)}).withDefaultPrecision .fp32

def loaded (M : Algebra α) (h : Call) (c : Config) (input : Nat → α) (q : Nat) : Values α .real [c.BT] :=
  fun i => M.cast (some .fp32) h.inputDType .real
    (if q*c.BT+i.1.val<h.T then input (q*c.BT+i.1.val)
     else M.cast (some .fp32) .real h.inputDType (M.literal (some .fp32) .real 0))
def total (M : Algebra α) (h : Call) (c : Config) (input : Nat → α) (q : Nat) : α :=
  M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false (loaded M h c input q) PUnit.unit
def updated (M : Algebra α) (h : Call) (c : Config) (input : Nat → α) (q : Nat) (acc : α) : α :=
  M.binary (some .fp32) .real .add acc (total M h c input q)
def output (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (input : Nat → α)
    (q : Nat) (acc : α) : Values α .real [c.BT] :=
  fun i => M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .sub (loaded M h c input q i)
      (scan (some .fp32) .sum ⟨0,by simp⟩ .forward (loaded M h c input q) i)) acc

def lanes (c : Config) : Values α .nat [c.BT] := fun i => i.1.val
def mask (h : Call) (c : Config) (q : Nat) : Values α .bool [c.BT] := fun i => decide (Geometry.active h c q i.1)
def times (h : Call) (c : Config) (q : Nat) : Values α .int [c.BT] := fun i => (Geometry.time h c q i.1:Int)
def pointer (h : Call) (c : Config) (row q : Nat) (r : RegionName) (strides : Strides) : Values α .ptr [c.BT] :=
  fun i => (r,strides.address h.H row (Geometry.time h c q i.1))

theorem index_eval (M : Algebra α) (j : Int) (u : State α) (hj : j<1)
    (hr : u.regs .int [] "__rev_i_t"=some (fun _ => j)) :
    FP.Structural.evalOp M none (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")) u=
      some (fun _ => ((-j).toNat:Int)) := by
  simp only [FP.Structural.evalOp,hr]
  apply congrArg some
  funext i
  change 0-j=((-j).toNat:Int)
  rw [Int.toNat_of_nonneg (by omega)]
  omega

theorem start_eval (M : Algebra α) (c : Config) (q : Nat) (u : State α)
    (hi : u.regs .int [] "i_t"=some (fun _ => (q:Int))) :
    FP.Structural.evalOp M none (startOp c) u=some (fun _ => (q*c.BT:Int)) := by
  simp [startOp,FP.Structural.evalOp,hi,FP.Structural.bop,numeric,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem active_eval (M : Algebra α) (h : Call) (c : Config) (q : Nat) (u : State α)
    (hq : q*c.BT<h.T)
    (hs : u.regs .int [] "start_t"=some (fun _ => (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lanes"=some (lanes c)) :
    FP.Structural.evalOp M none (activeOp h c) u=some (mask h c q) := by
  simp only [activeOp,FP.Structural.evalOp,hs,hl,numericLt]
  apply congrArg some
  funext i
  simp [lanes,mask,Geometry.active,bop,numeric,Broadcast.leftIndex,Broadcast.rightIndex]
  rw [←Int.natCast_mul,←Int.natCast_sub (Nat.le_of_lt hq)]
  change ((i.1.val:Int)<((h.T-q*c.BT:Nat):Int)) ↔ i.1.val<h.T-q*c.BT
  omega

theorem time_eval (M : Algebra α) (h : Call) (c : Config) (q : Nat) (u : State α)
    (hs : u.regs .int [] "start_t"=some (fun _ => (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lanes"=some (lanes c))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q)) :
    FP.Structural.evalOp M none (timeOp c) u=some (times h c q) := by
  simp only [timeOp,FP.Structural.evalOp,hs,hl,hm]
  apply congrArg some
  funext i
  simp [lanes,mask,times,Geometry.time,bop,numeric,Broadcast.leftIndex,Broadcast.rightIndex]

theorem pointer_eval (M : Algebra α) (h : Call) (c : Config) (row q : Nat)
    (r : RegionName) (strides : Strides) (u : State α)
    (hr : u.regs .int [] "i_bh"=some (fun _ => (row:Int)))
    (ht : u.regs .int [c.BT] "times"=some (times h c q)) :
    FP.Structural.evalOp M none (pointerOp h c r strides) u=some (pointer h c row q r strides) := by
  simp only [pointerOp,baseOp,Op.tritonDiv,Op.tritonMod,FP.Structural.evalOp,hr,ht]
  apply congrArg some
  funext i
  simp [pointer,times,Strides.address,Strides.base,bop,numeric,integralFloorDiv,integralMod,
    Broadcast.leftIndex,Broadcast.rightIndex]
  have nonnegative : ¬(row:Int)<0 := by omega
  simp only [NumericDType.sub,nonnegative,↓reduceIte]
  have remainder : (row:Int)-(h.H:Int)*((row/h.H:Nat):Int)=((row%h.H:Nat):Int) := by
    simp only [Int.natCast_ediv,Int.natCast_emod,Int.emod_def]
  simp only [←Int.natCast_ediv]
  rw [remainder]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

theorem load_eval (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (row q : Nat)
    (input : Nat → α) (u : State α) (start : q*c.BT<h.T)
    (hp : u.regs .ptr [c.BT] "p_s"=some (pointer h c row q h.input h.inputStrides))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q))
    (read : ∀ t, t<h.T → toFloat h.inputDType
      ((u.mem h.input (h.inputStrides.address h.H row t)).read h.inputDType.toTileDType)=input t) :
    FP.Structural.evalOp (machine M scan) (some .fp32) (loadOp h c) u=some (loaded M h c input q) := by
  simp only [loadOp,FP.Structural.evalOp,hp,hm]
  apply congrArg some
  funext i
  by_cases ha : Geometry.active h c q i.1
  · have ht := (Geometry.active_iff h c q i.1 start).mp ha
    simp [pointer,mask,ha,Geometry.active_time h c q i.1 ha,read _ ht,loaded,ht,
      machine,Algebra.withDefaultPrecision,resolvePrecision]
    rfl
  · have ht := (Geometry.active_iff h c q i.1 start).not.mp ha
    simp [pointer,mask,ha,loaded,ht,machine,Algebra.withDefaultPrecision,resolvePrecision]
    cases h.inputDType <;> rfl

theorem sum_eval (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config)
    (input : Nat → α) (q : Nat) (u : State α)
    (hb : u.regs .real [c.BT] "b_s"=some (loaded M h c input q)) :
    FP.Structural.evalOp (machine M scan) (some .fp32) (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT] "b_s")) u=
      some (fun _ => total M h c input q) := by
  simp only [FP.Structural.evalOp,hb]
  apply congrArg some
  funext i
  cases i
  rfl

theorem carry_eval (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config)
    (input : Nat → α) (q : Nat) (acc : α) (u : State α)
    (ha : u.regs .real [] "b_z"=some (fun _ => acc))
    (hb : u.regs .real [] "b_zz"=some (fun _ => total M h c input q)) :
    FP.Structural.evalOp (machine M scan) (some .fp32) (.add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz")) u=
      some (fun _ => updated M h c input q acc) := by
  simp only [FP.Structural.evalOp,ha,hb]
  rfl

theorem output_eval (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config)
    (input : Nat → α) (q : Nat) (acc : α) (u : State α)
    (ha : u.regs .real [] "b_z"=some (fun _ => acc))
    (hb : u.regs .real [c.BT] "b_s"=some (loaded M h c input q)) :
    FP.Structural.evalOp (machine M scan) (some .fp32) (outputOp c) u=some (output M scan h c input q acc) := by
  simp only [outputOp,FP.Structural.evalOp,ha,hb]
  rfl

#print axioms pointer_eval
#print axioms load_eval
#print axioms output_eval
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.FPValues
