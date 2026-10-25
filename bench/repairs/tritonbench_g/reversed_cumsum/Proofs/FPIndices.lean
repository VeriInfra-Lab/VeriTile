import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPProgram

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPIndices
open VeriTile.Triton FP.Structural FPProgram IndexProgram
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def lanes (n : Nat) : Values α .nat [n] := fun i => i.1.val
def maskT (h : Call) (c : Config) (q : Nat) : Values α .bool [c.BT] := fun i => decide (Geometry.activeT h c q i.1)
def maskS (h : Call) (f : Nat) : Values α .bool [BS] := fun j => decide (Geometry.activeS h f j.1)
def times (h : Call) (c : Config) (q : Nat) : Values α .int [c.BT] := fun i => (Geometry.time h c q i.1:Int)
def features (h : Call) (f : Nat) : Values α .int [BS] := fun j => (Geometry.feature h f j.1:Int)
def mask (h : Call) (c : Config) (q f : Nat) : Values α .bool [c.BT,BS] := fun i => decide (Geometry.active h c q f i.1 i.2.1)
def pointer (h : Call) (c : Config) (row q f : Nat) (r : RegionName) : Values α .ptr [c.BT,BS] :=
  fun i => (r,Geometry.address h row (Geometry.time h c q i.1) (Geometry.feature h f i.2.1))

theorem startT_eval (M : Algebra α) (c : Config) (q : Nat) (u : State α)
    (hi : u.regs .int [] "i_t"=some (fun _ => (q:Int))) :
    FP.Structural.evalOp M none (startTOp c) u=some (fun _ => (q*c.BT:Int)) := by
  simp [startTOp,FP.Structural.evalOp,hi,bop,Broadcast.leftIndex,Broadcast.rightIndex,numeric]
  rfl

theorem startS_eval (M : Algebra α) (f : Nat) (u : State α)
    (hi : u.regs .int [] "i_s"=some (fun _ => (f:Int))) :
    FP.Structural.evalOp M none startSOp u=some (fun _ => (f*BS:Int)) := by
  simp [startSOp,FP.Structural.evalOp,hi,bop,Broadcast.leftIndex,Broadcast.rightIndex,numeric]
  rfl

theorem activeT_eval (M : Algebra α) (h : Call) (c : Config) (q : Nat) (u : State α) (hq : q*c.BT<h.T)
    (hs : u.regs .int [] "start_t"=some (fun _ => (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lane_t"=some (lanes c.BT)) :
    FP.Structural.evalOp M none (activeTOp h c) u=some (maskT h c q) := by
  simp only [activeTOp,FP.Structural.evalOp,hs,hl,numericLt]
  apply congrArg some
  funext idx
  rcases idx with ⟨i,⟨⟩⟩
  simp [lanes,maskT,Geometry.activeT,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    numeric,numericLt,Int.natCast_sub (Nat.le_of_lt hq)]
  rw [←Int.natCast_mul,←Int.natCast_sub (Nat.le_of_lt hq)]
  change ((i.val:Int)<((h.T-q*c.BT:Nat):Int)) ↔ i.val<h.T-q*c.BT
  omega

theorem activeS_eval (M : Algebra α) (h : Call) (f : Nat) (u : State α)
    (hs : u.regs .int [] "start_s"=some (fun _ => (f*BS:Int)))
    (hl : u.regs .nat [BS] "lane_s"=some (lanes BS)) :
    FP.Structural.evalOp M none (activeSOp h) u=some (maskS h f) := by
  simp only [activeSOp,FP.Structural.evalOp,hs,hl,numericLt]
  apply congrArg some
  funext idx
  rcases idx with ⟨j,⟨⟩⟩
  simp [lanes,maskS,Geometry.activeS,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    numeric,numericLt]

  rw [←Int.natCast_mul]
  generalize f * BS = n
  change ((j.val : Int) < (h.S : Int) - (n : Int)) ↔ j.val < h.S - n
  by_cases hn : n ≤ h.S
  · rw [←Int.natCast_sub hn, Int.ofNat_lt]
  · have hz : h.S - n = 0 := Nat.sub_eq_zero_of_le (by omega)
    rw [hz]
    omega

theorem time_eval (M : Algebra α) (h : Call) (c : Config) (q : Nat) (u : State α)
    (hs : u.regs .int [] "start_t"=some (fun _ => (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lane_t"=some (lanes c.BT))
    (hm : u.regs .bool [c.BT] "active_t"=some (maskT h c q)) :
    FP.Structural.evalOp M none (timeOp c) u=some (times h c q) := by
  simp only [timeOp,FP.Structural.evalOp,hs,hl,hm]
  apply congrArg some
  funext idx
  rcases idx with ⟨i,⟨⟩⟩
  simp [lanes,maskT,times,Geometry.time,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,numeric]

theorem feature_eval (M : Algebra α) (h : Call) (f : Nat) (u : State α)
    (hs : u.regs .int [] "start_s"=some (fun _ => (f*BS:Int)))
    (hl : u.regs .nat [BS] "lane_s"=some (lanes BS))
    (hm : u.regs .bool [BS] "active_s"=some (maskS h f)) :
    FP.Structural.evalOp M none featureOp u=some (features h f) := by
  simp only [featureOp,FP.Structural.evalOp,hs,hl,hm]
  apply congrArg some
  funext idx
  rcases idx with ⟨j,⟨⟩⟩
  simp [lanes,maskS,features,Geometry.feature,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,numeric]

theorem mask_eval (M : Algebra α) (h : Call) (c : Config) (q f : Nat) (u : State α)
    (ht : u.regs .bool [c.BT] "active_t"=some (maskT h c q))
    (hs : u.regs .bool [BS] "active_s"=some (maskS h f)) :
    FP.Structural.evalOp M none (maskOp c) u=some (mask h c q f) := by
  simp only [maskOp,FP.Structural.evalOp,ht,hs]
  apply congrArg some
  funext idx
  rcases idx with ⟨i,j,⟨⟩⟩
  simp [mask,maskT,maskS,Geometry.active,bop,TileShape.dropInsertedIndex,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem pointer_eval (M : Algebra α) (h : Call) (c : Config) (row q f : Nat) (r : RegionName) (u : State α)
    (hr : u.regs .int [] "i_bh"=some (fun _ => (row:Int)))
    (ht : u.regs .int [c.BT] "times"=some (times h c q))
    (hs : u.regs .int [BS] "features"=some (features h f)) :
    FP.Structural.evalOp M none (pointerOp h c r) u=some (pointer h c row q f r) := by
  simp only [pointerOp,FP.Structural.evalOp,hr,ht,hs]
  apply congrArg some
  funext idx
  rcases idx with ⟨i,j,⟨⟩⟩
  simp [pointer,times,features,Geometry.address,bop,
    TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,
    numeric]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]


#print axioms pointer_eval
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPIndices
