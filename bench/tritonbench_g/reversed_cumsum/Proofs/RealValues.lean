import bench.tritonbench_g.reversed_cumsum.Proofs.RealProgram

namespace VeriTile.Bench.Repairs.ReversedCumsum.RealValues
open VeriTile.Triton RealProgram IndexProgram
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def lanes (n : Nat) : Tile .nat [n] := ⟨fun i => i.1.val⟩
def maskT (h : Call) (c : Config) (q : Nat) : Tile .bool [c.BT] := ⟨fun i => decide (Geometry.activeT h c q i.1)⟩
def maskS (h : Call) (f : Nat) : Tile .bool [BS] := ⟨fun j => decide (Geometry.activeS h f j.1)⟩
def times (h : Call) (c : Config) (q : Nat) : Tile .int [c.BT] := ⟨fun i => (Geometry.time h c q i.1:Int)⟩
def features (h : Call) (f : Nat) : Tile .nat [BS] := ⟨fun j => Geometry.feature h f j.1⟩
def mask (h : Call) (c : Config) (q f : Nat) : Tile .bool [c.BT,BS] := ⟨fun i => decide (Geometry.active h c q f i.1 i.2.1)⟩
def pointer (h : Call) (c : Config) (row q f : Nat) (r : RegionName) : Tile .ptr [c.BT,BS] :=
  ⟨fun i => (r,Geometry.address h row (Geometry.time h c q i.1) (Geometry.feature h f i.2.1))⟩
def inputTile (h : Call) (c : Config) (row f : Nat) (a : Nat → ℝ) (q : Nat) : Tile .real [c.BT,BS] :=
  VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.inputTile (layout h c) (launch row f) a (fun _ => 0) q

theorem startT_eval (c : Config) (q : Nat) (u : BlockState)
    (hi : u.regs .int [] "i_t"=some (Tile.scalar (q:Int))) :
    evalOp (startTOp c) u=some (Tile.scalar (q*c.BT:Int)) := by
  simp [-Tile.scalar_eta,startTOp,evalOp,hi,Tile.bop,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul]

theorem startS_eval (f : Nat) (u : BlockState)
    (hi : u.regs .nat [] "i_s"=some (Tile.scalar f)) :
    evalOp startSOp u=some (Tile.scalar (f*BS)) := by
  simp [-Tile.scalar_eta,startSOp,evalOp,hi,Tile.bop,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul]

theorem activeT_eval (h : Call) (c : Config) (q : Nat) (u : BlockState) (hq : q*c.BT<h.T)
    (hs : u.regs .int [] "start_t"=some (Tile.scalar (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lane_t"=some (lanes c.BT)) :
    evalOp (activeTOp h c) u=some (maskT h c q) := by
  simp only [activeTOp,evalOp,hs,hl]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,lanes,maskT,Geometry.activeT,Tile.bop,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.sub,ComparableDType.lt,Int.natCast_sub (Nat.le_of_lt hq)]
  rw [←Int.natCast_mul,←Int.natCast_sub (Nat.le_of_lt hq)]
  change ((i.val:Int)<((h.T-q*c.BT:Nat):Int)) ↔ i.val<h.T-q*c.BT
  omega

theorem activeS_eval (h : Call) (f : Nat) (u : BlockState)
    (hs : u.regs .nat [] "start_s"=some (Tile.scalar (f*BS)))
    (hl : u.regs .nat [BS] "lane_s"=some (lanes BS)) :
    evalOp (activeSOp h) u=some (maskS h f) := by
  simp only [activeSOp,evalOp,hs,hl]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [-Tile.scalar_eta,lanes,maskS,Geometry.activeS,Tile.bop,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.sub,ComparableDType.lt]

theorem time_eval (h : Call) (c : Config) (q : Nat) (u : BlockState)
    (hs : u.regs .int [] "start_t"=some (Tile.scalar (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lane_t"=some (lanes c.BT))
    (hm : u.regs .bool [c.BT] "active_t"=some (maskT h c q)) :
    evalOp (timeOp c) u=some (times h c q) := by
  simp only [timeOp,evalOp,hs,hl,hm]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,lanes,maskT,times,Geometry.time,Tile.bop,Tile.scalar,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

theorem feature_eval (h : Call) (f : Nat) (u : BlockState)
    (hs : u.regs .nat [] "start_s"=some (Tile.scalar (f*BS)))
    (hl : u.regs .nat [BS] "lane_s"=some (lanes BS))
    (hm : u.regs .bool [BS] "active_s"=some (maskS h f)) :
    evalOp featureOp u=some (features h f) := by
  simp only [featureOp,evalOp,hs,hl,hm]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [-Tile.scalar_eta,lanes,maskS,features,Geometry.feature,Tile.bop,Tile.scalar,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

theorem mask_eval (h : Call) (c : Config) (q f : Nat) (u : BlockState)
    (ht : u.regs .bool [c.BT] "active_t"=some (maskT h c q))
    (hs : u.regs .bool [BS] "active_s"=some (maskS h f)) :
    evalOp (maskOp c) u=some (mask h c q f) := by
  simp only [maskOp,evalOp,ht,hs]
  erw [evalOp_expandDim, evalOp_ref, ht, evalOp_expandDim, evalOp_ref, hs]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [mask,maskT,maskS,Geometry.active,Tile.bop,Tile.expandDim,TileShape.dropInsertedIndex,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem pointer_eval (h : Call) (c : Config) (row q f : Nat) (r : RegionName) (u : BlockState)
    (hr : u.regs .nat [] "i_bh"=some (Tile.scalar row))
    (ht : u.regs .int [c.BT] "times"=some (times h c q))
    (hs : u.regs .nat [BS] "features"=some (features h f)) :
    evalOp (pointerOp h c r) u=some (pointer h c row q f r) := by
  simp only [pointerOp,evalOp,hr,ht,hs]
  erw [evalOp_expandDim, evalOp_ref, ht, evalOp_expandDim, evalOp_ref, hs]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [-Tile.scalar_eta,pointer,times,features,Geometry.address,Tile.bop,Tile.scalar,
    Tile.expandDim,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.add,NumericDType.mul]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

theorem load_eval (h : Call) (c : Config) (row q : Nat) (f : Geometry.FeatureBlock h)
    (a : Nat → ℝ) (u : BlockState) (hq : q*c.BT<h.T)
    (hp : u.regs .ptr [c.BT,BS] "p_s"=some (pointer h c row q f.val h.input))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c q f.val))
    (read : ∀ off, u.readMem h.input off=a off) :
    evalOp (loadOp c) u=some (inputTile h c row f.val a q) := by
  simp only [loadOp,evalOp,hp,hm]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  by_cases ha : Geometry.active h c q f.val i j
  · have hv := (Geometry.active_iff h c q f i j hq).mp ha
    simp [pointer,mask,ha,Geometry.time,Geometry.feature,ha.1,ha.2,BlockState.readMemValue_real,read,
      inputTile,VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.inputTile,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.value,VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.active,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address,layout,launch,Geometry.address,Tile.mat,hv.1,hv.2]
  · have hv := (Geometry.active_iff h c q f i j hq).not.mp ha
    simp [pointer,mask,ha,inputTile,VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.inputTile,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.value,VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.active,
      layout,launch,Tile.mat,hv]

#print axioms pointer_eval
#print axioms load_eval
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealValues
