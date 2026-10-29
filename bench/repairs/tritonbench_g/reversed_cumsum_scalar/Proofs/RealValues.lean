import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealProgram

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealValues
open VeriTile.Triton RealProgram
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2400000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev inputTile := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.inputTile
abbrev chunkSum := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.chunkSum
abbrev chunkPrefix := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.chunkPrefix
abbrev outputTile := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.outputTile
abbrev sum_eval := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.sum_eval
abbrev carry_eval := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.carry_eval
abbrev output_eval := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.output_eval

def value (h : Call) (input : Nat → ℝ) (t : Nat) := if t<h.T then input t else 0
def lanes (c : Config) : Tile .nat [c.BT] := ⟨fun i => i.1.val⟩
def mask (h : Call) (c : Config) (q : Nat) : Tile .bool [c.BT] :=
  ⟨fun i => decide (Geometry.active h c q i.1)⟩
def times (h : Call) (c : Config) (q : Nat) : Tile .int [c.BT] :=
  ⟨fun i => (Geometry.time h c q i.1 : Int)⟩
def pointer (h : Call) (c : Config) (row q : Nat) (r : RegionName) (strides : Strides) : Tile .ptr [c.BT] :=
  ⟨fun i => (r,strides.address h.H row (Geometry.time h c q i.1))⟩

theorem start_eval (c : Config) (q : Nat) (u : BlockState)
    (hi : u.regs .int [] "i_t"=some (Tile.scalar (q:Int))) :
    evalOp (startOp c) u=some (Tile.scalar (q*c.BT:Int)) := by
  simp [-Tile.scalar_eta,startOp,evalOp,hi,Tile.bop,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul]

theorem active_eval (h : Call) (c : Config) (q : Nat) (u : BlockState)
    (hq : q*c.BT<h.T)
    (hs : u.regs .int [] "start_t"=some (Tile.scalar (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lanes"=some (lanes c)) :
    evalOp (activeOp h c) u=some (mask h c q) := by
  simp only [activeOp,evalOp,hs,hl]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,lanes,mask,Geometry.active,Tile.bop,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.sub,ComparableDType.lt,Int.natCast_sub (Nat.le_of_lt hq)]
  rw [←Int.natCast_mul,←Int.natCast_sub (Nat.le_of_lt hq)]
  change ((i.val:Int)<((h.T-q*c.BT:Nat):Int)) ↔ i.val<h.T-q*c.BT
  omega

theorem time_eval (h : Call) (c : Config) (q : Nat) (u : BlockState)
    (hs : u.regs .int [] "start_t"=some (Tile.scalar (q*c.BT:Int)))
    (hl : u.regs .nat [c.BT] "lanes"=some (lanes c))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q)) :
    evalOp (timeOp c) u=some (times h c q) := by
  simp only [timeOp,evalOp,hs,hl,hm]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,lanes,mask,times,Geometry.time,Tile.bop,Tile.scalar,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

theorem pointer_eval (h : Call) (c : Config) (row q : Nat) (r : RegionName)
    (strides : Strides) (u : BlockState)
    (hr : u.regs .int [] "i_bh"=some (Tile.scalar (row:Int)))
    (ht : u.regs .int [c.BT] "times"=some (times h c q)) :
    evalOp (pointerOp h c r strides) u=some (pointer h c row q r strides) := by
  simp only [pointerOp,baseOp,evalOp,hr,ht]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,pointer,times,Strides.address,Strides.base,Tile.bop,Tile.scalar,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,
    IntegralDType.floorDiv,IntegralDType.mod]
  have nonnegative : ¬(row:Int)<0 := by omega
  simp only [NumericDType.sub,nonnegative,↓reduceIte]
  have remainder : (row:Int)-(h.H:Int)*((row/h.H:Nat):Int)=((row%h.H:Nat):Int) := by
    simp only [Int.natCast_ediv,Int.natCast_emod,Int.emod_def]
  simp only [←Int.natCast_ediv]
  rw [remainder]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

theorem load_eval (h : Call) (c : Config) (row q : Nat) (input : Nat → ℝ) (u : BlockState)
    (hq : q*c.BT<h.T)
    (hp : u.regs .ptr [c.BT] "p_s"=some (pointer h c row q h.input h.inputStrides))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q))
    (read : ∀ t, t<h.T → u.readMem h.input (h.inputStrides.address h.H row t)=input t) :
    evalOp (loadOp c) u=some (inputTile c.BT q (value h input)) := by
  simp only [loadOp,evalOp,hp,hm]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  by_cases ha : Geometry.active h c q i
  · have ht := (Geometry.active_iff h c q i hq).mp ha
    simp [pointer,mask,ha,Geometry.active_time h c q i ha,BlockState.readMemValue_real,read _ ht,
      inputTile,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.inputTile,
      value,ht,Tile.vec]
  · have ht : ¬q*c.BT+i.val<h.T := (Geometry.active_iff h c q i hq).not.mp ha
    simp [pointer,mask,ha,inputTile,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.inputTile,
      value,ht,Tile.vec]

#print axioms pointer_eval
#print axioms load_eval
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealValues
