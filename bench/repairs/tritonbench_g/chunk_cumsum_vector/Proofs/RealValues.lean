import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealProgram
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealStep

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealValues
open VeriTile.Triton IndexProgram
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev lowerTriTile := VeriTile.Bench.TritonBenchG.ChunkCumsumVector.lowerTriTile
abbrev dotLowerTri_sum_if_some := VeriTile.Bench.TritonBenchG.ChunkCumsumVector.dotLowerTri_sum_if_some
abbrev coe_sum := @VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealStep.coe_sum

def naturalLanes (width : Nat) : Tile .nat [width] := Tile.vec fun i => i.val
def lanes (width : Nat) : Tile .int [width] := Tile.vec fun i => (i.val:Int)
def axisMask (length width start : Nat) : Tile .bool [width] :=
  Tile.vec fun i => decide (i.val<length-start)
def coordinate (length width start : Nat) : Tile .int [width] :=
  Tile.vec fun i => ((start+if i.val<length-start then i.val else 0 : Nat):Int)
def mask (h : Call) (c : Config) (feature q : Nat) : Tile .bool [c.BT,BS] :=
  ⟨fun idx => decide (Geometry.active h c feature q idx.1.val idx.2.1.val)⟩
def pointer (h : Call) (c : Config) (feature row q : Nat) (r : RegionName)
    (strides : Strides) : Tile .ptr [c.BT,BS] :=
  ⟨fun idx => (r,strides.address h.H row (Geometry.time h c q idx.1.val)
    (Geometry.column h feature idx.2.1.val))⟩
def inputTile (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat) : Tile .real [c.BT,BS] :=
  Tile.mat fun i j => some (f (q*c.BT+i.val) j)
def chunkPrefix (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat)
    (i : Fin c.BT) (j : Fin BS) : ℝ :=
  ∑ k ∈ Finset.univ.filter (fun k : Fin c.BT => k.val ≤ i.val),f (q*c.BT+k.val) j
def nextCarry (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat) (acc : Fin BS → ℝ)
    (j : Fin BS) : ℝ := acc j+∑ k : Fin c.BT,f (q*c.BT+k.val) j
def outputTile (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat) (acc : Fin BS → ℝ) : Tile .real [c.BT,BS] :=
  Tile.mat fun i j => some (acc j+chunkPrefix c f q i j)

theorem rowDiv_eval (row H : Nat) (u : BlockState)
    (hi : u.regs .int [] "i_bh"=some (Tile.scalar (row:Int))) :
    evalOp (.tritonDiv .int .nil (.ref .int [] "i_bh")
      (.castNatToInt (.constNat H))) u=some (Tile.scalar (row/H:Int)) := by
  have nonnegative : ¬ (row:Int)<0 := by omega
  simp [-Tile.scalar_eta,nonnegative,Tile.select,Op.tritonDiv,evalOp,hi,Tile.bop,Tile.cop,Tile.scalar,
    NumericDType.add,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,Int.emod_def]

theorem rowMod_eval (row H : Nat) (u : BlockState)
    (hi : u.regs .int [] "i_bh"=some (Tile.scalar (row:Int))) :
    evalOp (.tritonMod .int .nil (.ref .int [] "i_bh")
      (.castNatToInt (.constNat H))) u=some (Tile.scalar (row%H:Int)) := by
  have nonnegative : ¬ (row:Int)<0 := by omega
  simp [-Tile.scalar_eta,nonnegative,Tile.select,Op.tritonMod,Op.tritonDiv,evalOp,hi,Tile.bop,Tile.cop,Tile.scalar,
    NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,ComparableDType.lt,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,Int.emod_def]

theorem start_eval (name : RegName) (width value : Nat) (u : BlockState)
    (hi : u.regs .int [] name=some (Tile.scalar (value:Int))) :
    evalOp (startOp name width) u=some (Tile.scalar (value*width:Int)) := by
  simp only [startOp,evalOp,hi]
  rfl

theorem timeStart_eval (c : Config) (q : Nat) (u : BlockState)
    (hi : u.regs .int [] "i_t"=some (Tile.scalar (q:Int))) :
    evalOp (timeStartOp c) u=some (Tile.scalar (q*c.BT:Int)) := by
  simp only [timeStartOp,evalOp,hi]
  rfl

theorem active_eval (length width start : Nat) (ln sn : RegName) (u : BlockState)
    (hs : u.regs .int [] sn=some (Tile.scalar (start:Int)))
    (hl : u.regs .int [width] ln=some (lanes width)) :
    evalOp (activeOp length width ln sn) u=some (axisMask length width start) := by
  simp only [activeOp,evalOp,hs,hl]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,axisMask,lanes,Tile.bop,Tile.cop,Tile.vec,Tile.scalar,
    NumericDType.sub,ComparableDType.lt,Broadcast.leftIndex,Broadcast.rightIndex]
  change ((i.val:Int)<(length:Int)-(start:Int)) ↔ i.val<length-start
  by_cases hs : start ≤ length
  · rw [←Int.natCast_sub hs,Int.ofNat_lt]
  · have hz : length-start=0 := Nat.sub_eq_zero_of_le (by omega)
    rw [hz]
    omega

theorem coordinate_eval (length width start : Nat) (sn mn ln : RegName) (u : BlockState)
    (hs : u.regs .int [] sn=some (Tile.scalar (start:Int)))
    (hm : u.regs .bool [width] mn=some (axisMask length width start))
    (hl : u.regs .int [width] ln=some (lanes width)) :
    evalOp (coordinateOp width sn mn ln) u=some (coordinate length width start) := by
  simp only [coordinateOp,evalOp,hs,hm,hl]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [-Tile.scalar_eta,coordinate,axisMask,lanes,Tile.bop,Tile.vec,Tile.scalar,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

theorem mask_eval (h : Call) (c : Config) (feature q : Nat) (u : BlockState)
    (ht : u.regs .bool [c.BT] "active_t"=some (axisMask h.T c.BT (q*c.BT)))
    (hs : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (feature*BS))) :
    evalOp (maskOp c) u=some (mask h c feature q) := by
  simp only [maskOp,evalOp]
  erw [evalOp_expandDim,evalOp_ref,ht,evalOp_expandDim,evalOp_ref,hs]
  simp only [Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [mask,axisMask,Geometry.active,Geometry.activeTime,Geometry.activeColumn,
    Tile.bop,Tile.vec,Tile.expandDim,TileShape.dropInsertedIndex,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem pointer_eval (h : Call) (c : Config) (feature row q : Nat) (r : RegionName)
    (strides : Strides) (u : BlockState)
    (hb : u.regs .int [] "i_b"=some (Tile.scalar (row/h.H:Int)))
    (hh : u.regs .int [] "i_h"=some (Tile.scalar (row%h.H:Int)))
    (ht : u.regs .int [c.BT] "times"=some (coordinate h.T c.BT (q*c.BT)))
    (hs : u.regs .int [BS] "columns"=some (coordinate h.S BS (feature*BS))) :
    evalOp (pointerOp c r strides) u=some (pointer h c feature row q r strides) := by
  simp only [pointerOp,evalOp,hb,hh]
  erw [evalOp_expandDim,evalOp_ref,ht,evalOp_expandDim,evalOp_ref,hs]
  simp only [Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [pointer,coordinate,Geometry.time,Geometry.column,Geometry.activeTime,Geometry.activeColumn,
    Strides.address,Strides.base,Tile.ptrAdd,Tile.bop,Tile.vec,Tile.scalar,Tile.expandDim,
    TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul]
  simp only [Int.ofNat_eq_natCast,←Int.natCast_ediv,←Int.natCast_emod,←Int.natCast_mul,
    ←Int.natCast_add,Int.toNat_natCast,Nat.zero_add,Region.cast]

theorem load_eval (h : Call) (c : Config) (feature row q : Nat) (f : Nat → Fin BS → ℝ)
    (u : BlockState)
    (hp : u.regs .ptr [c.BT,BS] "p_s"=some (pointer h c feature row q h.input h.inputStrides))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c feature q))
    (read : ∀ (i : Fin c.BT) (j : Fin BS),
      (if Geometry.active h c feature q i.val j.val then
        u.readMem h.input (h.inputStrides.address h.H row (Geometry.time h c q i.val) (Geometry.column h feature j.val))
      else 0)=f (q*c.BT+i.val) j) :
    evalOp (RealProgram.loadOp c) u=some (inputTile c f q) := by
  simp only [RealProgram.loadOp,evalOp,hp,hm]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  have hr := read i j
  by_cases ha : Geometry.active h c feature q i.val j.val
  all_goals simp [pointer,mask,ha,inputTile,Tile.mat,BlockState.readMemValue_real,
    BlockState.paddingValue,BlockState.unspecifiedValue] at hr ⊢
  all_goals exact congrArg some hr

theorem dot_value (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat)
    (i : Fin c.BT) (j : Fin BS) :
    (Tile.dot [] (lowerTriTile c.BT) (inputTile c f q)).data (i,j,PUnit.unit)=
      some (chunkPrefix c f q i j) := by
  have h := dotLowerTri_sum_if_some c.BT BS (fun k j => f (q*c.BT+k.val) j)
    (fun _ _ => True) i j
  simpa only [inputTile,Tile.mat,ite_true,and_true,chunkPrefix] using h

theorem output_eval (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat)
    (acc : Fin BS → ℝ) (u : BlockState)
    (ha : u.regs .real [BS] "b_z"=some (Tile.vec fun j => some (acc j)))
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (lowerTriTile c.BT))
    (hb : u.regs .real [c.BT,BS] "b_s"=some (inputTile c f q)) :
    evalOp (outputOp c) u=some (outputTile c f q acc) := by
  rw [outputOp,evalOp_add]
  erw [evalOp_expandDim,evalOp_ref,ha,evalOp_dot,evalOp_ref,hm,evalOp_ref,hb]
  simp only [Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,Tile.expandDim,
    TileShape.dropInsertedIndex,Tile.vec,NumericDType.add,dot_value,outputTile,
    Tile.mat,WithBot.realAdd,Option.map₂_some_some]
  erw [dot_value]
  rfl

theorem carry_eval (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat)
    (acc : Fin BS → ℝ) (u : BlockState)
    (ha : u.regs .real [BS] "b_z"=some (Tile.vec fun j => some (acc j)))
    (hb : u.regs .real [c.BT,BS] "b_s"=some (inputTile c f q)) :
    evalOp (carryOp c) u=some (Tile.vec fun j => some (nextCarry c f q acc j)) := by
  rw [carryOp,evalOp_add]
  erw [evalOp_ref,ha,evalOp_reduceSum,evalOp_ref,hb]
  simp only [Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    Tile.reduceSum_false,Tile.reduceSumDrop_data,NumericDType.add,inputTile,Tile.mat,
    TileShape.insertAxisIndex,nextCarry]
  erw [coe_sum]
  rfl

#axiomsClean pointer_eval
#axiomsClean load_eval
#axiomsClean output_eval
#axiomsClean carry_eval
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealValues
