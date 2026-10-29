import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.RealProjection

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def Ready (pid : Nat) (s : BlockState) : Prop :=
  s.regs .nat [] "i_bh"=some (Tile.scalar pid) ∧ (s.regs .real [] "b_z").isSome

def Rel (a b : BlockState) : Prop :=
  a.mem=b.mem ∧ a.undef=b.undef ∧ a.pids=b.pids ∧ a.numPids=b.numPids ∧
    a.regs .nat [] "i_bh"=b.regs .nat [] "i_bh" ∧ a.regs .real [] "b_z"=b.regs .real [] "b_z"

@[simp] theorem int_address (p T j BT lane : Nat) :
    ((p:Int)*T+((j:Int)*BT+lane)).toNat=p*T+(j*BT+lane) := by omega

@[simp] theorem int_lt (j BT lane T : Nat) :
    ((j:Int)*BT+lane<(T:Int)) ↔ j*BT+lane<T := by norm_cast

@[simp] theorem masked_undef {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : BlockState) :
    (l.foldl (fun t i => if active i then t.writeMem r (off i) (val i) else t) s).undef=s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split <;> rfl

@[simp] theorem masked_numPids {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : BlockState) :
    (l.foldl (fun t i => if active i then t.writeMem r (off i) (val i) else t) s).numPids=s.numPids := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split <;> rfl

@[simp] theorem masked_pids {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : BlockState) :
    (l.foldl (fun t i => if active i then t.writeMem r (off i) (val i) else t) s).pids=s.pids := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split <;> rfl

theorem masked_mem {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s t : BlockState) (hm : s.mem=t.mem) :
    (l.foldl (fun t i => if active i then t.writeMem r (off i) (val i) else t) s).mem=
    (l.foldl (fun t i => if active i then t.writeMem r (off i) (val i) else t) t).mem := by
  induction l generalizing s t with
  | nil => exact hm
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases h : active i
    · simp only [if_pos h,BlockState.writeMem,hm]
    · simpa only [if_neg h] using hm

@[simp] theorem expand_ref (s : BlockState) :
    evalOp (Op.expandDim (shape := []) (dtype := .real) ⟨0,by simp⟩ (.ref .real [] "b_z")) s=
      (s.regs .real [] "b_z").map (fun v => Tile.expandDim ⟨0,by simp⟩ v) := by
  simp only [evalOp.eq_def,Option.bind,Option.map]
  cases s.regs .real [] "b_z" <;> rfl

@[simp] theorem sum_ref (BT : Nat) (s : BlockState) :
    evalOp (Op.reduceSum (shape := [BT]) ⟨0,by simp⟩ Bool.false (.ref .real [BT] "b_s")) s=
      (s.regs .real [BT] "b_s").map (fun v => Tile.reduceSum ⟨0,by simp⟩ Bool.false v) := by
  simp only [evalOp.eq_def,Option.bind,Option.map]
  cases s.regs .real [BT] "b_s" <;> rfl

@[simp] theorem set_mem (s : BlockState) (name : RegName) (d : TileDType) (sh : TileShape) (v : Tile d sh) :
    (s.setReg name d sh v).mem=s.mem := rfl

@[simp] theorem set_undef (s : BlockState) (name : RegName) (d : TileDType) (sh : TileShape) (v : Tile d sh) :
    (s.setReg name d sh v).undef=s.undef := rfl

theorem body (input output : RegionName) (T BT pid index : Nat) (a b : BlockState)
    (related : Rel a b) (ready : Ready pid b) :
    ∃ x y,
      stepStmts (actualBody input output T BT)
        (a.setReg "i_t" .int [] (Tile.scalar (index:Int)))=some x ∧
      stepStmts (ccLoopBody input output T BT)
        (b.setReg "i_t" .nat [] (Tile.scalar index))=some y ∧
      Rel x y ∧ Ready pid y := by
  obtain ⟨hm,hu,hp,hn,hi,hc⟩ := related
  obtain ⟨hb,hz⟩ := ready
  obtain ⟨carry,hz⟩ := Option.isSome_iff_exists.mp hz
  have ha := hi.trans hb
  have hca := hc.trans hz
  simp (config := {maxSteps := 300000}) [-Tile.scalar_eta,-Tile.scan_data,-Tile.scan_reverse_data,
    -Tile.reduceSum_false,-Tile.reduceSum_true,-Tile.reduceSumDrop_data,-Tile.reduceSumKeep_data,actualBody,pointer,mask,prelude,ccLoopBody,
    stepStmts,stepStmt,evalOp,expand_ref,sum_ref,evalOp_expandDim,evalOp_reduceSum,Tile.cop,List.mapM,List.mapM.loop,Option.bind,Option.map,
    Tile.bop,Tile.ptrAdd,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,TileShape.axisDim,Region.cast,
    BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped,BlockState.paddingValue,BlockState.unspecifiedValue,
    Ready,Rel,ha,hca,hb,hz,hm,hu,hp,hn,TileShape.indexToList]
  simp only [evalOp.eq_def,Option.bind,Option.map]
  simp (config := {maxSteps := 300000}) [-Tile.scalar_eta,-Tile.scan_data,-Tile.scan_reverse_data,
    -Tile.reduceSum_false,-Tile.reduceSum_true,-Tile.reduceSumDrop_data,-Tile.reduceSumKeep_data,
    Option.bind,Option.map,Tile.bop,Tile.ptrAdd,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,TileShape.axisDim,Region.cast,
    BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped,BlockState.paddingValue,BlockState.unspecifiedValue,
    Ready,Rel,ha,hca,hb,hz,hm,hu,hp,hn,TileShape.indexToList]
  apply masked_mem
  simpa only [set_mem] using hm

#axiomsClean body
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
