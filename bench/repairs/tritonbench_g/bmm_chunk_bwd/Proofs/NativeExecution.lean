import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeMask
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeAddress
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPPhysical

namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault NativeWord
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- Task-specific native schedule: actual signed-i64 masks and byte pointers,
physical FP32 loads, source casts, ordered dot/carry, residual, and typed store.
The dot callback can be the observed staged TF32 carried-dot graph. -/
def left (M : Algebra α) (h : HostCall) (c : Tuning) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (q : HostFP.Pid) (s : State α) (step : Nat) : Values α .real [c.BM,c.BCS] := fun i =>
  M.cast (some .fp32) .real .real
    (if NativeMask.dout h c q step i then
      (s.mem (a.region h.Dout) (NativeAddress.decode a alloc h.Dout (NativeAddress.dout h c a alloc q i step))).read .fp32
    else M.literal (some .fp32) .real 0)
def right (M : Algebra α) (h : HostCall) (c : Tuning) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (q : HostFP.Pid) (s : State α) (step : Nat) : Values α .real [c.BCS,c.BN] := fun i =>
  M.cast (some .fp32) .real .real
    (if NativeMask.a h c q step i then
      (s.mem (a.region h.A) (NativeAddress.decode a alloc h.A (NativeAddress.tensorA h c a alloc q i step))).read .fp32
    else M.literal (some .fp32) .real 0)

def carry (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (q : HostFP.Pid) (s : State α) : Nat → Values α .real [c.BM,c.BN]
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | n+1 => fun i => M.binary (some .fp32) .real .add (carry M D h c a alloc q s n i)
      (D.plain (some .fp32) (batch := []) (m := c.BM) (k := c.BCS) (n := c.BN)
        (left M h c a alloc q s n) (right M h c a alloc q s n) i)

def value (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (q : HostFP.Pid) (s : State α) (i : TileIndex [c.BM,c.BN]) : α :=
  let p := HostContract.parameters h
  let acc := carry M D h c a alloc q s (NativeScalar.setup h c q).stop.toNat i
  let acc := if h.hasResidual then
    let residual := if NativeMask.output h c q i then
      (s.mem (a.region h.Residual) (NativeAddress.decode a alloc h.Residual
        (NativeAddress.tensor h c a alloc q h.Residual p.rBatch p.rRow p.rHead p.rColumn i))).read .fp32
      else M.literal (some .fp32) .fp32 0
    M.binary (some .fp32) .real .add acc (M.cast (some .fp32) .fp32 .real residual)
    else acc
  M.cast (some .fp32) .real .fp32 acc

def tile (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (q : HostFP.Pid) (s : State α) : State α :=
  let p := HostContract.parameters h
  (TileShape.allIndices [c.BM,c.BN]).foldl (fun t i => if NativeMask.output h c q i then
    t.write (a.region h.Out) (NativeAddress.decode a alloc h.Out
      (NativeAddress.tensor h c a alloc q h.Out p.oBatch p.oRow p.oHead p.oColumn i))
      (.mk .fp32 (value M D h c a alloc q s i)) else t) s

def grid (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) : List HostFP.Pid → State α → State α
  | [],s => s
  | q::rest,s => grid M D h c a alloc rest (tile M D h c a alloc q s)

theorem left_eq (M : Algebra α) (h : HostCall) (choice : Fin 9)
    (metadata : NativeControl.Metadata h (tuning choice)) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc) (layout : HostContract.Layout h (tuning choice) abstract)
    (extent : a.extent = abstract.extent) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice))
    (s : State α) (step : Nat) (hs : step < (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS) :
    left M h (tuning choice) a alloc q s step =
      FPReference.leftBlock M (tuning choice).BM (tuning choice).BN (tuning choice).BCS (HostContract.parameters h)
        (HostFP.launch h (tuning choice) q (FPFlat.lift h a s)) step := by
  funext i
  unfold left
  rw [NativeMask.dout_exact h choice metadata q hq step hs i]
  simp only [decide_eq_true_eq]
  unfold FPReference.leftBlock
  change M.cast _ _ _ (if FPLogical.row h (tuning choice) q i.1 < h.chunk ∧
      i.2.1.val < FPLogical.length h q-step*(tuning choice).BCS then _ else _) =
    M.cast _ _ _ (if FPLogical.row h (tuning choice) q i.1 < h.chunk ∧
      i.2.1.val < FPLogical.length h q-step*(tuning choice).BCS then _ else _)
  split_ifs with hi
  · rw [NativeAddress.dout_word h choice metadata a alloc q hq i step,
      NativeAddress.decode_exact a alloc h.Dout _ (by rw [extent]; exact NativeAddress.d_bound h _ abstract layout q hq step i hi.1 hi.2)]
    exact congrArg (M.cast (some .fp32) .real .real)
      (FPFlat.lift_input h a s h.Dout (NativeAddress.dOffset h (tuning choice) q step i) (Or.inr rfl)).symm
  · rfl

theorem right_eq (M : Algebra α) (h : HostCall) (choice : Fin 9)
    (metadata : NativeControl.Metadata h (tuning choice)) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc) (layout : HostContract.Layout h (tuning choice) abstract)
    (extent : a.extent = abstract.extent) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice))
    (s : State α) (step : Nat) (hs : step < (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS) :
    right M h (tuning choice) a alloc q s step =
      FPReference.rightBlock M (tuning choice).BN (tuning choice).BCS (HostContract.parameters h)
        (HostFP.launch h (tuning choice) q (FPFlat.lift h a s)) step := by
  funext i
  unfold right
  rw [NativeMask.a_exact h choice metadata q hq step hs i]
  simp only [decide_eq_true_eq]
  unfold FPReference.rightBlock
  change M.cast _ _ _ (if i.1.val < FPLogical.length h q-step*(tuning choice).BCS ∧
      FPLogical.column h (tuning choice) q i.2.1 < h.K then _ else _) =
    M.cast _ _ _ (if i.1.val < FPLogical.length h q-step*(tuning choice).BCS ∧
      FPLogical.column h (tuning choice) q i.2.1 < h.K then _ else _)
  split_ifs with hi
  · rw [NativeAddress.a_word h choice metadata a alloc q hq i step,
      NativeAddress.decode_exact a alloc h.A _ (by rw [extent]; exact NativeAddress.a_bound h _ abstract layout q hq step i hi.1 hi.2)]
    exact congrArg (M.cast (some .fp32) .real .real)
      (FPFlat.lift_input h a s h.A (NativeAddress.aOffset h (tuning choice) q step i) (Or.inl rfl)).symm
  · rfl

theorem carry_eq (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (choice : Fin 9)
    (metadata : NativeControl.Metadata h (tuning choice)) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc) (layout : HostContract.Layout h (tuning choice) abstract)
    (extent : a.extent = abstract.extent) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice))
    (s : State α) (n : Nat) (hn : n ≤ (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS) :
    carry M D h (tuning choice) a alloc q s n =
      FPReference.carry M D (tuning choice).BM (tuning choice).BN (tuning choice).BCS (HostContract.parameters h)
        (HostFP.launch h (tuning choice) q (FPFlat.lift h a s)) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have hs : n < (FPLogical.length h q+(tuning choice).BCS-1)/(tuning choice).BCS := by omega
    simp only [carry,FPReference.carry,ih (by omega),left_eq M h choice metadata a alloc abstract layout extent q hq s n hs,
      right_eq M h choice metadata a alloc abstract layout extent q hq s n hs]

theorem value_eq (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (choice : Fin 9)
    (metadata : NativeControl.Metadata h (tuning choice)) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc) (layout : HostContract.Layout h (tuning choice) abstract)
    (extent : a.extent = abstract.extent) (tag : FPRoles.ResidualTag h)
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) (s : State α)
    (i : TileIndex [(tuning choice).BM,(tuning choice).BN])
    (hi : FPLogical.row h (tuning choice) q i.1 < FPLogical.length h q ∧ FPLogical.column h (tuning choice) q i.2.1 < h.K) :
    value M D h (tuning choice) a alloc q s i =
      FPReference.expected M D (tuning choice).BM (tuning choice).BN (tuning choice).BCS (HostContract.parameters h)
        (HostFP.launch h (tuning choice) q (FPFlat.lift h a s)) i := by
  unfold value
  rw [NativeMask.stop_exact h choice metadata q hq,carry_eq M D h choice metadata a alloc abstract layout extent q hq s _ (by rfl)]
  rw [NativeMask.output_exact h choice metadata q hq i]
  simp only [decide_eq_true_eq,hi,↓reduceIte]
  unfold FPReference.expected
  by_cases hr : h.hasResidual = Bool.true
  · simp only [HostContract.parameters,hr,↓reduceIte]
    rw [NativeAddress.tensor_word h choice metadata a alloc q hq,
      NativeAddress.decode_exact a alloc h.Residual _ (by rw [extent]; exact NativeAddress.residual_bound h _ abstract layout q hq i hi hr)]
    simp only [HostFP.launch]
    rw [FPFlat.lift_fp32 h a s h.Residual _ tag]
    rfl
  · simp only [HostContract.parameters,hr,↓reduceIte]
    rfl

/-- Complete native tile execution agrees with physical typed execution.
All loaded lanes, loop steps, residual casts, and active stores are covered. -/
theorem tile_eq (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (choice : Fin 9)
    (metadata : NativeControl.Metadata h (tuning choice)) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc) (layout : HostContract.Layout h (tuning choice) abstract)
    (extent : a.extent = abstract.extent) (tag : FPRoles.ResidualTag h)
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice)) (s : State α) :
    tile M D h (tuning choice) a alloc q s = FPPhysical.tile M D h (tuning choice) a q s := by
  unfold tile FPPhysical.tile
  dsimp only
  congr 1
  funext t i
  rw [NativeMask.output_exact h choice metadata q hq i]
  simp only [decide_eq_true_eq]
  change (if FPLogical.row h (tuning choice) q i.1 < FPLogical.length h q ∧
      FPLogical.column h (tuning choice) q i.2.1 < h.K then _ else _) =
    (if FPLogical.row h (tuning choice) q i.1 < FPLogical.length h q ∧
      FPLogical.column h (tuning choice) q i.2.1 < h.K then _ else _)
  split_ifs with ha
  · rw [value_eq M D h choice metadata a alloc abstract layout extent tag q hq s i ha,
      NativeAddress.tensor_word h choice metadata a alloc q hq,
      NativeAddress.decode_exact a alloc h.Out _ (by rw [extent]; exact NativeAddress.output_bound h _ abstract layout q hq i ha)]
    rfl
  · rfl

/-- Full actual launch list, including empty-grid calls. -/
theorem grid_eq (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (choice : Fin 9)
    (metadata : NativeControl.Metadata h (tuning choice)) (a : FPFlat.Placement)
    (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc) (layout : HostContract.Layout h (tuning choice) abstract)
    (extent : a.extent = abstract.extent) (tag : FPRoles.ResidualTag h)
    (jobs : List HostFP.Pid) (within : ∀ q ∈ jobs, q ∈ HostFP.grid h (tuning choice)) (s : State α) :
    grid M D h (tuning choice) a alloc jobs s = FPPhysical.grid M D h (tuning choice) a jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    simp only [grid,FPPhysical.grid,tile_eq M D h choice metadata a alloc abstract layout extent tag q (within q (by simp))]
    exact ih (fun u hu => within u (by simp [hu])) _

theorem source_grid (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (choice : Fin 9)
    (precision : FPGrid.DefaultPrecision h) (metadata : NativeControl.Metadata h (tuning choice))
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc)
    (layout : HostContract.Layout h (tuning choice) abstract) (extent : a.extent = abstract.extent)
    (tag : FPRoles.ResidualTag h) (fresh : FPPhysical.FreshStorage h a)
    (s : State α) (typed : FPPhysical.ScratchTyped h a s) :
    ∃ source, HostFP.run (backend M D) h (tuning choice) Bool.false (HostFP.grid h (tuning choice)) (FPFlat.lift h a s) = some source ∧
      (grid M D h (tuning choice) a alloc (HostFP.grid h (tuning choice)) s).mem =
        (FPFlat.publish a h.Out s source).mem := by
  rw [grid_eq M D h choice metadata a alloc abstract layout extent tag _ (fun q hq => hq)]
  exact FPPhysical.source_grid M D h (tuning choice) precision a abstract layout extent fresh s typed

#axiomsClean left_eq
#axiomsClean right_eq
#axiomsClean carry_eq
#axiomsClean value_eq
#axiomsClean tile_eq
#axiomsClean grid_eq
#axiomsClean source_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeExecution
