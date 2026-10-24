import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 3000000

/-- An actual signed-int64 product has these bits, whether or not it overflows.
Pointer addition uses the same modulo-64 encoding of its signed interpretation. -/
def rowOffset64 (row stride : Nat) : BitVec 64 := BitVec.ofNat 64 row * BitVec.ofNat 64 stride

def rowPointer64 (base row stride : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base + BitVec.ofNat 64 4 * rowOffset64 row stride

def lanePointer64 (pointer : BitVec 64) (col : Nat) : BitVec 64 :=
  pointer + BitVec.ofNat 64 4 * BitVec.ofNat 64 col

/-- Actual loop-carried pointer: initial row_start*stride, then one pointer
increment by the observed stride for each executed iteration. -/
def loopPointer64 (base first stride : Nat) : Nat → BitVec 64
  | 0 => rowPointer64 base first stride
  | k+1 => loopPointer64 base first stride k + BitVec.ofNat 64 4 * BitVec.ofNat 64 stride

 theorem rowPointer64_bits (base row stride : Nat) :
    rowPointer64 base row stride = BitVec.ofNat 64 (base+4*(row*stride)) := by
  simp only [rowPointer64,rowOffset64,← BitVec.ofNat_mul,← BitVec.ofNat_add]

 theorem lanePointer64_bits (base row stride col : Nat) :
    lanePointer64 (rowPointer64 base row stride) col = BitVec.ofNat 64 (base+4*(row*stride+col)) := by
  simp only [lanePointer64,rowPointer64_bits,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  congr 1
  ring

 theorem loopPointer64_bits (base first stride k : Nat) :
    loopPointer64 base first stride k = rowPointer64 base (first+k) stride := by
  induction k with
  | zero => simp [loopPointer64]
  | succ k ih =>
    rw [loopPointer64,ih]
    simp only [rowPointer64_bits,← BitVec.ofNat_mul,← BitVec.ofNat_add]
    congr 1
    ring

 theorem signed_product_encoding (row stride : Nat) :
    BitVec.ofInt 64 (rowOffset64 row stride).toInt = rowOffset64 row stride := by simp

/-- Ordinary API laws admit arbitrary singleton row strides. In particular,
this proposition places no bound on the otherwise unused row stride. -/
 theorem singleton_rowStride (b : Buffer) (h : Valid b [1,columns]) : RowStride b 1 columns := by
  intro row col hr hc
  have row_zero : row = 0 := by omega
  subst row
  have bound : col < b.size := by simpa [valid_size h] using hc
  simpa using h.tensor.contiguous h.contiguous col bound

 theorem active_pointer64 (b : Buffer) (h : Valid b [rows,columns])
    (stride : RowStride b rows columns) (binding : Bound pointers b)
    (hc : 0 < columns) (row col : Nat) (hr : row < rows) (hj : col < columns) :
    (lanePointer64 (rowPointer64 b.pointer row b.stride0) col).toNat =
      address pointers (b.region,row*columns+col) := by
  rw [lanePointer64_bits,active_row_stride h stride hc row hr]
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (valid_access h _ (input_bound h _ _ hr hj)).1]
  simp [address,binding.pointer]

 theorem active_loop_pointer64 (b : Buffer) (h : Valid b [rows,columns])
    (stride : RowStride b rows columns) (binding : Bound pointers b)
    (hc : 0 < columns) (first row col : Nat) (lo : first ≤ row)
    (hr : row < rows) (hj : col < columns) :
    (lanePointer64 (loopPointer64 b.pointer first b.stride0 (row-first)) col).toNat =
      address pointers (b.region,row*columns+col) := by
  rw [loopPointer64_bits,Nat.add_sub_of_le lo]
  exact active_pointer64 b h stride binding hc row col hr hj

/-- Only these row pointers are dereferenced by range(row_start,row_end,1).
The pointer increment after its last element is absent from this observation. -/
def visitedRows (first last : Nat) : List Nat := (List.range (last-first)).map (first+·)

 theorem visitedRows_bounds (first last row : Nat) (h : row ∈ visitedRows first last) :
    first ≤ row ∧ row < last := by
  obtain ⟨k,hk,rfl⟩ := List.mem_map.mp h
  have hk := List.mem_range.mp hk
  omega

 theorem inactive_rows (first last : Nat) (inactive : last ≤ first) : visitedRows first last = [] := by
  simp [visitedRows,Nat.sub_eq_zero_of_le inactive]

/-- Operational empty-group statement: the actual source loop returns its
entry state without executing even one body statement or memory load. -/
 theorem inactive_range {α : Type} [Inhabited α] (M : Algebra α)
    (c : BackwardCall) (s : State α) (inactive : c.last ≤ c.first) :
    FP.Structural.range M "_row" c.first c.last 1 (Backward.body c.parameters) s = some s := by
  rw [FP.Structural.range]
  simp [Nat.not_lt_of_ge inactive]

/-- An idle source program's entire memory result is its two zero-partial
stores. Prelude pointer values (including wrapped ones) are not loaded. -/
 theorem inactive_reference {α : Type} [Inhabited α] (M : Algebra α)
    (R : VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.SumPrimitive α)
    (c : BackwardCall) (s : State α) (pid : s.pids 0 = c.pid) (inactive : c.last ≤ c.first) :
    Backward.reference M R c.parameters s =
      Backward.finish M c.parameters c.pid (Backward.initial M c.parameters s) := by
  unfold Backward.reference Backward.accumulated
  have stop : ¬ Backward.start c.parameters s < Backward.stop c.parameters s := by
    simpa [Backward.start,Backward.stop,BackwardCall.parameters,BackwardCall.first,BackwardCall.last,pid] using
      Nat.not_lt_of_ge inactive
  simp [foldRange,stop,pid]

/-- Concrete reviewer case: the inactive prelude product wraps as signed64.
No theorem equates that product with its canonical-stride counterpart. -/
 theorem singleton_inactive_wrap :
    (rowOffset64 2 (2^62)).toInt = -(2^63 : Int) ∧
    (rowOffset64 2 1).toInt = 2 ∧ visitedRows 2 1 = [] := by decide

/-- Actual input pointer observations for every active forward load. -/
def ForwardAddresses (c : ForwardCall) (pointers : RegionName → Nat) : Prop :=
  ∀ col, col < c.columns →
    (lanePointer64 (rowPointer64 c.x.pointer c.pid c.x.stride0) col).toNat =
      address pointers (c.x.region,c.pid*c.columns+col)

/-- Actual loop-carried input pointers, plus explicit inactivity. Other source
row strides are the observed canonical output-allocation strides. -/
def BackwardAddresses (c : BackwardCall) (pointers : RegionName → Nat) : Prop :=
  (∀ k,
    loopPointer64 c.x.pointer c.first c.x.stride0 k = rowPointer64 c.x.pointer (c.first+k) c.x.stride0 ∧
    loopPointer64 c.dy.pointer c.first c.dy.stride0 k = rowPointer64 c.dy.pointer (c.first+k) c.dy.stride0 ∧
    loopPointer64 c.dx.pointer c.first c.dx.stride0 k = rowPointer64 c.dx.pointer (c.first+k) c.dx.stride0 ∧
    loopPointer64 c.mean.pointer c.first 1 k = rowPointer64 c.mean.pointer (c.first+k) 1 ∧
    loopPointer64 c.rstd.pointer c.first 1 k = rowPointer64 c.rstd.pointer (c.first+k) 1) ∧
  (∀ row ∈ visitedRows c.first c.last, ∀ col, col < c.columns →
    (lanePointer64 (loopPointer64 c.x.pointer c.first c.x.stride0 (row-c.first)) col).toNat =
      address pointers (c.x.region,row*c.columns+col) ∧
    (lanePointer64 (loopPointer64 c.dy.pointer c.first c.dy.stride0 (row-c.first)) col).toNat =
      address pointers (c.dy.region,row*c.columns+col)) ∧
  (c.last ≤ c.first → visitedRows c.first c.last = []) ∧
  (∀ (α : Type) [Inhabited α] (M : Algebra α) (s : State α), c.last ≤ c.first →
    FP.Structural.range M "_row" c.first c.last 1 (Backward.body c.parameters) s = some s)

 theorem forward_addresses (c : ForwardCall) (h : ForwardDomain c pointers) : ForwardAddresses c pointers := by
  intro col hc
  exact active_pointer64 c.x h.x h.xStride (h.bound _ (by simp [ForwardCall.buffers,ForwardCall.inputs]))
    h.columns_pos c.pid col h.pid_bound hc

 theorem backward_addresses (c : BackwardCall) (h : BackwardDomain c pointers) : BackwardAddresses c pointers := by
  refine ⟨fun k => ⟨loopPointer64_bits _ _ _ k,loopPointer64_bits _ _ _ k,
    loopPointer64_bits _ _ _ k,loopPointer64_bits _ _ _ k,loopPointer64_bits _ _ _ k⟩,?_,inactive_rows _ _,fun _ _ M s inactive => inactive_range M c s inactive⟩
  intro row member col hc
  have bounds := visitedRows_bounds _ _ _ member
  have hr : row < c.rows := lt_of_lt_of_le bounds.2 (min_le_right _ _)
  constructor
  · exact active_loop_pointer64 c.x h.x h.xStride (h.bound _ (by simp [BackwardCall.buffers,BackwardCall.inputs]))
      h.columns_pos c.first row col bounds.1 hr hc
  · exact active_loop_pointer64 c.dy h.dy h.dyStride (h.bound _ (by simp [BackwardCall.buffers,BackwardCall.inputs]))
      h.columns_pos c.first row col bounds.1 hr hc

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
