import bench.repairs.tritonbench_g.square_matrix.Proofs.Indices
import bench.tritonbench_g.add_example.Proofs.FinitePhysical

/-! Reuse the existing typed FP32 byte-memory interface and frame lemmas.
Only row-square address geometry and its row schedule are task-specific. -/
namespace VeriTile.Bench.Repairs.SquareMatrix.Native
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite (bytes SpanDisjoint)
open VeriTile.Torch.FP32 (DataView)
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

abbrev address := VeriTile.Torch.FP32.DataView.address

def view (v : DataView) : PhysicalView :=
  ⟨BitVec.ofNat 64 v.base,v.base,v.base,id,4*v.elements,.fp32⟩

structure Layout (x : TensorMetadata) where
  input : DataView
  output : DataView
  inputSize : input.elements=x.numel
  outputSize : output.elements=x.numel
  inputAlignment : input.base%4=0
  outputAlignment : output.base%4=0
  fresh : (view input).Disjoint (view output)

structure LocalMemory {M A : Type} (ops : MemoryPrimitives M A)
    {x : TensorMetadata} (L : Layout x) : Prop where
  readable : ∀ k, k < 4*x.numel → ops.readableByte ops.launchDevice (L.input.base+k)
  writable : ∀ k, k < 4*x.numel → ops.writableByte ops.launchDevice (L.output.base+k)
  read_write : ∀ s a v, ops.read (ops.write s .fp32 a v) .fp32 a=v

theorem inside (x : TensorMetadata) (v : DataView) (size : v.elements=x.numel)
    (i : Nat) (hi : i<x.numel) : (view v).Inside (address v i) := by
  have span := v.spanFits
  simp only [PhysicalView.Inside,view,address,DataView.address,bytes]
  rw [size] at span ⊢
  omega

def indices (n row : Nat) : List Nat := (List.range n).map fun c => row*n+c

theorem indices_mem (n row i : Nat) : i∈indices n row ↔ row * n ≤ i ∧ i < row * n + n := by
  simp only [indices,List.mem_map,List.mem_range]
  constructor
  · rintro ⟨c,hc,rfl⟩; omega
  · rintro ⟨lo,hi⟩; exact ⟨i-row*n,by omega,by omega⟩

theorem indices_bound (x : TensorMetadata) (h : InputContract x) (row i : Nat)
    (hr : row<rows x) (hi : i∈indices (cols x) row) : i<x.numel := by
  have bounds := (indices_mem _ _ _).mp hi
  have upper := Nat.mul_le_mul_right (cols x) (Nat.succ_le_of_lt hr)
  rw [Nat.succ_mul] at upper
  rw [numel_eq x h]
  omega

def result {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    {x : TensorMetadata} (L : Layout x) (s : M) (i : Nat) : A :=
  let v := ops.read s .fp32 (address L.input i)
  mul v v

def writes {A : Type} (v : DataView) (values : Nat → A) (is : List Nat) : List (Write A) :=
  is.map fun i => ⟨address v i,values i⟩

/-- Load the complete row before its masked stores, preserving source order.
The two finite pointer additions and the signed mask are the source operations. -/
def sourceRow {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    {x : TensorMetadata} (L : Layout x) (row : Nat) (s : M) : M :=
  let active := (List.range (block (cols x))).filter (Indices.mask (cols x))
  let stores := active.map fun c =>
    let value := ops.read s .fp32 ((Indices.pointer (BitVec.ofNat 64 L.input.base) (cols x) row c).toNat)
    (⟨(Indices.pointer (BitVec.ofNat 64 L.output.base) (cols x) row c).toNat,mul value value⟩ : Write A)
  applyWrites ops .fp32 stores s

theorem filter_range (B n : Nat) (cover : n≤B) :
    (List.range B).filter (fun c => decide (c<n))=List.range n := by
  induction B generalizing n with
  | zero =>
    have hz : n=0 := by omega
    simp [hz]
  | succ B ih =>
    by_cases h : n≤B
    · simp [List.range_succ,List.filter_append,ih n h,show ¬B<n by omega]
    · have eq : n=B+1 := by omega
      subst n
      apply List.filter_eq_self.mpr
      intro c hc
      simpa only [decide_eq_true_eq] using List.mem_range.mp hc

theorem source_row {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    (x : TensorMetadata) (h : InputContract x) (L : Layout x) (row : Nat)
    (hr : row<rows x) (nonempty : 0<x.numel) (s : M) :
    sourceRow ops mul L row s =
      applyWrites ops .fp32 (writes L.output (result ops mul L s) (indices (cols x) row)) s := by
  have active : (List.range (block (cols x))).filter (Indices.mask (cols x))=List.range (cols x) := by
    rw [List.filter_congr (fun c hc => Indices.mask_exact x h row c hr (List.mem_range.mp hc) nonempty)]
    exact filter_range _ _ (block_covers _)
  simp only [sourceRow,active,writes,indices,List.map_map]
  apply congrArg (fun ws => applyWrites ops .fp32 ws s)
  apply List.map_congr_left
  intro c hc
  have hi := List.mem_range.mp hc
  have ix := Indices.pointer_exact x h L.input.base (by rw [←L.inputSize]; exact L.input.spanFits) row c hr hi
  have io := Indices.pointer_exact x h L.output.base (by rw [←L.outputSize]; exact L.output.spanFits) row c hr hi
  simp only [Function.comp_def,ix.1,io.1,result,address,DataView.address]

theorem address_disjoint (v : DataView) (i j : Nat) (h : i≠j) :
    SpanDisjoint (address v i) (address v i+bytes .fp32) (address v j) (address v j+bytes .fp32) := by
  simp only [SpanDisjoint,address,DataView.address,bytes]
  omega

theorem writes_read {M A : Type} (ops : MemoryPrimitives M A)
    (x : TensorMetadata) (L : Layout x) (localMemory : LocalMemory ops L)
    (values : Nat → A) (is : List Nat) (s : M) (i : Nat) :
    ops.read (applyWrites ops .fp32 (writes L.output values is) s) .fp32 (address L.output i) =
      if i∈is then values i else ops.read s .fp32 (address L.output i) := by
  induction is generalizing s with
  | nil => simp [writes,applyWrites]
  | cons j rest ih =>
    simp only [writes,List.map_cons,applyWrites]
    rw [←writes,ih]
    by_cases hr : i∈rest
    · simp [hr]
    · by_cases he : i=j
      · subst j
        simp only [List.mem_cons,hr,or_false,if_false]
        exact localMemory.read_write s _ _
      · simp only [List.mem_cons,hr,he,or_false,if_false]
        exact ops.read_disjoint s .fp32 _ _ .fp32 _ (address_disjoint _ j i (Ne.symm he))

theorem row_frame {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    (x : TensorMetadata) (h : InputContract x) (L : Layout x) (row : Nat)
    (hr : row<rows x) (nonempty : 0<x.numel) (s : M) :
    Preserved ops (view L.output) (sourceRow ops mul L row s) s := by
  rw [source_row ops mul x h L row hr nonempty s]
  apply writes_preserved ops (view L.output)
  intro w hw
  obtain ⟨i,hi,rfl⟩ := List.mem_map.mp hw
  exact inside x L.output L.outputSize i (indices_bound x h row i hr hi)

theorem result_preserved {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    (x : TensorMetadata) (L : Layout x) (s t : M)
    (frame : Preserved ops (view L.output) t s) (i : Nat) (hi : i<x.numel) :
    result ops mul L t i=result ops mul L s i := by
  have same := input_preserved ops (view L.input) (view L.output) t s frame L.fresh _
    (inside x L.input L.inputSize i hi)
  exact congrArg (fun v => mul v v) same

def runRows {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    {x : TensorMetadata} (L : Layout x) : List Nat → M → M
  | [],s => s
  | row::rest,s => runRows ops mul L rest (sourceRow ops mul L row s)

theorem rows_correct {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    (x : TensorMetadata) (h : InputContract x) (L : Layout x) (localMemory : LocalMemory ops L)
    (schedule : List Nat) (valid : ∀ r∈schedule,r<rows x) (nonempty : 0<x.numel) (s : M) :
    (∀ i,i<x.numel → ops.read (runRows ops mul L schedule s) .fp32 (address L.output i)=
      if ∃ r∈schedule,i∈indices (cols x) r then result ops mul L s i
      else ops.read s .fp32 (address L.output i)) ∧
    Preserved ops (view L.output) (runRows ops mul L schedule s) s := by
  induction schedule generalizing s with
  | nil => exact ⟨by simp [runRows],preserved_refl _ _ _⟩
  | cons row rest ih =>
    have hr := valid row (by simp)
    have frame := row_frame ops mul x h L row hr nonempty s
    obtain ⟨values,tailFrame⟩ := ih (fun r hm => valid r (by simp [hm])) (sourceRow ops mul L row s)
    refine ⟨?_,preserved_trans _ _ _ _ _ tailFrame frame⟩
    intro i hi
    change ops.read (runRows ops mul L rest (sourceRow ops mul L row s)) .fp32 _ = _
    rw [values i hi,result_preserved ops mul x L s _ frame i hi,
      source_row ops mul x h L row hr nonempty s,writes_read ops x L localMemory]
    by_cases restHit : ∃ r∈rest,i∈indices (cols x) r
    · simp [restHit]
    · by_cases headHit : i∈indices (cols x) row <;> simp [restHit,headHit]

theorem launch_correct {M A : Type} (ops : MemoryPrimitives M A) (mul : A → A → A)
    (x : TensorMetadata) (h : InputContract x) (L : Layout x) (localMemory : LocalMemory ops L)
    (schedule : List Nat) (complete : schedule.Perm (List.range (rows x)))
    (nonempty : 0<x.numel) (s : M) :
    (∀ i,i<x.numel → ops.read (runRows ops mul L schedule s) .fp32 (address L.output i)=result ops mul L s i) ∧
    Preserved ops (view L.output) (runRows ops mul L schedule s) s := by
  have valid : ∀ r∈schedule,r<rows x := fun r hr => List.mem_range.mp (complete.mem_iff.mp hr)
  obtain ⟨values,frame⟩ := rows_correct ops mul x h L localMemory schedule valid nonempty s
  refine ⟨?_,frame⟩
  intro i hi
  rw [values i hi]
  apply if_pos
  have positive : 0<cols x := by have size := numel_eq x h; nlinarith
  have size := numel_eq x h
  have hr : i/cols x<rows x := (Nat.div_lt_iff_lt_mul positive).mpr (by omega)
  refine ⟨i/cols x,complete.mem_iff.mpr (List.mem_range.mpr hr),?_⟩
  apply (indices_mem _ _ _).mpr
  have hmod := Nat.mod_lt i positive
  have split := Nat.mod_add_div i (cols x)
  rw [Nat.mul_comm (cols x)] at split
  constructor <;> omega


theorem read_access {M A : Type} (ops : MemoryPrimitives M A)
    (x : TensorMetadata) (L : Layout x) (memory : LocalMemory ops L)
    (i : Nat) (hi : i<x.numel) : ReadOK ops (view L.input) (address L.input i) := by
  have span := L.input.spanFits
  have align := L.inputAlignment
  simp only [ReadOK,view,id_eq,bytes]
  refine ⟨inside x L.input L.inputSize i hi,?_,?_,?_,?_,?_⟩
  · simpa [address,DataView.address,Nat.add_mod] using align
  · simp [address,DataView.address]
  · simp only [address,DataView.address]; rw [L.inputSize]; omega
  · simp only [address,DataView.address]; rw [L.inputSize] at span; omega
  · intro k hk
    constructor
    · have bound : 4*i+k < 4*x.numel := by omega
      simpa [address,DataView.address,Nat.add_assoc] using memory.readable _ bound
    · trivial

theorem write_access {M A : Type} (ops : MemoryPrimitives M A)
    (x : TensorMetadata) (L : Layout x) (memory : LocalMemory ops L)
    (i : Nat) (hi : i<x.numel) : WriteOK ops (view L.output) (address L.output i) := by
  have span := L.output.spanFits
  have align := L.outputAlignment
  simp only [WriteOK,view,id_eq,bytes]
  refine ⟨inside x L.output L.outputSize i hi,?_,?_,?_,?_,?_⟩
  · simpa [address,DataView.address,Nat.add_mod] using align
  · simp [address,DataView.address]
  · simp only [address,DataView.address]; rw [L.outputSize]; omega
  · simp only [address,DataView.address]; rw [L.outputSize] at span; omega
  · intro k hk
    constructor
    · have bound : 4*i+k < 4*x.numel := by omega
      simpa [address,DataView.address,Nat.add_assoc] using memory.writable _ bound
    · trivial

theorem source_accesses {M A : Type} (ops : MemoryPrimitives M A)
    (x : TensorMetadata) (h : InputContract x) (L : Layout x) (memory : LocalMemory ops L)
    (r c : Nat) (hr : r<rows x) (hc : c<cols x) :
    ReadOK ops (view L.input)
      ((Indices.pointer (BitVec.ofNat 64 L.input.base) (cols x) r c).toNat) ∧
    WriteOK ops (view L.output)
      ((Indices.pointer (BitVec.ofNat 64 L.output.base) (cols x) r c).toNat) := by
  have ix := Indices.pointer_exact x h L.input.base (by rw [←L.inputSize]; exact L.input.spanFits) r c hr hc
  have io := Indices.pointer_exact x h L.output.base (by rw [←L.outputSize]; exact L.output.spanFits) r c hr hc
  have hi : r*cols x+c<x.numel := by rw [numel_eq x h]; exact flat_lt _ _ _ _ hr hc
  rw [ix.1,io.1]
  exact ⟨read_access ops x L memory _ hi,write_access ops x L memory _ hi⟩

#axiomsClean source_accesses

#axiomsClean source_row
#axiomsClean rows_correct
#axiomsClean launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.Native
