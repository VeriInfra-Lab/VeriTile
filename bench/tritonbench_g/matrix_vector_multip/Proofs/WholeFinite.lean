import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeHost

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeFinite
open VeriTile.Triton FP.Structural WholeSource WholeGrid WholeHost
noncomputable section
set_option maxHeartbeats 8000000

abbrev i32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n
abbrev u64 (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

theorem i32_exact {n : Nat} (h : n<2^31) : (i32 n).toInt=(n:Int) := by
  have hn : (i32 n).toNat=n := by
    rw [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (lt_trans h (by norm_num))
  have hd : 2*(i32 n).toNat<2^32 := by rw [hn]; omega
  rw [BitVec.toInt_eq_toNat_of_lt hd,hn]

def chunks (x : Inputs) (c : Config) := (x.columns+c.blockColumns-1)/c.blockColumns
def paddedRows (x : Inputs) (c : Config) := gridSize (x.parameters c)*c.blockRows
def paddedColumns (x : Inputs) (c : Config) := chunks x c*c.blockColumns

def matrixEnvelope (x : Inputs) (c : Config) :=
  (paddedRows x c-1)*x.strideRow+(c.blockColumns-1)*x.strideColumn+paddedColumns x c*x.strideColumn

def vectorEnvelope (x : Inputs) (c : Config) :=
  (c.blockColumns-1)*x.strideVector+paddedColumns x c*x.strideVector

/-- Explicit machine domain for one of the original48 configurations. Integer
products are bounded separately before pointer64 addition: their total need
not fit signed32. Padded lanes and final pointer/loop increments are included. -/
structure ConfigAdmission (x : Inputs) (c : Config) where
  paddedRowsI32 : paddedRows x c<2^31
  paddedColumnsI32 : paddedColumns x c<2^31
  rowProductI32 : (paddedRows x c-1)*x.strideRow<2^31
  matrixColumnI32 : (c.blockColumns-1)*x.strideColumn<2^31
  vectorColumnI32 : (c.blockColumns-1)*x.strideVector<2^31
  matrixIncrementI32 : c.blockColumns*x.strideColumn<2^31
  vectorIncrementI32 : c.blockColumns*x.strideVector<2^31
  matrixPointerEnd : 4*(x.matrixBase+matrixEnvelope x c)+4 ≤ 2^64
  vectorPointerEnd : 4*(x.vectorBase+vectorEnvelope x c)+4 ≤ 2^64

/-- All possible autotune trials are covered, including the final selection.
No contiguity, non-aliasing between inputs, or single-chunk restriction occurs. -/
structure FiniteAdmission {α : Type} (x : Inputs) (entry : Nat → α) (preexisting : Nat → Prop)
    (a : Allocated α x entry preexisting) where
  rowsI32 : x.rows<2^31
  columnsI32 : x.columns<2^31
  stridesI32 : ∀ v∈[x.strideRow,x.strideColumn,x.strideVector],v<2^31
  configs : ∀ c,ConfigAdmission x c
  matrixStart : Nat
  matrixEnd : Nat
  vectorStart : Nat
  vectorEnd : Nat
  matrixBounds : ∀ i j,i<x.rows → j<x.columns →
    matrixStart ≤ x.matrixAddress i j ∧ x.matrixAddress i j<matrixEnd
  vectorBounds : ∀ j,j<x.vectorLength → vectorStart ≤ x.vectorAddress j ∧ x.vectorAddress j<vectorEnd
  matrixProtected : ∀ addr,matrixStart ≤ addr → addr<matrixEnd → preexisting addr
  vectorProtected : ∀ addr,vectorStart ≤ addr → addr<vectorEnd → preexisting addr
  matrixAllocationEnd : 4*matrixEnd ≤ 2^64
  vectorAllocationEnd : 4*vectorEnd ≤ 2^64
  outputAllocationEnd : 4*(a.base+x.rows) ≤ 2^64
  outputPaddedEnd : ∀ c,4*(a.base+(paddedRows x c-1))+4 ≤ 2^64

def rowBits (c : Config) (pid lane : Nat) : BitVec 32 := i32 pid*i32 c.blockRows+i32 lane
def columnBits (c : Config) (step col : Nat) : BitVec 32 := i32 step*i32 c.blockColumns+i32 col
def productBits (n stride : Nat) : BitVec 32 := i32 n*i32 stride

def bytePointer (base offset : Nat) : BitVec 64 := u64 (4*base)+u64 (4*offset)

/-- Signed32 product is widened only after the original multiplication. -/
def signedByteOffset (v : BitVec 32) : BitVec 64 := BitVec.ofInt 64 (4*v.toInt)

theorem productBits_exact (n stride : Nat) (bound : n*stride<2^31) :
    (productBits n stride).toInt=((n*stride:Nat):Int) := by
  unfold productBits
  rw [BitVec.ofNat_mul_ofNat]
  exact i32_exact bound

theorem rowBits_exact (c : Config) (pid lane : Nat) (bound : pid*c.blockRows+lane<2^31) :
    (rowBits c pid lane).toInt=((pid*c.blockRows+lane:Nat):Int) := by
  unfold rowBits
  rw [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact i32_exact bound

theorem columnBits_exact (c : Config) (step col : Nat) (bound : step*c.blockColumns+col<2^31) :
    (columnBits c step col).toInt=((step*c.blockColumns+col:Nat):Int) := by
  unfold columnBits
  rw [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact i32_exact bound

theorem bytePointer_exact (base offset : Nat) (bound : 4*(base+offset)<2^64) :
    (bytePointer base offset).toNat=4*(base+offset) := by
  unfold bytePointer
  rw [BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  rw [show 4*base+4*offset=4*(base+offset) by omega,Nat.mod_eq_of_lt bound]

theorem padded_row (x : Inputs) (c : Config) (pid lane : Nat)
    (hp : pid<gridSize (x.parameters c)) (hl : lane<c.blockRows) :
    pid*c.blockRows+lane<paddedRows x c := by
  unfold paddedRows
  have h := Nat.mul_le_mul_right c.blockRows (show pid+1 ≤ gridSize (x.parameters c) by omega)
  rw [Nat.add_mul,Nat.one_mul] at h
  omega

theorem padded_column (x : Inputs) (c : Config) (step col : Nat)
    (ht : step<chunks x c) (hc : col<c.blockColumns) :
    step*c.blockColumns+col<paddedColumns x c ∧
      step*c.blockColumns+c.blockColumns ≤ paddedColumns x c := by
  unfold paddedColumns
  have h := Nat.mul_le_mul_right c.blockColumns (show step+1 ≤ chunks x c by omega)
  rw [Nat.add_mul,Nat.one_mul] at h
  omega

variable {α : Type} {x : Inputs} {entry : Nat → α} {preexisting : Nat → Prop}

/-- Masks, row/column products, and the final loop backedge all have their
natural-number meanings in the admitted signed32 domain. -/
theorem source_integer_correspondence (a : Allocated α x entry preexisting)
    (f : FiniteAdmission x entry preexisting a) (c : Config) (pid step lane col : Nat)
    (hp : pid<gridSize (x.parameters c)) (ht : step<chunks x c)
    (hl : lane<c.blockRows) (hc : col<c.blockColumns) :
    (rowBits c pid lane).toInt=((pid*c.blockRows+lane:Nat):Int) ∧
    (columnBits c step col).toInt=((step*c.blockColumns+col:Nat):Int) ∧
    (productBits (pid*c.blockRows+lane) x.strideRow).toInt=
      (((pid*c.blockRows+lane)*x.strideRow:Nat):Int) ∧
    (productBits col x.strideColumn).toInt=((col*x.strideColumn:Nat):Int) ∧
    (productBits col x.strideVector).toInt=((col*x.strideVector:Nat):Int) ∧
    (productBits c.blockColumns x.strideColumn).toInt=((c.blockColumns*x.strideColumn:Nat):Int) ∧
    (productBits c.blockColumns x.strideVector).toInt=((c.blockColumns*x.strideVector:Nat):Int) ∧
    (columnBits c step c.blockColumns).toInt=((step*c.blockColumns+c.blockColumns:Nat):Int) ∧
    ((rowBits c pid lane).slt (i32 x.rows)=Bool.true ↔ pid*c.blockRows+lane<x.rows) ∧
    ((columnBits c step col).slt (i32 x.columns)=Bool.true ↔ step*c.blockColumns+col<x.columns) := by
  have rbound := padded_row x c pid lane hp hl
  have cbound := padded_column x c step col ht hc
  have config:=f.configs c
  have r32 : pid*c.blockRows+lane<2^31 := lt_trans rbound config.paddedRowsI32
  have c32 : step*c.blockColumns+col<2^31 := lt_trans cbound.1 config.paddedColumnsI32
  have rp : (pid*c.blockRows+lane)*x.strideRow<2^31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right x.strideRow (by omega)) config.rowProductI32
  have ac : col*x.strideColumn<2^31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right x.strideColumn (by omega)) config.matrixColumnI32
  have bc : col*x.strideVector<2^31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right x.strideVector (by omega)) config.vectorColumnI32
  refine ⟨rowBits_exact c pid lane r32,columnBits_exact c step col c32,
    productBits_exact _ _ rp,productBits_exact _ _ ac,productBits_exact _ _ bc,
    productBits_exact _ _ config.matrixIncrementI32,productBits_exact _ _ config.vectorIncrementI32,
    columnBits_exact c step c.blockColumns (lt_of_le_of_lt cbound.2 config.paddedColumnsI32),?_,?_⟩
  · rw [BitVec.slt_iff_toInt_lt,rowBits_exact c pid lane r32,i32_exact f.rowsI32]
    omega
  · rw [BitVec.slt_iff_toInt_lt,columnBits_exact c step col c32,i32_exact f.columnsI32]
    omega

/-- Pointer64 expressions include all padded initial lanes and every repeated
increment through the last backedge. The combined offset is never truncated
to signed32. This is the source's actual pointer-rebinding recurrence. -/
def matrixPointer (x : Inputs) (c : Config) (pid lane col : Nat) : Nat → BitVec 64
  | 0 => u64 (4*x.matrixBase)+signedByteOffset (productBits (pid*c.blockRows+lane) x.strideRow)+
      signedByteOffset (productBits col x.strideColumn)
  | step+1 => matrixPointer x c pid lane col step+
      signedByteOffset (productBits c.blockColumns x.strideColumn)

def vectorPointer (x : Inputs) (c : Config) (col : Nat) : Nat → BitVec 64
  | 0 => u64 (4*x.vectorBase)+signedByteOffset (productBits col x.strideVector)
  | step+1 => vectorPointer x c col step+signedByteOffset (productBits c.blockColumns x.strideVector)

theorem signedByteOffset_exact (n stride : Nat) (bound : n*stride<2^31) :
    signedByteOffset (productBits n stride)=u64 (4*(n*stride)) := by
  unfold signedByteOffset
  rw [productBits_exact n stride bound]
  rw [show (4:Int)*((n*stride:Nat):Int)=((4*(n*stride):Nat):Int) by norm_cast]
  exact BitVec.ofInt_natCast _ _

theorem matrixPointer_formula (x : Inputs) (c : Config) (f : ConfigAdmission x c)
    (pid lane col step : Nat) (hp : pid<gridSize (x.parameters c))
    (hl : lane<c.blockRows) (hc : col<c.blockColumns) :
    matrixPointer x c pid lane col step=bytePointer x.matrixBase
      ((pid*c.blockRows+lane)*x.strideRow+col*x.strideColumn+(step*c.blockColumns)*x.strideColumn) := by
  have rbound := padded_row x c pid lane hp hl
  have rp : (pid*c.blockRows+lane)*x.strideRow<2^31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right x.strideRow (by omega)) f.rowProductI32
  have cp : col*x.strideColumn<2^31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right x.strideColumn (by omega)) f.matrixColumnI32
  induction step with
  | zero =>
    simp only [matrixPointer,signedByteOffset_exact _ _ rp,signedByteOffset_exact _ _ cp,
      bytePointer,u64,BitVec.ofNat_add_ofNat,Nat.zero_mul,Nat.add_zero]
    congr 1
    ring
  | succ step ih =>
    rw [matrixPointer,ih,signedByteOffset_exact _ _ f.matrixIncrementI32]
    simp only [bytePointer,u64,BitVec.ofNat_add_ofNat]
    congr 1
    ring

theorem vectorPointer_formula (x : Inputs) (c : Config) (f : ConfigAdmission x c)
    (col step : Nat) (hc : col<c.blockColumns) :
    vectorPointer x c col step=bytePointer x.vectorBase
      (col*x.strideVector+(step*c.blockColumns)*x.strideVector) := by
  have cp : col*x.strideVector<2^31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right x.strideVector (by omega)) f.vectorColumnI32
  induction step with
  | zero => simp [vectorPointer,signedByteOffset_exact _ _ cp,bytePointer]
  | succ step ih =>
    rw [vectorPointer,ih,signedByteOffset_exact _ _ f.vectorIncrementI32]
    simp only [bytePointer,u64,BitVec.ofNat_add_ofNat]
    congr 1
    ring

theorem source_pointer_correspondence (a : Allocated α x entry preexisting)
    (f : FiniteAdmission x entry preexisting a) (c : Config) (pid step lane col : Nat)
    (hp : pid<gridSize (x.parameters c)) (ht : step ≤ chunks x c)
    (hl : lane<c.blockRows) (hc : col<c.blockColumns) :
    (matrixPointer x c pid lane col step).toNat=
      4*(x.matrixBase+(pid*c.blockRows+lane)*x.strideRow+col*x.strideColumn+step*c.blockColumns*x.strideColumn) ∧
    (vectorPointer x c col step).toNat=4*(x.vectorBase+col*x.strideVector+step*c.blockColumns*x.strideVector) ∧
    (bytePointer a.base (pid*c.blockRows+lane)).toNat=4*(a.base+(pid*c.blockRows+lane)) := by
  have conf:=f.configs c
  have rb:=padded_row x c pid lane hp hl
  have rle:=Nat.mul_le_mul_right x.strideRow (show pid*c.blockRows+lane ≤ paddedRows x c-1 by omega)
  have acle:=Nat.mul_le_mul_right x.strideColumn (show col ≤ c.blockColumns-1 by omega)
  have bcle:=Nat.mul_le_mul_right x.strideVector (show col ≤ c.blockColumns-1 by omega)
  have offle : step*c.blockColumns ≤ paddedColumns x c := Nat.mul_le_mul_right _ ht
  have aoff:=Nat.mul_le_mul_right x.strideColumn offle
  have boff:=Nat.mul_le_mul_right x.strideVector offle
  have ab:=conf.matrixPointerEnd
  have bb:=conf.vectorPointerEnd
  have ob:=f.outputPaddedEnd c
  unfold matrixEnvelope at ab
  unfold vectorEnvelope at bb
  rw [matrixPointer_formula x c conf pid lane col step hp hl hc,vectorPointer_formula x c conf col step hc]
  refine ⟨?_,?_,bytePointer_exact a.base _ (by omega)⟩
  · simpa only [Nat.add_assoc] using bytePointer_exact x.matrixBase
      ((pid*c.blockRows+lane)*x.strideRow+col*x.strideColumn+step*c.blockColumns*x.strideColumn) (by omega)
  · simpa only [Nat.add_assoc] using bytePointer_exact x.vectorBase
      (col*x.strideVector+step*c.blockColumns*x.strideVector) (by omega)

/-- Actual active loads and stores include every byte of the FP32 word and
respect the original backing allocations. Input allocations may alias. -/
theorem active_accesses (a : Allocated α x entry preexisting)
    (f : FiniteAdmission x entry preexisting a) (compatible : x.columns=x.vectorLength)
    (i j : Nat) (hi : i<x.rows) (hj : j<x.columns) :
    4*(x.matrixAddress i j)+4 ≤ 2^64 ∧ 4*(x.vectorAddress j)+4 ≤ 2^64 ∧
    4*(a.base+i)+4 ≤ 2^64 ∧ preexisting (x.matrixAddress i j) ∧ preexisting (x.vectorAddress j) ∧
    4 ∣ 4*(x.matrixAddress i j) ∧ 4 ∣ 4*(x.vectorAddress j) ∧ 4 ∣ 4*(a.base+i) := by
  have hm:=f.matrixBounds i j hi hj
  have hv:=f.vectorBounds j (by simpa only [←compatible] using hj)
  have hmb:=f.matrixAllocationEnd
  have hvb:=f.vectorAllocationEnd
  have hob:=f.outputAllocationEnd
  exact ⟨by omega,by omega,by omega,f.matrixProtected _ hm.1 hm.2,f.vectorProtected _ hv.1 hv.2,
    dvd_mul_right _ _,dvd_mul_right _ _,dvd_mul_right _ _⟩

theorem output_bytes_separate (a : Allocated α x entry preexisting) (i addr b₁ b₂ : Nat)
    (hi : i<x.rows) (old : preexisting addr) (h₁ : b₁<4) (h₂ : b₂<4) :
    4*(a.base+i)+b₁≠4*addr+b₂ := by
  intro same
  have wordEq : a.base+i=addr := by omega
  exact a.fresh i hi (by simpa only [wordEq] using old)

/-- A concrete rejected boundary: each ABI argument fits signed32, but the
row-times-stride expression does not. This is excluded by rowProductI32. -/
theorem outside_domain_overflow :
    (productBits 2 1073741824).toInt = -2147483648 ∧
    (2:Nat)*1073741824=2147483648 ∧ ¬(2:Nat)*1073741824<2^31 := by decide

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeFinite
