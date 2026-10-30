import bench.repairs.tritonbench_g.triton_matmul.Proofs.CheckedHost
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Native
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FiniteExecution
open VeriTile.Triton FP.Structural FPValue
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 8000000

/-- Finite signed64 source masks and offsets; masked lanes keep the source's
storage-typed zero. The three-argument dot receives the live FP32 carry. -/
def loadA {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (pid block : Nat) (i : AIndex r.parameters) : α :=
  if Index.innerMask r block i.2.1.val then
    FPValue.read r.storage (s.mem r.A (Index.aOffset r pid block i.1.val i.2.1.val).toNat)
  else M.cast none .real (dtype r.storage) (M.literal none .real 0.0)
def loadB {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (pid block : Nat) (i : BIndex r.parameters) : α :=
  if Index.innerMask r block i.1.val then
    FPValue.read r.storage (s.mem r.B (Index.bOffset r pid block i.1.val i.2.1.val).toNat)
  else M.cast none .real (dtype r.storage) (M.literal none .real 0.0)

theorem loads {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (basis : State α) (t : TensorAPI r basis) (s : State α) (pid block : Nat)
    (hp : pid<programs r) (hl : programs r<2^31) (hb : block<r.parameters.numKBlocks) :
    loadA M r s pid block=aValue M r.parameters r.storage (instanceState r pid s) block ∧
    loadB M r s pid block=bValue M r.parameters r.storage (instanceState r pid s) block := by
  constructor
  · funext i
    have mask := Index.inner_mask_word r basis t block i.2.1.val hb i.2.1.isLt
    have off := (Index.offsets_word r basis t pid block i.1.val 0 i.2.1.val hp hl hb i.1.isLt (by decide) i.2.1.isLt).1
    obtain ⟨hr,_⟩ := nonempty_program r pid hp
    have hk := (inner_base_machine r basis t block hb).1
    have bound := safeA_machine r basis t (Index.rowCoordinate r pid i.1.val) (Index.innerCoordinate r block i.2.1.val) hr (by omega)
    have coord := (Index.source_coordinates r s pid i.1 ⟨0,by change 0<256; decide⟩).1
    simp only [loadA,mask,decide_eq_true_eq,aValue,Request.parameters,SourceScope.parameters,BK]
    have test : Index.innerCoordinate r block i.2.1.val<r.inner ↔ i.2.1.val<r.inner-block*SourceScope.blockK r.storage := by
      unfold Index.innerCoordinate; omega
    by_cases active : Index.innerCoordinate r block i.2.1.val<r.inner
    · rw [if_pos active,if_pos (test.mp active),off,Index.word_nat (by omega)]
      change FPValue.read r.storage (s.mem r.A (safeAOffset r _ _))=FPValue.read r.storage (s.mem r.A (aOffset r.parameters (instanceState r pid s) block i))
      congr 2
      change safeCoordinate r.rows _*r.strideAM + safeCoordinate r.inner _*r.strideAK =
        (if row r.parameters (instanceState r pid s) i.1<r.rows then row r.parameters (instanceState r pid s) i.1 else 0)*r.strideAM +
          i.2.1.val*r.strideAK + block*(SourceScope.blockK r.storage*r.strideAK)
      rw [coord]
      simp only [safeCoordinate,if_pos active]
      unfold Index.innerCoordinate
      ring
    · rw [if_neg active,if_neg (test.not.mp active)]
  · funext i
    have mask := Index.inner_mask_word r basis t block i.1.val hb i.1.isLt
    have off := (Index.offsets_word r basis t pid block 0 i.2.1.val i.1.val hp hl hb (by decide) i.2.1.isLt i.1.isLt).2
    obtain ⟨_,hc⟩ := nonempty_program r pid hp
    have hk := (inner_base_machine r basis t block hb).1
    have bound := safeB_machine r basis t (Index.innerCoordinate r block i.1.val) (Index.colCoordinate r pid i.2.1.val) (by omega) hc
    have coord := (Index.source_coordinates r s pid ⟨0,by change 0<128; decide⟩ i.2.1).2
    simp only [loadB,mask,decide_eq_true_eq,bValue,Request.parameters,SourceScope.parameters,BK]
    have test : Index.innerCoordinate r block i.1.val<r.inner ↔ i.1.val<r.inner-block*SourceScope.blockK r.storage := by
      unfold Index.innerCoordinate; omega
    by_cases active : Index.innerCoordinate r block i.1.val<r.inner
    · rw [if_pos active,if_pos (test.mp active),off,Index.word_nat (by omega)]
      change FPValue.read r.storage (s.mem r.B (safeBOffset r _ _))=FPValue.read r.storage (s.mem r.B (bOffset r.parameters (instanceState r pid s) block i))
      congr 2
      change safeCoordinate r.inner _*r.strideBK + safeCoordinate r.columns _*r.strideBN =
        i.1.val*r.strideBK + (if col r.parameters (instanceState r pid s) i.2.1<r.columns then col r.parameters (instanceState r pid s) i.2.1 else 0)*r.strideBN +
          block*(SourceScope.blockK r.storage*r.strideBK)
      rw [coord]
      simp only [safeCoordinate,if_pos active]
      unfold Index.innerCoordinate
      ring
    · rw [if_neg active,if_neg (test.not.mp active)]

def update {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (s : State α) (pid block : Nat) (carry : CIndex r.parameters → α) : CIndex r.parameters → α :=
  D.apply (some .fp32) (dtype r.storage) (dtype r.storage) .real {} (batch := [])
    (loadA M r s pid block) (loadB M r s pid block) carry

def loop {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (s : State α) (pid : Nat) : Nat → BitVec 64 → (CIndex r.parameters → α) → Option (CIndex r.parameters → α)
  | 0,k,carry => if k.slt (Index.ceiling (Index.scalarABI r.inner) (SourceScope.blockK r.storage)) then none else some carry
  | fuel+1,k,carry => if k.slt (Index.ceiling (Index.scalarABI r.inner) (SourceScope.blockK r.storage)) then
      loop M D r s pid fuel (k+Index.word 1) (update M D r s pid k.toNat carry)
    else some carry

theorem loop_eq {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (basis : State α) (t : TensorAPI r basis) (s : State α) (pid : Nat)
    (hp : pid<programs r) (hl : programs r<2^31) (fuel k : Nat)
    (sum : k+fuel=r.parameters.numKBlocks) (carry : CIndex r.parameters → α) :
    loop M D r s pid fuel (Index.word k) carry=
      some (foldRange (FPValue.update M D r.parameters r.storage (instanceState r pid s))
        k r.parameters.numKBlocks 1 carry) := by
  have small := Index.blocks_bound r basis t
  have control (q : Nat) (hq : q≤r.parameters.numKBlocks) :
      (Index.word q).slt (Index.ceiling (Index.scalarABI r.inner) (SourceScope.blockK r.storage))=
        decide (q<r.parameters.numKBlocks) := by
    simpa only [Index.counter_word] using (Index.loop_word r basis t q hq).2.1
  induction fuel generalizing k carry with
  | zero =>
    have stop : ¬k<r.parameters.numKBlocks := by omega
    rw [loop,control k (by omega),foldRange]
    simp only [decide_eq_true_eq,stop,if_false,Nat.one_ne_zero,↓reduceDIte]
  | succ fuel ih =>
    have active : k<r.parameters.numKBlocks := by omega
    rw [loop,control k (by omega),if_pos (by simp [active]),BitVec.ofNat_add_ofNat,
      Index.word_nat (by omega : k<2^64)]
    have step : update M D r s pid k carry=FPValue.update M D r.parameters r.storage (instanceState r pid s) k carry := by
      simp only [update,FPValue.update,(loads M r basis t s pid k hp hl active).1,
        (loads M r basis t s pid k hp hl active).2]
    rw [step,ih (k+1) (by omega)]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,active,↓reduceDIte]

def outputOffset (r : Request) (pid : Nat) (i : CIndex r.parameters) : BitVec 64 :=
  Index.scalarABI (max r.columns 1)*Index.rowBits r pid i.1.val +
    Index.scalarABI 1*Index.colBits r pid i.2.1.val

theorem output_offset {α : Type} (r : Request) (basis : State α) (t : TensorAPI r basis)
    (s : State α) (pid : Nat) (hp : pid<programs r) (hl : programs r<2^31) (i : CIndex r.parameters) :
    (outputOffset r pid i).toNat=Source.cSafeOffset r.parameters (instanceState r pid s) i := by
  have coords := Index.source_coordinates r s pid i.1 i.2.1
  have nonempty := nonempty_program r pid hp
  have bounds := safeC_machine r basis t (Index.rowCoordinate r pid i.1.val)
    (Index.colCoordinate r pid i.2.1.val) nonempty.1 nonempty.2
  rw [outputOffset,Index.row_word r basis t pid i.1.val hp hl i.1.isLt,
    Index.col_word r basis t pid i.2.1.val hp hl i.2.1.isLt]
  simp only [Index.scalar_word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.one_mul]
  rw [Index.word_nat (by change safeCOffset r _ _<2^64; omega),candidate_store_offset,coords.1,coords.2]
  rfl

/-- Folding pure writes changes only memory. Keeping the memory fold explicit
prevents the kernel from normalizing a concrete 128-by-256 state list merely
to project its memory field. -/
theorem fold_write_state {α I : Type} (region : RegionName) (mask : I → Bool)
    (offset : I → Nat) (value : I → Cell α) (items : List I) (s : State α) :
    items.foldl (fun t i => if mask i then t.write region (offset i) (value i) else t) s =
      {s with mem := items.foldl (fun mem i => if mask i then
        fun r o => if r = region ∧ o = offset i then value i else mem r o
        else mem) s.mem} := by
  induction items generalizing s with
  | nil => rfl
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases h : mask i = Bool.true
    · simp only [if_pos h]
      rw [ih]
      rfl
    · simp only [if_neg h, ih]

def writeResult {α : Type} (M : Algebra α) (r : Request) (pid : Nat)
    (carry : CIndex r.parameters → α) (s : State α) : State α :=
  let memory := (TileShape.allIndices [BM r.parameters,BN r.parameters]).foldl (fun mem i =>
    if Index.storeMask r pid i.1.val i.2.1.val then
      (fun region offset => if region = r.C ∧ offset = (outputOffset r pid i).toNat then
        stored r.storage (M.cast none .real (dtype r.storage) (carry i)) else mem region offset)
    else mem) s.mem
  {instanceState r pid s with mem := memory}


theorem writeResult_eq {α : Type} (M : Algebra α) (r : Request) (pid : Nat)
    (carry : CIndex r.parameters → α) (s : State α) :
    writeResult M r pid carry s =
      (TileShape.allIndices [BM r.parameters,BN r.parameters]).foldl (fun current i =>
        if Index.storeMask r pid i.1.val i.2.1.val then
          current.write r.C (outputOffset r pid i).toNat
            (stored r.storage (M.cast none .real (dtype r.storage) (carry i))) else current)
        (instanceState r pid s) :=
  (fold_write_state r.C _ _ _ _ (instanceState r pid s)).symm

def program {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (pid : Nat) (s : State α) : Option (State α) :=
  (loop M D r s pid (Index.ceiling (Index.scalarABI r.inner) (SourceScope.blockK r.storage)).toNat
    (Index.word 0) (fun _ => M.literal (some .fp32) .real 0)).map
      (fun carry => writeResult M r pid carry s)

theorem program_eq {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (basis : State α) (t : TensorAPI r basis) (s : State α) (pid : Nat)
    (hp : pid<programs r) (hl : programs r<2^31) :
    program M D r pid s=some (Source.writtenSafe M D r.parameters r.storage (instanceState r pid s)) := by
  have blocks := congrArg Index.Control.blocks (Index.control_word r basis t pid hp hl)
  change Index.ceiling (Index.scalarABI r.inner) (SourceScope.blockK r.storage)=Index.word r.parameters.numKBlocks at blocks
  have small := Index.blocks_bound r basis t
  simp only [program,writeResult_eq,blocks,Index.word_nat (by omega : r.parameters.numKBlocks<2^64),
    loop_eq M D r basis t s pid hp hl r.parameters.numKBlocks 0 (by omega),Option.map_some,Source.writtenSafe]
  congr 2
  funext current i
  have masks := (Index.typed_output r basis t pid i hp hl).2
  change Index.storeMask r pid i.1.val i.2.1.val=decide (active r.parameters (instanceState r pid s) i) at masks
  simp only [masks,decide_eq_true_eq,output_offset r basis t s pid hp hl i]
  rfl

theorem program_source {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (r : Request)
    (basis : State α) (t : TensorAPI r basis) (s : State α) (pid : Nat)
    (hp : pid<programs r) (hl : programs r<2^31) :
    ∃ actual typed,program M D r pid s=some actual ∧
      FP.Structural.exec (backend M D) (kernel r) (instanceState r pid s)=some typed ∧ actual.mem=typed.mem := by
  obtain ⟨typed,he,hm⟩ := Source.execution_memory M D r (instanceState r pid s)
  refine ⟨_,typed,program_eq M D r basis t s pid hp hl,he,?_⟩
  rw [Source.writtenSafe_eq,hm]

#axiomsClean output_offset
#axiomsClean program_eq
#axiomsClean program_source
#axiomsClean loads
#axiomsClean loop_eq
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FiniteExecution
