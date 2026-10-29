import bench.repairs.tritonbench_g.triton_argmax.Proofs.Memory

/-! Both global stages, actual intermediate scalar storage and initialized
padding. The source-combiner graph supplies every stored payload; the only
memory interfaces are scalar reads/writes. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.GlobalMemory
open Source Memory

structure Buffers where
  input : Allocation
  midValue : Allocation
  midIndex : Allocation
  output : Allocation
  deriving Repr

def input {M : Type} (mem : Primitives M) (p : Buffers) (s : M) : Nat → Word :=
  fun i => mem.read 4 s (address p.input 4 i)

def firstProgram {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (snapshot : Nat → Word) (s : M) (pid : Nat) : M :=
  let value := stage1Value ops c snapshot pid
  let index := BitVec.ofNat 64 (stage1Index ops c snapshot pid)
  mem.write 8 (mem.write d.bytes s (address p.midValue d.bytes pid) value)
    (address p.midIndex 8 pid) index

/-- Live loads execute against the current memory. The snapshot form is a
proved consequence of nonaliasing written buffers, not an input hypothesis. -/
def firstLive {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) (pid : Nat) : M :=
  firstProgram mem ops c p (input mem p s) s pid

def firstGrid {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) : M :=
  (List.range c.blocks).foldl (firstLive mem ops c p) s

def secondProgram {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) : M :=
  let mid := fun i => mem.read d.bytes s (address p.midValue d.bytes i)
  let selected := stage2Selected ops c mid
  let payload := mem.read 8 s (address p.midIndex 8 selected.val)
  mem.write 8 s (address p.output 8 0) payload

structure Layout (d : MidFormat) (c : GlobalConfig) (p : Buffers) : Prop where
  inputCapacity : 4*c.size ≤ p.input.bytes
  midCapacity : d.bytes*c.blocks ≤ p.midValue.bytes
  indexCapacity : 8*c.midBlock ≤ p.midIndex.bytes
  outputCapacity : 8 ≤ p.output.bytes
  midFits : c.blocks ≤ c.midBlock
  inputMid : p.midValue.Disjoint p.input
  inputIndex : p.midIndex.Disjoint p.input
  inputOutput : p.output.Disjoint p.input
  midIndex : p.midIndex.Disjoint p.midValue
  midOutput : p.output.Disjoint p.midValue
  indexOutput : p.output.Disjoint p.midIndex

theorem mid_input_apart {d : MidFormat} {c : GlobalConfig} {p : Buffers}
    (layout : Layout d c p) (pid i : Nat) (hp : pid < c.blocks) (hi : i < c.size) :
    Disjoint (address p.midValue d.bytes pid) d.bytes (address p.input 4 i) 4 := by
  apply subword_disjoint _ _ layout.inputMid
  · exact (Nat.mul_le_mul_left _ (show pid+1 ≤ c.blocks by omega)).trans layout.midCapacity
  · exact (Nat.mul_le_mul_left _ (show i+1 ≤ c.size by omega)).trans layout.inputCapacity

theorem index_input_apart {d : MidFormat} {c : GlobalConfig} {p : Buffers}
    (layout : Layout d c p) (pid i : Nat) (hp : pid < c.blocks) (hi : i < c.size) :
    Disjoint (address p.midIndex 8 pid) 8 (address p.input 4 i) 4 := by
  apply subword_disjoint _ _ layout.inputIndex
  · exact (Nat.mul_le_mul_left _ ((Nat.succ_le_iff.mpr hp).trans layout.midFits)).trans layout.indexCapacity
  · exact (Nat.mul_le_mul_left _ (show i+1 ≤ c.size by omega)).trans layout.inputCapacity

theorem first_preserves_input {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (snapshot : Nat → Word)
    (s : M) (pid i : Nat) (hp : pid < c.blocks) (hi : i < c.size) :
    input mem p (firstProgram mem ops c p snapshot s pid) i=input mem p s i := by
  unfold firstProgram input
  rw [mem.readFrame _ _ _ _ _ _ (index_input_apart layout pid i hp hi),
      mem.readFrame _ _ _ _ _ _ (mid_input_apart layout pid i hp hi)]

theorem source_congr {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (a b : Nat → Word) (same : ∀ i,i < c.size → a i=b i) (pid : Nat) :
    stage1Value ops c a pid=stage1Value ops c b pid ∧
      stage1Index ops c a pid=stage1Index ops c b pid := by
  have loaded : c.loaded a ops.negInf32 pid=c.loaded b ops.negInf32 pid := by
    funext lane
    unfold GlobalConfig.loaded
    split
    · exact same _ (by assumption)
    · rfl
  simp only [stage1Value,stage1Index,GlobalConfig.selected,loaded]
  simp

theorem first_snapshot {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (a : Nat → Word) (s : M)
    (same : ∀ i,i < c.size → input mem p s i=a i) (pid : Nat) :
    firstLive mem ops c p s pid=firstProgram mem ops c p a s pid := by
  obtain ⟨value,index⟩ := source_congr ops c (input mem p s) a same pid
  unfold firstLive firstProgram
  rw [value,index]

theorem grid_snapshot {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (jobs : List Nat)
    (valid : ∀ pid,pid∈jobs → pid < c.blocks) (a : Nat → Word) (s : M)
    (same : ∀ i,i < c.size → input mem p s i=a i) :
    jobs.foldl (firstLive mem ops c p) s=jobs.foldl (firstProgram mem ops c p a) s ∧
      ∀ i,i < c.size → input mem p (jobs.foldl (firstLive mem ops c p) s) i=a i := by
  induction jobs generalizing s with
  | nil => exact ⟨rfl,same⟩
  | cons pid rest ih =>
    have hp := valid pid (by simp)
    have first := first_snapshot mem ops c p a s same pid
    have remains : ∀ i,i < c.size → input mem p (firstLive mem ops c p s pid) i=a i := by
      intro i hi
      rw [first,first_preserves_input mem ops c p layout a s pid i hp hi,same i hi]
    have next := ih (fun pid h => valid pid (by simp [h])) _ remains
    refine ⟨?_,next.2⟩
    simpa only [List.foldl_cons,first] using next.1

theorem first_index_readback {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (a : Nat → Word) (s : M) (pid : Nat) :
    mem.read 8 (firstProgram mem ops c p a s pid) (address p.midIndex 8 pid)=
      BitVec.ofNat 64 (stage1Index ops c a pid) := mem.readSame _ _ _ _

theorem first_value_readback {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (a : Nat → Word)
    (s : M) (pid : Nat) (hp : pid < c.blocks) :
    mem.read d.bytes (firstProgram mem ops c p a s pid) (address p.midValue d.bytes pid)=
      stage1Value ops c a pid := by
  unfold firstProgram
  rw [mem.readFrame]
  · exact mem.readSame _ _ _ _
  · apply subword_disjoint _ _ layout.midIndex
    · exact (Nat.mul_le_mul_left _ ((Nat.succ_le_iff.mpr hp).trans layout.midFits)).trans layout.indexCapacity
    · exact (Nat.mul_le_mul_left _ (show pid+1 ≤ c.blocks by omega)).trans layout.midCapacity

theorem second_gather_bound {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (mid : Nat → MidWord d) : (stage2Selected ops c mid).val < c.midBlock :=
  (stage2Selected ops c mid).isLt

theorem second_readback {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) :
    mem.read 8 (secondProgram mem ops c p s) (address p.output 8 0)=
      mem.read 8 s (address p.midIndex 8
        (stage2Selected ops c (fun i => mem.read d.bytes s (address p.midValue d.bytes i))).val) :=
  mem.readSame _ _ _ _

theorem second_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) (point : Address)
    (apart : Disjoint (address p.output 8 0) 8 point 1) :
    mem.byte (secondProgram mem ops c p s) point=mem.byte s point := mem.byteFrame _ _ _ _ _ apart

#axiomsClean first_preserves_input
#axiomsClean source_congr
#axiomsClean grid_snapshot
#axiomsClean first_value_readback
#axiomsClean first_index_readback
#axiomsClean second_gather_bound
#axiomsClean second_readback
#axiomsClean second_frame
end VeriTile.Bench.Repairs.TritonArgmax.GlobalMemory
