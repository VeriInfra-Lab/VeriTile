import bench.repairs.tritonbench_g.triton_argmax.Proofs.GlobalGrid
import bench.repairs.tritonbench_g.triton_argmax.Proofs.Native

/-! Byte execution of the complete dimension grid. Each row retains its full
source loop. Invalid row lanes perform no store; valid rows write disjoint i64
cells. The raw graph uses live loads, with snapshot equality derived below. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.DimMemory
open Source Memory

structure Buffers where
  input : Allocation
  output : Allocation

structure Layout (c : DimConfig) (p : Buffers) : Prop where
  inputCapacity : 4*(c.outer*c.extent*c.inner) ≤ p.input.bytes
  outputCapacity : 8*(c.outer*c.inner) ≤ p.output.bytes
  apart : p.output.Disjoint p.input

abbrev Job := Nat × Nat

def input {M : Type} (mem : Primitives M) (p : Buffers) (s : M) : Nat → Word :=
  fun i => mem.read 4 s (address p.input 4 i)

def jobs (c : DimConfig) : List Job :=
  (List.range ((c.outer+127)/128)).flatMap fun pid =>
    (List.range c.inner).flatMap fun inner =>
      (List.range 128).map fun lane => (pid*128+lane,inner)

theorem job_inner (c : DimConfig) (j : Job) (member : j∈jobs c) : j.2 < c.inner := by
  simp only [jobs,List.mem_flatMap,List.mem_map,List.mem_range] at member
  obtain ⟨pid,_,inner,hi,lane,_,equal⟩ := member
  subst j
  exact hi

theorem job_coverage (c : DimConfig) (row inner : Nat) (hr : row < c.outer) (hi : inner < c.inner) :
    (row,inner)∈jobs c := by
  obtain ⟨pid,split,lane⟩ := Native.grid_row_coverage c.outer row hr
  simp only [jobs,List.mem_flatMap,List.mem_map,List.mem_range]
  exact ⟨row/128,pid,inner,hi,row%128,lane,by simp only [split]⟩

def key (c : DimConfig) (j : Job) : Nat := dimOutputOffset c j.1 j.2

def value {d : MidFormat} (ops : Scalars d) (c : DimConfig) (a : Nat → Word) (j : Job) : BitVec 64 :=
  BitVec.ofNat 64 (dimIndex ops c a j.1 j.2)

def rowProgram {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (a : Nat → Word) (s : M) (j : Job) : M :=
  if j.1 < c.outer then mem.write 8 s (address p.output 8 (key c j)) (value ops c a j) else s

def liveRow {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (s : M) (j : Job) : M :=
  rowProgram mem ops c p (input mem p s) s j

def run {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (s : M) : M :=
  (jobs c).foldl (liveRow mem ops c p) s

theorem source_congr {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (a b : Nat → Word) (same : ∀ i,i < c.outer*c.extent*c.inner → a i=b i)
    (row inner : Nat) (hi : inner < c.inner) : dimIndex ops c a row inner=dimIndex ops c b row inner := by
  have loaded : ∀ tile,c.loaded a ops.negInf32 row inner tile=c.loaded b ops.negInf32 row inner tile := by
    intro tile
    funext lane
    dsimp only [DimConfig.loaded]
    split
    · rename_i active
      apply same
      exact Native.dim_offset_bound _ _ _ _ _ _ active.1 active.2 hi
    · rfl
  have localSame : ∀ tile,dimLocal ops c a row inner tile=dimLocal ops c b row inner tile := by
    intro tile
    simp only [dimLocal,loaded]
  have step : dimStep ops c a row inner=dimStep ops c b row inner := by
    funext current tile
    simp only [dimStep,localSame]
  simp only [dimIndex,step]

theorem output_input_apart (c : DimConfig) (p : Buffers) (layout : Layout c p)
    (j : Job) (hr : j.1 < c.outer) (hi : j.2 < c.inner)
    (i : Nat) (bound : i < c.outer*c.extent*c.inner) :
    Disjoint (address p.output 8 (key c j)) 8 (address p.input 4 i) 4 := by
  apply subword_disjoint _ _ layout.apart
  · have keyBound := Native.dim_output_bound _ _ _ _ hr hi
    exact (Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr keyBound)).trans layout.outputCapacity
  · exact (Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr bound)).trans layout.inputCapacity

theorem preserves_input {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (layout : Layout c p) (a : Nat → Word)
    (s : M) (j : Job) (hi : j.2 < c.inner) (i : Nat) (bound : i < c.outer*c.extent*c.inner) :
    input mem p (rowProgram mem ops c p a s j) i=input mem p s i := by
  unfold rowProgram
  split
  · unfold input
    exact mem.readFrame _ _ _ _ _ _ (output_input_apart c p layout j (by assumption) hi i bound)
  · rfl

theorem row_snapshot {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (a : Nat → Word) (s : M) (j : Job) (hi : j.2 < c.inner)
    (same : ∀ i,i < c.outer*c.extent*c.inner → input mem p s i=a i) :
    liveRow mem ops c p s j=rowProgram mem ops c p a s j := by
  have index := source_congr ops c (input mem p s) a same j.1 j.2 hi
  unfold liveRow rowProgram value
  rw [index]

theorem grid_snapshot {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (layout : Layout c p) (work : List Job)
    (valid : ∀ j,j∈work → j.2 < c.inner) (a : Nat → Word) (s : M)
    (same : ∀ i,i < c.outer*c.extent*c.inner → input mem p s i=a i) :
    work.foldl (liveRow mem ops c p) s=work.foldl (rowProgram mem ops c p a) s ∧
      ∀ i,i < c.outer*c.extent*c.inner → input mem p (work.foldl (liveRow mem ops c p) s) i=a i := by
  induction work generalizing s with
  | nil => exact ⟨rfl,same⟩
  | cons j rest ih =>
    have hi := valid j (by simp)
    have first := row_snapshot mem ops c p a s j hi same
    have next : ∀ i,i < c.outer*c.extent*c.inner → input mem p (liveRow mem ops c p s j) i=a i := by
      intro i bound
      rw [first,preserves_input mem ops c p layout a s j hi i bound,same i bound]
    have restResult := ih (fun k hk => valid k (by simp [hk])) _ next
    refine ⟨?_,restResult.2⟩
    simpa only [List.foldl_cons,first] using restResult.1

theorem snapshot_stores {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (a : Nat → Word) (work : List Job) (s : M) :
    work.foldl (rowProgram mem ops c p a) s=
      stores mem p.output 8 (key c) (value ops c a) (work.filter fun j => j.1 < c.outer) s := by
  induction work generalizing s with
  | nil => rfl
  | cons j rest ih =>
    by_cases active : j.1 < c.outer
    · simpa [List.foldl_cons,rowProgram,active,stores] using
        ih (mem.write 8 s (address p.output 8 (key c j)) (value ops c a j))
    · simpa [List.foldl_cons,rowProgram,active,stores] using ih s

theorem dimension_readback {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (layout : Layout c p) (s : M)
    (row inner : Nat) (hr : row < c.outer) (hi : inner < c.inner) :
    mem.read 8 (run mem ops c p s) (address p.output 8 (dimOutputOffset c row inner))=
      BitVec.ofNat 64 (dimIndex ops c (input mem p s) row inner) := by
  have snap := grid_snapshot mem ops c p layout (jobs c) (job_inner c) (input mem p s) s (fun _ _ => rfl)
  unfold run
  rw [snap.1,snapshot_stores]
  apply readback mem p.output 8 (by omega) (key c) (value ops c (input mem p s)) _ s (row,inner)
  · simp only [List.mem_filter,decide_eq_true_eq]
    exact ⟨job_coverage c row inner hr hi,hr⟩
  · intro other member same
    have member' := (List.mem_filter.mp member).1
    have otherInner := job_inner c other member'
    have equal := Native.output_unique c.inner other.1 other.2 row inner otherInner hi same
    obtain ⟨r,k⟩ := other
    dsimp only at equal
    obtain ⟨rfl,rfl⟩ := equal
    rfl

theorem dimension_bytes_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : Buffers) (layout : Layout c p) (s : M) (point : Address)
    (apart : ∀ row inner,row < c.outer → inner < c.inner →
      Disjoint (address p.output 8 (dimOutputOffset c row inner)) 8 point 1) :
    mem.byte (run mem ops c p s) point=mem.byte s point := by
  have snap := grid_snapshot mem ops c p layout (jobs c) (job_inner c) (input mem p s) s (fun _ _ => rfl)
  unfold run
  rw [snap.1,snapshot_stores]
  apply bytes_frame
  intro j member
  have facts := List.mem_filter.mp member
  exact apart j.1 j.2 (by simpa using facts.2) (job_inner c j facts.1)

#axiomsClean job_coverage
#axiomsClean source_congr
#axiomsClean preserves_input
#axiomsClean grid_snapshot
#axiomsClean dimension_readback
#axiomsClean dimension_bytes_frame
end VeriTile.Bench.Repairs.TritonArgmax.DimMemory
