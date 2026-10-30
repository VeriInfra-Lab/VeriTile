import bench.tritonbench_g.chunk_gated_attention.Sources.Ownership

/-! Local use of the shared finite-allocation and ordinary-grid semantics.
Flattening commutes with merging disjoint bounded frames; this preserves a
region representative for the next actual host batch. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.FlatBatch
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000

def image (A : FlatAlloc) (P : WriteFootprint) : WriteFootprint := fun cell =>
  cell.1=A.flat ∧ ∃ rg off, P (rg,off) ∧ A.addr rg off=cell.2

theorem frame (A : FlatAlloc) (P : WriteFootprint) (u v : BlockState)
    (h : BlockState.WriteWithin P u v) :
    BlockState.WriteWithin (image A P) (A.flattenState u) (A.flattenState v) := by
  intro rg off hn
  by_cases hr : rg=A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨r,o⟩ := cell
      obtain ⟨_,he,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply h
      intro hw
      exact hn ⟨rfl,r,o,hw,he.symm⟩
  · simp [FlatAlloc.flattenState,hr]

theorem image_decode (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (P : WriteFootprint)
    (hb : ∀ rg off, P (rg,off) → off < A.extent rg)
    (off : Nat) : image A P (A.flat,off) ↔ ∃ rg o, A.decode off=some (rg,o) ∧ P (rg,o) := by
  constructor
  · rintro ⟨_,rg,o,hp,he⟩
    have hbo := hb rg o hp
    have hr : rg ∈ A.regions := by by_contra hn; rw [hcov rg hn] at hbo; omega
    refine ⟨rg,o,?_,hp⟩
    change A.addr rg o=off at he
    rw [← he]
    exact A.decode_addr hd hr hbo
  · rintro ⟨rg,o,he,hp⟩
    exact ⟨rfl,rg,o,hp,(A.decode_sound he).2.1.symm⟩

def liftFrames (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0)
    (k : Kernel) (g : Grid) (u : BlockState) (fs : Kernel.GridFrames k g u)
    (hu : u.undef=fun _ _ => 0) (hs : ∀ i : GridIndex g, k.TraceSafe A.extent (u.withGridIndex i))
    (hok : k.FlattenOk) : Kernel.GridFrames (A.flattenKernel k) g (A.flattenState u) := fun idx =>
  { final := A.flattenState (fs idx).final
    writes := image A (fs idx).writes
    h_exec := by
      have h := A.exec_flatten hd hcov k (u.withGridIndex idx) (hs idx) hok hu
      rw [(fs idx).h_exec,Option.map_some] at h
      exact h
    h_writeWithin := frame A (fs idx).writes _ _ (fs idx).h_writeWithin }

theorem lift_disjoint (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0)
    (k : Kernel) (g : Grid) (u : BlockState) (fs : Kernel.GridFrames k g u)
    (hu : u.undef=fun _ _ => 0) (hs : ∀ i : GridIndex g, k.TraceSafe A.extent (u.withGridIndex i)) (hok : k.FlattenOk)
    (hdis : Kernel.GridWritesDisjoint fs)
    (hb : ∀ i rg off, (fs i).writes (rg,off) → off < A.extent rg) :
    Kernel.GridWritesDisjoint (liftFrames A hd hcov k g u fs hu hs hok) := by
  intro i j hn cell hi hj
  rcases cell with ⟨rg,off⟩
  have hr : rg=A.flat := hi.1
  subst rg
  obtain ⟨r,o,he,hwi⟩ := (image_decode A hd hcov _ (hb i) off).mp hi
  obtain ⟨r',o',he',hwj⟩ := (image_decode A hd hcov _ (hb j) off).mp hj
  have heq : (r,o)=(r',o') := Option.some.inj (he.symm.trans he')
  cases heq
  exact hdis i j hn (r,o) hwi hwj

theorem merge_mem (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0)
    (k : Kernel) (g : Grid) (u : BlockState) (fs : Kernel.GridFrames k g u)
    (hu : u.undef=fun _ _ => 0) (hs : ∀ i : GridIndex g, k.TraceSafe A.extent (u.withGridIndex i)) (hok : k.FlattenOk)
    (hdis : Kernel.GridWritesDisjoint fs)
    (hb : ∀ i rg off, (fs i).writes (rg,off) → off < A.extent rg) :
    (A.flattenState (Kernel.mergeFrames g u fs)).mem =
      (Kernel.mergeFrames g (A.flattenState u) (liftFrames A hd hcov k g u fs hu hs hok)).mem := by
  classical
  let ls := liftFrames A hd hcov k g u fs hu hs hok
  have hld := lift_disjoint A hd hcov k g u fs hu hs hok hdis hb
  funext rg off
  by_cases hr : rg=A.flat
  · subst rg
    cases hdec : A.decode off with
    | none =>
      have hn : ¬ Kernel.GridWriteFootprint ls (A.flat,off) := by
        rintro ⟨i,hi⟩
        obtain ⟨r,o,he,_⟩ := (image_decode A hd hcov _ (hb i) off).mp hi
        rw [hdec] at he
        cases he
      rw [Kernel.mergeFrames_mem_unwritten _ _ hn]
      simp [FlatAlloc.flattenState,FlatAlloc.readFlat,hdec]
    | some cell =>
      obtain ⟨r,o⟩ := cell
      by_cases hw : ∃ i, (fs i).writes (r,o)
      · obtain ⟨i,hi⟩ := hw
        have hwi : (ls i).writes (A.flat,off) :=
          (image_decode A hd hcov _ (hb i) off).mpr ⟨r,o,hdec,hi⟩
        rw [Kernel.mergeFrames_mem_written hld i _ _ hwi]
        simp only [liftFrames,ls,FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat,hdec]
        rw [Kernel.mergeFrames_mem_written hdis i r o hi]
      · have hn : ¬ Kernel.GridWriteFootprint ls (A.flat,off) := by
          rintro ⟨i,hi⟩
          obtain ⟨r',o',he,hwi⟩ := (image_decode A hd hcov _ (hb i) off).mp hi
          have heq : (r,o)=(r',o') := Option.some.inj (hdec.symm.trans he)
          cases heq
          exact hw ⟨i,hwi⟩
        rw [Kernel.mergeFrames_mem_unwritten _ _ hn]
        simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat,hdec]
        rw [Kernel.mergeFrames_mem_unwritten r o hw]
  · have hn : ¬ Kernel.GridWriteFootprint ls (rg,off) := by
      rintro ⟨i,hi⟩
      exact hr hi.1
    rw [Kernel.mergeFrames_mem_unwritten _ _ hn]
    simp [FlatAlloc.flattenState,hr]

theorem merge_eq (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0)
    (k : Kernel) (g : Grid) (u : BlockState) (fs : Kernel.GridFrames k g u)
    (hu : u.undef=fun _ _ => 0) (hs : ∀ i : GridIndex g, k.TraceSafe A.extent (u.withGridIndex i)) (hok : k.FlattenOk)
    (hdis : Kernel.GridWritesDisjoint fs)
    (hb : ∀ i rg off, (fs i).writes (rg,off) → off < A.extent rg) :
    A.flattenState (Kernel.mergeFrames g u fs) =
      Kernel.mergeFrames g (A.flattenState u) (liftFrames A hd hcov k g u fs hu hs hok) := by
  have hm := merge_mem A hd hcov k g u fs hu hs hok hdis hb
  change BlockState.mk _ _ _ _ _ = BlockState.mk _ _ _ _ _
  congr 1

def launched (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0)
    (k : Kernel) (g : Grid) (u : BlockState) (fs : Kernel.GridFrames k g u)
    (hu : u.undef=fun _ _ => 0) (hs : ∀ i : GridIndex g, k.TraceSafe A.extent (u.withGridIndex i)) (hok : k.FlattenOk)
    (hdis : Kernel.GridWritesDisjoint fs)
    (hb : ∀ i rg off, (fs i).writes (rg,off) → off < A.extent rg) :
    Kernel.GridLaunchedOrdinary (A.flattenKernel k) g (A.flattenState u)
      (A.flattenState (Kernel.mergeFrames g u fs)) :=
  ⟨liftFrames A hd hcov k g u fs hu hs hok,
    lift_disjoint A hd hcov k g u fs hu hs hok hdis hb,
    merge_eq A hd hcov k g u fs hu hs hok hdis hb⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.FlatBatch
