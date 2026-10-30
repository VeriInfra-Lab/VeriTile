import bench.repairs.tritonbench_g.matmul_triton1.Contract

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton FP.Structural
open Revision001 SharedContract TensorBinding
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem source_eq (p : Core.Parameters) : source p = Core.source p := rfl

theorem output_numel (x : Launch) :
    x.c.metadata.numel = rows x.a.metadata * columns x.b.metadata := by
  simp [VeriTile.Torch.TensorMetadata.numel,x.outputShape]

theorem canonical (x : Launch) : FiniteSource.Canonical x.parameters :=
  ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem ready (x : Launch) : FiniteIndex.Ready x.parameters := by
  refine ⟨x.admitted.extent.1,x.admitted.extent.2.2,x.admitted.extent.2.1,
    by norm_num [Launch.parameters],by norm_num [Launch.parameters],
    by norm_num [Launch.parameters],by norm_num [Launch.parameters],x.launch⟩

theorem storage (x : Launch) : FiniteIndex.StorageBounds x.parameters := by
  have a := TensorBinding.span x.a x.la x.admitted.contiguous.1
  have b := TensorBinding.span x.b x.lb x.admitted.contiguous.2
  have c := TensorBinding.span x.c x.lc x.outputContiguous
  rw [matrix_numel x.a.metadata x.admitted.rank.1] at a
  rw [matrix_numel x.b.metadata x.admitted.rank.2,←x.admitted.dimensions] at b
  rw [output_numel] at c
  exact ⟨a,b,c⟩

theorem geometry (x : Launch) : Geometry.Valid (Core.geometry x.parameters) := by
  refine ⟨by norm_num [Core.geometry,Launch.parameters],
    by norm_num [Core.geometry,Launch.parameters],by norm_num [Core.geometry,Launch.parameters],?_⟩
  exact Core.dense_injective _ _ _ _ (rows x.a.metadata*columns x.b.metadata)
    ⟨le_rfl,fun _ _ _ _ => by simp [Launch.parameters,Core.geometry]⟩

theorem offset_bound (x : Launch) (i : Index x) : offset x i < x.c.metadata.numel := by
  rw [output_numel]
  exact FiniteIndex.linear_bound _ _ _ _ i.1.isLt i.2.isLt

theorem index_covers (x : Launch) (n : Nat) (hn : n < x.c.metadata.numel) :
    ∃ i : Index x, offset x i = n := by
  rw [output_numel] at hn
  have hp : 0 < columns x.b.metadata := by nlinarith
  refine ⟨(⟨n / columns x.b.metadata,(Nat.div_lt_iff_lt_mul hp).mpr hn⟩,
    ⟨n % columns x.b.metadata,Nat.mod_lt _ hp⟩),?_⟩
  exact Nat.div_add_mod' _ _

def grid {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (x : Launch) (s : State α) :=
  GridExecution.sequence (fun pid state => FP.Structural.exec (Core.backend rest P)
    (source x.parameters) (GridExecution.atProgram state pid))
    (List.range (Geometry.count (Core.geometry x.parameters))) s

theorem grid_eq {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (x : Launch) (s : State α) : grid rest P x s = Core.grid rest P x.parameters s := rfl

def Post {α : Type} (x : Launch) (before final : State α) (value : Index x → α) : Prop :=
  (∀ i, final.mem x.rc (offset x i) = Cell.mk .real (value i)) ∧
  (∀ R n, (R ≠ x.rc ∨ x.c.metadata.numel ≤ n) → final.mem R n = before.mem R n)

theorem execute {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (x : Launch) (s : State α) :
    ∃ finite typed, FiniteSource.grid P x.parameters s = some finite ∧
      grid rest P x s = some typed ∧ finite.mem = typed.mem ∧
      Post x s typed (fun i => Core.logical P x.parameters s i.1.val i.2.val) := by
  obtain ⟨finite,hf,fr,ff⟩ := FiniteSource.grid_correct rest P x.parameters
    (canonical x) (ready x) (storage x) (geometry x) x.separateA x.separateB s
  obtain ⟨typed,ht,tr,tf⟩ := Core.grid_correct rest P x.parameters
    (geometry x) (by norm_num [Launch.parameters]) x.separateA x.separateB s
  have asOffset (i : Index x) : Geometry.address (Core.geometry x.parameters) i = offset x i := by
    simp [Geometry.address,Core.geometry,Launch.parameters,offset]
  have mem : finite.mem = typed.mem := by
    funext R n
    by_cases hr : R = x.rc
    · subst R
      by_cases hn : n < x.c.metadata.numel
      · obtain ⟨i,hi⟩ := index_covers x n hn
        rw [←hi,←asOffset i]
        exact (fr i trivial).trans (tr i trivial).symm
      · have outside : ∀ i : Index x, True → n ≠ Geometry.address (Core.geometry x.parameters) i := by
          intro i _ eq
          rw [asOffset] at eq
          have := offset_bound x i
          omega
        exact (ff _ _ (Or.inr outside)).trans (tf _ _ (Or.inr outside)).symm
    · exact (ff _ _ (Or.inl hr)).trans (tf _ _ (Or.inl hr)).symm
  refine ⟨finite,typed,hf,ht,mem,?_,?_⟩
  · intro i
    simpa only [asOffset] using tr i trivial
  · intro R n outside
    apply tf
    rcases outside with hr | hn
    · exact Or.inl hr
    · right
      intro i _ eq
      rw [asOffset] at eq
      have := offset_bound x i
      omega

#axiomsClean source_eq
#axiomsClean execute
end
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
