import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Contract
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Grid
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Real
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Native
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Model
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

/-- Unused scalar slots are concrete. The complete-source theorem below holds
for every replacement of these slots, so no result depends on this choice. -/
def unused {α : Type} (z : α) : Algebra α where
  literal := fun _ _ _ => z
  negInf := z
  binary := fun _ _ _ _ _ => z
  unary := fun _ _ _ => z
  cast := fun _ _ _ x => x
  fromNat := fun _ _ => z
  fromInt := fun _ _ => z
  fp32Bits := fun _ => z
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => z
  reduceSum := fun _ _ _ _ _ _ => z

def parameters (c : Arguments) (choice : Fin 8) : Core.Parameters :=
  ⟨"A","B","C",rows c.a.metadata,columns c.b.metadata,columns c.a.metadata,
    columns c.a.metadata,1,columns c.b.metadata,1,columns c.b.metadata,1,
    (Native.config choice).blockRows,(Native.config choice).blockColumns,
    (Native.config choice).blockInner,(Native.config choice).groupRows⟩

def address (t : Tensor) (i : Nat) : Nat := t.metadata.dataPtr.toNat+4*i

def entry {α : Type} (read : Nat → α) (c : Arguments) (other : State α) : State α :=
  {other with mem := fun r i =>
    if r="A" then .mk .real (read (address c.a i))
    else if r="B" then .mk .real (read (address c.b i))
    else if r="C" then .mk .real (read (address c.output i))
    else other.mem r i}

def run {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (c : Arguments) (choice : Fin 8) (before : State α) :=
  Core.grid rest P (parameters c choice) before

theorem geometry (c : Arguments) (choice : Fin 8) :
    Geometry.Valid (Core.geometry (parameters c choice)) := by
  obtain ⟨hm,hn,hk,hg,_⟩ := Native.config_legal choice
  refine ⟨hm,hn,hg,?_⟩
  exact Core.dense_injective (rows c.a.metadata) (columns c.b.metadata)
    (columns c.b.metadata) 1 (rows c.a.metadata*columns c.b.metadata)
    ⟨le_refl _,by intro r col hr hc; simp⟩

theorem complete {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (c : Arguments) (choice : Fin 8) (before : State α) :
    ∃ after,run rest P c choice before=some after ∧
      (∀ r col,r < rows c.a.metadata → col < columns c.b.metadata →
        after.mem "C" (r*columns c.b.metadata+col)=.mk .real
          (Core.logical P (parameters c choice) before r col)) ∧
      (∀ tag i,tag ≠ ("C" : RegionName) ∨ rows c.a.metadata*columns c.b.metadata ≤ i →
        after.mem tag i=before.mem tag i) := by
  obtain ⟨after,ran,values,frame⟩ := Core.grid_correct rest P (parameters c choice)
    (geometry c choice) (Native.config_legal choice).2.2.1
    (by change ("A" : RegionName) ≠ "C"; decide)
    (by change ("B" : RegionName) ≠ "C"; decide) before
  refine ⟨after,ran,?_,?_⟩
  · intro r col hr hc
    simpa only [Geometry.address,Core.geometry,parameters,Nat.mul_one] using values (⟨r,hr⟩,⟨col,hc⟩) trivial
  · intro tag i outside
    apply frame tag i
    rcases outside with outside|outside
    · exact .inl outside
    · right
      intro x _ same
      have hr : x.1.val < rows c.a.metadata := x.1.isLt
      have hc : x.2.val < columns c.b.metadata := x.2.isLt
      change i=x.1.val*columns c.b.metadata+x.2.val*1 at same
      have bound : x.1.val*columns c.b.metadata+x.2.val < rows c.a.metadata*columns c.b.metadata := by nlinarith
      omega

theorem empty {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (c : Arguments) (choice : Fin 8) (before : State α)
    (empty : rows c.a.metadata=0 ∨ columns c.b.metadata=0) :
    run rest P c choice before=some before := by
  have blocks := Native.config_legal choice
  have zero (k : Nat) (h : 0 < k) : (0+k-1)/k=0 := Nat.div_eq_of_lt (by omega)
  have count : Geometry.count (Core.geometry (parameters c choice))=0 := by
    rcases empty with hm|hn
    · simp only [Geometry.count,Geometry.gridRows,Geometry.gridColumns,Core.geometry,parameters,hm,zero _ blocks.1,Nat.zero_mul]
    · simp only [Geometry.count,Geometry.gridRows,Geometry.gridColumns,Core.geometry,parameters,hn,zero _ blocks.2.1,Nat.mul_zero]
  simp [run,Core.grid,count,GridExecution.sequence]

#axiomsClean complete
#axiomsClean empty
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Model
