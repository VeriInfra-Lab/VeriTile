import bench.tritonbench_g.matmul_triton1.Proofs.Bounds
import bench.tritonbench_g.matmul_triton1.Proofs.Real
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

structure Tensor where
  (objectId device rank rows columns rowStride columnStride capacity : Nat)
  (region : RegionName)
  (fp32 contiguous negative cuda : Bool)
structure Config where
  (blockRows blockColumns blockInner groupRows : Nat)
def sourceConfig : Config := ⟨32,32,32,1⟩
structure Arguments where
  (a b : Tensor)
  (config : Config)
  (defaultFP32 : Bool)
def Guard (x : Arguments) : Prop :=
  x.a.rank=2 ∧ x.b.rank=2 ∧ x.a.fp32=Bool.true ∧ x.b.fp32=Bool.true ∧
  x.a.cuda=Bool.true ∧ x.b.cuda=Bool.true ∧ x.a.device=0 ∧ x.b.device=0 ∧
  x.defaultFP32=Bool.true ∧ x.a.contiguous=Bool.true ∧ x.b.contiguous=Bool.true ∧
  x.a.negative=Bool.false ∧ x.b.negative=Bool.false ∧ x.a.columns=x.b.rows
instance (x : Arguments) : Decidable (Guard x) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
def Empty (x : Arguments) : Prop := x.a.rows=0 ∨ x.b.columns=0
instance (x : Arguments) : Decidable (Empty x) := inferInstanceAs (Decidable (_ ∨ _))
def Dense (t : Tensor) := Core.DenseView t.rows t.columns t.rowStride t.columnStride t.capacity
def Storage {α : Type} (t : Tensor) (s : State α) : Prop :=
  ∀ r c,r<t.rows → c<t.columns → ∃ value : α,
    s.mem t.region (r*t.rowStride+c*t.columnStride)=Cell.mk .real value
structure Valid (x : Arguments) : Prop where
  guard : Guard x
  selected : x.config=sourceConfig
  aDense : Dense x.a
  bDense : Dense x.b
  blockRows : 0<x.config.blockRows
  blockColumns : 0<x.config.blockColumns
  blockInner : 0<x.config.blockInner
  groupRows : 0<x.config.groupRows

def parameters (x : Arguments) (c : Tensor) : Core.Parameters :=
  ⟨x.a.region,x.b.region,c.region,x.a.rows,x.b.columns,x.a.columns,
    x.a.rowStride,x.a.columnStride,x.b.rowStride,x.b.columnStride,c.rowStride,c.columnStride,
    x.config.blockRows,x.config.blockColumns,x.config.blockInner,x.config.groupRows⟩
abbrev Index (x : Arguments) := Fin x.a.rows × Fin x.b.columns
def address (c : Tensor) (i : Index x) := i.1.val*c.rowStride+i.2.val*c.columnStride

/-- A local torch.empty allocation contract. Fresh storage is reserved without
initializing its contents; the total backing-memory model already carries
arbitrary bytes for it. This does not constrain any matrix result. -/
structure Allocated {α : Type} (x : Arguments) (before after : State α) (c : Tensor) : Prop where
  rank : c.rank=2
  rows : c.rows=x.a.rows
  columns : c.columns=x.b.columns
  dtype : c.fp32=Bool.true
  contiguous : c.contiguous=Bool.true
  device : c.device=x.a.device
  cuda : c.cuda=Bool.true
  resolved : c.negative=Bool.false
  dense : Dense c
  freshA : x.a.region≠c.region
  freshB : x.b.region≠c.region
  freshObjectA : c.objectId≠x.a.objectId
  freshObjectB : c.objectId≠x.b.objectId
  memory : after.mem=before.mem
abbrev Allocator (α : Type) := Arguments → State α → Option (Tensor × State α)
def AllocationLaw (allocate : Allocator α) : Prop :=
  ∀ x, Guard x → ∀ before, ∃ c after, allocate x before=some (c,after) ∧ Allocated x before after c

def call {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Allocator α) (x : Arguments) (s : State α) : Option (Tensor × State α) :=
  if Guard x then do
    let (c,allocated) ← allocate x s
    if Empty x then some (c,allocated)
    else (Core.grid rest P (parameters x c) allocated).map (fun final => (c,final))
  else none

def Post {α : Type} [Inhabited α] (P : Core.Primitives α) (x : Arguments)
    (initial final : State α) (c : Tensor) : Prop :=
  c.rank=2 ∧ c.rows=x.a.rows ∧ c.columns=x.b.columns ∧ c.fp32=Bool.true ∧
  c.contiguous=Bool.true ∧ c.device=x.a.device ∧ c.cuda=Bool.true ∧ c.negative=Bool.false ∧
  c.objectId≠x.a.objectId ∧ c.objectId≠x.b.objectId ∧
  (∀ i : Index x, final.mem c.region (address c i)=
    Cell.mk .real (Core.logical P (parameters x c) initial i.1.val i.2.val)) ∧
  (∀ r o, (r≠c.region ∨ ∀ i : Index x,o≠address c i) → final.mem r o=initial.mem r o)

theorem geometry_valid {α : Type} (x : Arguments) (h : Valid x)
    (initial allocated : State α) (c : Tensor) (a : Allocated x initial allocated c) :
    Geometry.Valid (Core.geometry (parameters x c)) := by
  refine ⟨h.blockRows,h.blockColumns,by change 0<(1:Nat); decide,?_⟩
  have hd : Core.DenseView x.a.rows x.b.columns c.rowStride c.columnStride c.capacity := by
    simpa only [Dense,a.rows,a.columns] using a.dense
  exact Core.dense_injective _ _ _ _ _ hd

theorem correct {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Allocator α) (allocation : AllocationLaw allocate)
    (x : Arguments) (h : Valid x) (initial : State α) :
    ∃ c final, call rest P allocate x initial=some (c,final) ∧ Post P x initial final c := by
  obtain ⟨c,allocated,ha,a⟩ := allocation x h.guard initial
  have hv := geometry_valid x h initial allocated c a
  have unchanged : ∀ r col,Core.logical P (parameters x c) allocated r col=
      Core.logical P (parameters x c) initial r col :=
    Core.logical_preserved P (parameters x c) initial allocated a.freshA a.freshB
      (fun _ _ _ => congrFun (congrFun a.memory _) _)
  by_cases he : Empty x
  · refine ⟨c,allocated,?_,a.rank,a.rows,a.columns,a.dtype,a.contiguous,a.device,a.cuda,a.resolved,
      a.freshObjectA,a.freshObjectB,?_,?_⟩
    · simp [call,h.guard,ha,he]
    · intro i
      rcases he with hm|hn
      · have := i.1.isLt; omega
      · have := i.2.isLt; omega
    · intro r o _; exact congrFun (congrFun a.memory r) o
  · obtain ⟨final,hgrid,hread,hframe⟩ := Core.grid_correct rest P (parameters x c) hv h.blockInner
      a.freshA a.freshB allocated
    refine ⟨c,final,?_,a.rank,a.rows,a.columns,a.dtype,a.contiguous,a.device,a.cuda,a.resolved,
      a.freshObjectA,a.freshObjectB,?_,?_⟩
    · simp [call,h.guard,ha,he,hgrid]
    · intro i
      have hi := hread i trivial
      simpa only [unchanged] using hi
    · intro r o hf
      exact (hframe r o (hf.imp id (fun h i _ => h i))).trans (congrFun (congrFun a.memory r) o)

theorem rejected {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Allocator α) (x : Arguments) (s : State α) (h : ¬Guard x) :
    call rest P allocate x s=none := by simp [call,h]

theorem empty_return {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Allocator α) (x : Arguments) (initial allocated : State α) (c : Tensor)
    (guard : Guard x) (empty : Empty x) (allocation : allocate x initial=some (c,allocated)) :
    call rest P allocate x initial=some (c,allocated) := by simp [call,guard,empty,allocation]

theorem zero_inner {α : Type} [Inhabited α] (P : Core.Primitives α)
    (x : Arguments) (c : Tensor) (s : State α) (hK : x.a.columns=0)
    (hB : 0<x.config.blockInner) (r col : Nat) :
    Core.logical P (parameters x c) s r col=P.zero := by
  have hdiv : (x.config.blockInner-1)/x.config.blockInner=0 := Nat.div_eq_of_lt (by omega)
  simp [Core.logical,Core.steps,Core.inner,parameters,hK,hdiv,Core.logicalAccum]

#axiomsClean geometry_valid
#axiomsClean correct
#axiomsClean rejected
#axiomsClean empty_return
#axiomsClean zero_inner
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Host
