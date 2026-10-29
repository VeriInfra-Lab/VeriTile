import bench.repairs.tritonbench_g.matmul_triton2.revision001.Correct
import bench.repairs.tritonbench_g.matmul_triton2.revision001.FPEquiv
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Kernels
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Reader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩
theorem whole_real (rest : Algebra ℝ) (allocate : Host.Allocator ℝ)
    (allocation : Host.AllocationLaw allocate) (x : Host.Arguments)
    (h : Host.Valid x) (initial : State ℝ)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest Core.realPrimitives allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.realMatmul (Host.parameters x c) initial i.1.val i.2.val) := by
  exact matmul_triton2_correct rest allocate allocation x h initial _ha _hb

theorem whole_native (rest : Algebra Native.Word)
    (actual : Native.Instructions) (choice : Fin 8)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (_selected : x.config=Native.config choice)
    (h : Host.Valid x) (initial : State Native.Word)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives actual (Native.family choice)) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives actual (Native.family choice))
          (Host.parameters x c) initial i.1.val i.2.val) := by
  exact matmul_triton2_native_execution rest actual choice allocate allocation x _selected h initial _ha _hb

theorem whole_fp (rest : Algebra Native.Word)
    (actual : Native.Instructions) (reference : Native.Reference)
    (numeric : Native.Correspondence actual reference) (choice : Fin 8)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (_selected : x.config=Native.config choice)
    (h : Host.Valid x) (initial : State Native.Word)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives actual (Native.family choice)) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives (Native.referenceInstructions reference) (Native.family choice))
          (Host.parameters x c) initial i.1.val i.2.val) := by
  exact matmul_triton2_fp_correct rest actual reference numeric choice allocate allocation x _selected h initial _ha _hb

theorem load_a_bounds {α : Type} (p : Core.Parameters) (s : State α) (cap block : Nat)
    (h : Core.DenseView p.rows (Core.inner p) p.inputRow p.inputInner cap)
    (i : Core.AIndex p) (hr : Core.row p s i.1<p.rows)
    (hk : i.2.1.val< Core.inner p-block*p.blockInner) : Core.aOffset p s block i<cap := by
  exact Core.active_a_bound p s cap block h i hr hk

theorem load_b_bounds {α : Type} (p : Core.Parameters) (s : State α) (cap block : Nat)
    (h : Core.DenseView (Core.inner p) p.columns p.weightInner p.weightColumn cap)
    (i : Core.BIndex p) (hk : i.1.val< Core.inner p-block*p.blockInner)
    (hc : Core.column p s i.2.1<p.columns) : Core.bOffset p s block i<cap := by
  exact Core.active_b_bound p s cap block h i hk hc

theorem store_bounds {α : Type} (p : Core.Parameters) (s : State α) (cap : Nat)
    (h : Core.DenseView p.rows p.columns p.outputRow p.outputColumn cap)
    (i : Core.CIndex p) (hi : Core.active p s i) : Core.cOffset p s i<cap := by
  exact Core.active_c_bound p s cap h i hi

theorem grid_covers (g : Geometry.Layout) (h : Geometry.Valid g) (x : Geometry.Cell g) :
    ∃ pid,pid<Geometry.count g ∧ Geometry.Owns g pid x := by
  exact Geometry.covers g h x

theorem grid_unique (g : Geometry.Layout) (h : Geometry.Valid g) (p q : Nat)
    (hp : p<Geometry.count g) (hq : q<Geometry.count g) (x : Geometry.Cell g)
    (hx : Geometry.Owns g p x) (hy : Geometry.Owns g q x) : p=q := by
  exact Geometry.owner_unique g h p q hp hq x hx hy

theorem grid_disjoint (g : Geometry.Layout) (h : Geometry.Valid g) (p q : Nat)
    (hp : p<Geometry.count g) (hq : q<Geometry.count g) (hne : p≠q)
    (i j : Geometry.Lane g) (hi : Geometry.Active g p i) (hj : Geometry.Active g q j) :
    Geometry.offset g p i≠Geometry.offset g q j := by
  exact Geometry.stores_disjoint g h p q hp hq hne i j hi hj

theorem inner_zero {α : Type} [Inhabited α] (P : Core.Primitives α)
    (x : Host.Arguments) (c : Host.Tensor) (s : State α) (hk : x.a.columns=0)
    (hb : 0<x.config.blockInner) (r col : Nat) :
    Core.logical P (Host.parameters x c) s r col=P.zero := by
  exact Host.zero_inner P x c s hk hb r col

theorem empty_host {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Host.Allocator α) (x : Host.Arguments) (initial allocated : State α) (c : Host.Tensor)
    (guard : Host.Guard x) (empty : Host.Empty x) (allocation : allocate x initial=some (c,allocated)) :
    Host.call rest P allocate x initial=some (c,allocated) := by
  exact Host.empty_return rest P allocate x initial allocated c guard empty allocation

theorem rejects {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Host.Allocator α) (x : Host.Arguments) (s : State α) (h : ¬Host.Guard x) :
    Host.call rest P allocate x s=none := by exact Host.rejected rest P allocate x s h

theorem k8_coverage (k : Nat) (j : Fin k) :
    j.val/8*8+j.val%8=j.val ∧ j.val%8<8 ∧ j.val/8<(k+7)/8 := by
  exact Native.lane_index k j

theorem k8_unique (g h : Nat) (i j : Fin 8) (he : g*8+i.val=h*8+j.val) :
    g=h ∧ i=j := by exact Native.lane_unique g h i j he

theorem k8_complete_blocks (choice : Fin 8) (g : Nat)
    (hg : g<((Native.config choice).blockInner+7)/8) (j : Fin 8) :
    g*8+j.val<(Native.config choice).blockInner := by
  exact Native.all_k8_lanes_in_range choice g hg j

/-- Every executed K iteration starts before K. Thus the source's signed
`K-k*BK` subtraction agrees with the natural subtraction in its load mask. -/
theorem loop_start_before_inner (p : Core.Parameters) (hb : 0<p.blockInner)
    (block : Nat) (h : block<Core.steps p) : block*p.blockInner<Core.inner p := by
  have hn : block+1≤(Core.inner p+p.blockInner-1)/p.blockInner := by
    simpa only [Core.steps] using h
  have hm := Nat.mul_le_mul_right p.blockInner hn
  have hd := Nat.div_mul_le_self (Core.inner p+p.blockInner-1) p.blockInner
  rw [Nat.add_mul,Nat.one_mul] at hm
  omega

theorem local_numeric_consistent (reference : Native.Reference) :
    Native.Correspondence (Native.referenceInstructions reference) reference := by
  exact Native.correspondence_consistent reference

theorem local_to_whole_call (actual : Native.Instructions) (reference : Native.Reference)
    (numeric : Native.Correspondence actual reference) (family : Native.Family)
    (rest : Algebra Native.Word) (allocate : Host.Allocator Native.Word)
    (x : Host.Arguments) (s : State Native.Word) :
    Host.call rest (Native.primitives actual family) allocate x s=
      Host.call rest (Native.primitives (Native.referenceInstructions reference) family) allocate x s := by
  exact Native.host_eq actual reference numeric family rest allocate x s

#axiomsClean whole_real
#axiomsClean whole_native
#axiomsClean whole_fp
#axiomsClean load_a_bounds
#axiomsClean load_b_bounds
#axiomsClean store_bounds
#axiomsClean grid_covers
#axiomsClean grid_unique
#axiomsClean grid_disjoint
#axiomsClean inner_zero
#axiomsClean empty_host
#axiomsClean rejects
#axiomsClean k8_coverage
#axiomsClean k8_unique
#axiomsClean k8_complete_blocks
#axiomsClean loop_start_before_inner
#axiomsClean local_numeric_consistent
#axiomsClean local_to_whole_call
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Reader
