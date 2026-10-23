import bench.repairs.tritonbench_g.matmul_triton1.revision001.Correct
import bench.repairs.tritonbench_g.matmul_triton1.revision001.FPEquiv
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Kernels
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Reader
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
  exact matmul_triton1_correct rest allocate allocation x h initial _ha _hb

theorem whole_software (rest : Algebra Native.Word)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (h : Host.Valid x) (initial : State Native.Word)
    (ha : Host.Storage x.a initial) (hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives Native.referenceInstructions) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives Native.referenceInstructions)
          (Host.parameters x c) initial i.1.val i.2.val) := by
  exact matmul_triton1_fp_correct rest allocate allocation x h initial ha hb

theorem whole_native (rest : Algebra Native.Word)
    (actual : Native.Instructions) (numeric : Native.Correspondence actual)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (h : Host.Valid x) (initial : State Native.Word)
    (ha : Host.Storage x.a initial) (hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives actual) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives Native.referenceInstructions)
          (Host.parameters x c) initial i.1.val i.2.val) := by
  exact matmul_triton1_native_correct rest actual numeric allocate allocation x h initial ha hb

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

theorem fma_step (I : Native.Instructions) {n : Nat}
    (a b : Fin (n+1) → Native.Word) (z : Native.Word) :
    Native.dot I a b z=I.fma (a (Fin.last n)) (b (Fin.last n))
      (Native.dot I (fun i => a i.castSucc) (fun i => b i.castSucc) z) := by
  exact Native.dot_step I a b z

theorem scalar_K_coverage (n : Nat) (k : Fin n) : k∈List.finRange n := by
  exact Native.all_inner_lanes n k

theorem source_grid_rowmajor (p : Core.Parameters) (pid : Nat)
    (hp : pid<Geometry.count (Core.geometry p)) :
    Geometry.blockRow (Core.geometry p) pid=pid/Core.blocksN p ∧
    Geometry.blockColumn (Core.geometry p) pid=pid%Core.blocksN p := by
  exact Core.geometry_coordinates p pid hp

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

theorem local_numeric_consistent : Native.Correspondence Native.referenceInstructions := by
  exact Native.correspondence_consistent

theorem local_to_whole_call (actual : Native.Instructions)
    (numeric : Native.Correspondence actual)
    (rest : Algebra Native.Word) (allocate : Host.Allocator Native.Word)
    (x : Host.Arguments) (s : State Native.Word) :
    Host.call rest (Native.primitives actual) allocate x s=
      Host.call rest (Native.primitives Native.referenceInstructions) allocate x s := by
  exact Native.host_eq actual numeric rest allocate x s

#axiomsClean whole_real
#axiomsClean whole_native
#axiomsClean whole_software
#axiomsClean load_a_bounds
#axiomsClean load_b_bounds
#axiomsClean store_bounds
#axiomsClean grid_covers
#axiomsClean grid_unique
#axiomsClean grid_disjoint
#axiomsClean inner_zero
#axiomsClean empty_host
#axiomsClean rejects
#axiomsClean fma_step
#axiomsClean scalar_K_coverage
#axiomsClean source_grid_rowmajor
#axiomsClean loop_start_before_inner
#axiomsClean local_numeric_consistent
#axiomsClean local_to_whole_call
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Reader
