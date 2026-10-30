import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Host
import bench.tritonbench_g.kldiv_compute.Proofs.Contracts

namespace VeriTile.Bench.Repairs.KldivCompute.Math
open VeriTile.Triton FP.Structural Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor World Preserves)
noncomputable section

/-- Independent mathematics, including Lean's total Real.log/div conventions.
This is a pointwise KL term, not a reduction or a zero-extended KL convention. -/
def formula (x y : ℝ) : ℝ := x * Real.log (x/y)
def algebra : Algebra ℝ where
  literal := fun _ _ r => r
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
    | .max => max x y | .pow => 0
  unary := fun _ op x => match op with | .log => Real.log x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

theorem scalar_formula (x y : ℝ) : Program.scalar algebra x y=formula x y := rfl

/-- Reuse the original independent Real interpreter and flat-memory bridge;
this includes actual successful execution, active output values, and all-cell
frame for every valid disjoint flat placement, arbitrary N/B and input values. -/
theorem original_flat_real (X Y O : RegionName) (N B : Nat) :
    VeriTile.Spec.Real
      (VeriTile.Triton.MaskedKernelIO₂.Implements
        (VeriTile.Bench.TritonBenchG.KldivCompute.typedIO
          .fp32 .fp32 .fp32 Bool.false X Y O N B)
        (fun xs ys i => formula (xs i) (ys i))) := by
  exact VeriTile.Bench.TritonBenchG.KldivCompute.typed_real_correct_bridge
    .fp32 .fp32 .fp32 Bool.false X Y O N B

abbrev RealWorld := World ℝ Unit

/-- Ordinary coordinate-preserving tensor-library contract interpreted on
arbitrary Real cells with the *same actual returned tensor descriptors*.
These are local allocation, resolve/contiguous and copy operations only.
No relation between FP words and Real arithmetic is postulated. -/
structure Shadow {α σ : Type} [Inhabited α] (rt : Runtime α σ) : Prop where
  empty : ∀ a s b t, rt.emptyLike a s=.ok b t → ∀ rs : RealWorld,
    rs.live=s.live → ∃ out : RealWorld, out.live=t.live ∧ Allocation a b rs out
  prepare : ∀ a s b t, Host.prepare rt a s=.ok b t → ∀ rs : RealWorld,
    rs.live=s.live → ∃ out : RealWorld, out.live=t.live ∧
      Materialized Neg.neg a b rs out ∧ b.Dense
  work : ∀ a s b t, rt.emptyWork a s=.ok b t → ∀ rs : RealWorld,
    rs.live=s.live → ∃ out : RealWorld, out.live=t.live ∧ Allocation a b rs out ∧ b.Dense
  copy : ∀ dst src s out t, rt.copy dst src s=.ok out t → ∀ rs : RealWorld,
    rs.live=s.live → dst.shape=src.shape → ∃ last : RealWorld,
      last.live=t.live ∧
      (∀ i, i<dst.numel → value Neg.neg last.kernel dst i=value Neg.neg rs.kernel src i) ∧
      (∀ r i, r≠dst.region → last.kernel.mem r i=rs.kernel.mem r i) ∧
      (∀ i, (∀ j, j<dst.numel → i≠dst.index j) → last.kernel.mem dst.region i=rs.kernel.mem dst.region i)

structure SameCallReal {α σ : Type} [Inhabited α] {neg : α → α} {M : Algebra α}
    {rt : Runtime α σ} {x y : Tensor} {initial : World α σ} {output : Tensor} {final : World α σ}
    (e : ReturnedData neg M rt x y initial output final) (before after : RealWorld) where
  (allocated sx sy work : RealWorld)
  computed : State ℝ
  source_call : Program.grid algebra (params e.px e.py e.work output.numel) work.kernel=some computed
  work_inputs : ∀ i, i<x.numel →
    (work.kernel.mem e.px.region i).read .real=value Neg.neg before.kernel x i ∧
    (work.kernel.mem e.py.region i).read .real=value Neg.neg before.kernel y i
  copy_values : ∀ i, i<output.numel → value Neg.neg after.kernel output i=value Neg.neg computed e.work i
  contents : ∀ i, i<x.numel → value Neg.neg after.kernel output i=
    formula (value Neg.neg before.kernel x i) (value Neg.neg before.kernel y i)
  caller_frame : ∀ r i, before.live r → after.kernel.mem r i=before.kernel.mem r i
  output_padding : ∀ i, (∀ j, j<output.numel → i≠output.index j) →
    after.kernel.mem output.region i=allocated.kernel.mem output.region i

/-- Construct the complete independent Real run along the actual successful
call's allocations, normalized input handles, launch request and copyback.
The output in the conclusion is precisely e's caller-returned output. -/
theorem same_call {α σ : Type} [Inhabited α] (neg : α → α) (M : Algebra α)
    (rt : Runtime α σ) (lib : Library neg rt) (shadow : Shadow rt)
    (x y output : Tensor) (initial final : World α σ)
    (e : ReturnedData neg M rt x y initial output final)
    (before : RealWorld) (live : before.live=initial.live)
    (hx : before.live x.region) (hy : before.live y.region) :
    ∃ after, Nonempty (SameCallReal e before after) := by
  obtain ⟨allocated,la,ca⟩ := shadow.empty _ _ _ _ e.allocation_call before live
  obtain ⟨sx,lx,cx,dx⟩ := shadow.prepare _ _ _ _ e.prepare_x_call allocated la
  obtain ⟨sy,ly,cy,dy⟩ := shadow.prepare _ _ _ _ e.prepare_y_call sx lx
  obtain ⟨sw,lw,cw,dw⟩ := shadow.work _ _ _ _ e.work_call sy ly
  obtain ⟨computed,he,hm⟩ := Program.grid_complete algebra
    (params e.px e.py e.work output.numel) sw.kernel (by change 0<1024; decide)
  let realComputed : RealWorld := {sw with kernel:=computed}
  have lc : realComputed.live=e.computed.live := lw.trans e.launch_data.live.symm
  have shape_work : e.work.shape=x.shape := cw.metadata.shape.trans cx.metadata.shape
  have shape_copy : output.shape=e.work.shape := ca.metadata.shape.trans shape_work.symm
  obtain ⟨after,lf,hcopy,hframe,hpadding⟩ := shadow.copy _ _ _ _ _ e.copy_call realComputed lc shape_copy
  have work_ne_output : e.work.region≠output.region := by
    intro eq
    exact cw.fresh (eq ▸ cy.frame.live _ (cx.frame.live _ ca.live))
  have inputs : ∀ i, i<x.numel →
      (sw.kernel.mem e.px.region i).read .real=value Neg.neg before.kernel x i ∧
      (sw.kernel.mem e.py.region i).read .real=value Neg.neg before.kernel y i := by
    intro i hi
    have hix : i<e.px.numel := by simpa [cx.metadata.numel] using hi
    have hiy : i<y.numel := by simpa [Tensor.numel,e.input_shape] using hi
    have hipy : i<e.py.numel := by simpa [cy.metadata.numel] using hiy
    constructor
    · have h := preserved_value Neg.neg (cy.frame.trans cw.frame) e.px cx.live i
      have c := (cx.values i hi).trans (preserved_value Neg.neg ca.frame x hx i)
      simpa [value,dx.1,dx.2 i hix] using h.trans c
    · have h := preserved_value Neg.neg cw.frame e.py cy.live i
      have c := (cy.values i hiy).trans (preserved_value Neg.neg (ca.frame.trans cx.frame) y hy i)
      simpa [value,dy.1,dy.2 i hipy] using h.trans c
  have memory : computed.mem=fun r i =>
      if r=e.work.region ∧ i<output.numel then
        .mk .real (formula ((sw.kernel.mem e.px.region i).read .real)
          ((sw.kernel.mem e.py.region i).read .real)) else sw.kernel.mem r i := by
    simpa only [params,Program.resultCell,scalar_formula,
      VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory] using hm
  refine ⟨after,⟨⟨allocated,sx,sy,sw,computed,he,inputs,hcopy,?_,?_,?_⟩⟩⟩
  · intro i hi
    have hio : i<output.numel := by simpa [ca.metadata.numel] using hi
    have hiw : i<e.work.numel := by simpa [Tensor.numel,shape_work] using hi
    rw [hcopy i hio]
    simp only [value,dw.1,Bool.false_eq_true,↓reduceIte,dw.2 i hiw,realComputed,memory,
      hio,and_self,↓reduceIte,Cell.read_same,(inputs i hi).1,(inputs i hi).2]
  · intro r i hr
    have no : r≠output.region := by intro eq; exact ca.fresh (eq ▸ hr)
    have nw : r≠e.work.region := by
      intro eq
      exact cw.fresh (eq ▸ ((ca.frame.trans cx.frame).trans cy.frame).live _ hr)
    rw [hframe r i no]
    change computed.mem r i=before.kernel.mem r i
    rw [memory]
    simp only [nw,false_and,↓reduceIte]
    exact (((ca.frame.trans cx.frame).trans cy.frame).trans cw.frame).frame r i hr
  · intro i outside
    rw [hpadding i outside]
    change computed.mem output.region i=allocated.kernel.mem output.region i
    rw [memory]
    simp only [Ne.symm work_ne_output,false_and,↓reduceIte]
    exact ((cx.frame.trans cy.frame).trans cw.frame).frame output.region i ca.live

#specNonCircular formula avoiding [Program.scalar,Program.grid,Host.run]
#specNonCircular Shadow avoiding [Program.scalar,Program.grid,Host.run,formula]
#axiomsClean original_flat_real
#axiomsClean same_call
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Math
