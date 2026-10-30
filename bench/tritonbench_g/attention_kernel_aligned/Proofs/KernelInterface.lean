import bench.tritonbench_g.attention_kernel_aligned.Proofs.AcceptedReal
import bench.tritonbench_g.attention_kernel_aligned.Proofs.NativeProjection
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Binding
import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16AcceptedReal
import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16NativeProjection
import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16Binding

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.RawKernel
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
open VeriTile.Bench.Repairs.AttentionKernelAligned Source Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- No Tensor metadata: only raw shape, tile and stride/index conditions. -/
theorem no_grid (p : Parameters) (g : Guard p) (empty : ¬Whole.Nonempty p) :
    programs (grid p)=[] := by
  have bm:=g.selected.blockM_pos p
  by_cases hb : p.batches=0
  · simp [programs,grid,hb]
  by_cases hh : p.heads=0
  · simp [programs,grid,hh]
  have hn : p.length=0 := by unfold Whole.Nonempty at empty;omega
  have divide : (p.length+p.blockM-1)/p.blockM=0 := Nat.div_eq_of_lt (by omega)
  simp [programs,grid,divide]

/-- Complete raw grid, every logical store and the whole exterior-cell frame. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p)
    (before : State α) (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>M) (fun _=>P) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        after.mem p.output (logicalAddress p b h n d)=.mk .fp16
          (answer P p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)
            (lane p g.selected n d hd))) ∧
      (∀ r off,OutsideLogical p r off → after.mem r off=before.mem r off) := by
  by_cases live : Whole.Nonempty p
  · have unique : Unique p (grid p) work before := by
      intro a ha b hb i j hi hj same
      exact grid_unique p g before a (order.mem_iff.mpr ha) b (order.mem_iff.mpr hb) i j hi hj same
    obtain ⟨after,run,values,frame⟩:=run_correct (fun _=>M) (fun _=>P) p (separate live)
      (g.selected.blockN_pos p) (grid p) work before unique
    refine ⟨after,run,?_,?_⟩
    · intro b hb h hh n hn d hd
      have active : Whole.active p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)
          (lane p g.selected n d hd) := by
        change n/p.blockM*p.blockM+n%p.blockM<p.length
        simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
      have covers:=(grid_covers p g.selected b h n hb hh hn).1
      have value:=values _ (order.mem_iff.mp covers) _ active
      rw [physical_address p g.selected before b h n d hh hd] at value
      exact value
    · intro r off outside
      apply frame
      rcases outside with outside|outside
      · exact Or.inl outside
      · right
        intro a ha i hi he
        obtain ⟨hx,hy,hz⟩:=(mem_programs (grid p) a.1 a.2.1 a.2.2).mp (order.mem_iff.mpr ha)
        obtain ⟨_,hB,hH⟩:=head_decode p a.2.1 hy
        exact outside _ hB _ hH _ hi _ i.2.1.isLt he
  · have nil : work=[] := List.nil_perm.mp (by simpa [no_grid p g live] using order)
    refine ⟨before,by simp [nil,runPrograms],?_,fun _ _ _=>rfl⟩
    intro b hb h hh n hn d hd
    exact False.elim (live ⟨by omega,by omega,by omega⟩)

/-- Independent Real bias-attention formula on raw initial input arrays. -/
theorem real (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p)
    (before : State Real.R) (μ : RegionName → Nat → ℝ) (reads : Real.Reads p before μ)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.model) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.head,
        after.mem p.output (logicalAddress p b h n d)=.mk .fp16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,OutsideLogical p r off → after.mem r off=before.mem r off) := by
  obtain ⟨after,run,values,frame⟩:=complete Real.algebra Real.model p g separate before work order
  refine ⟨after,run,?_,frame⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd]
  have hreads : Real.Reads p (start (grid p) (n/p.blockM,b*p.heads+h,0) before) μ := reads
  have active : Source.row p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)
      (lane p g.selected n d hd).1<p.length := by
    change n/p.blockM*p.blockM+n%p.blockM<p.length
    simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
  change Cell.mk .fp16 (Source.outputValue Real.model p _ _)=_
  rw [Real.answer_real p _ μ hreads _ active (g.selected.blockN_pos p),
    Real.expected_at p g.selected before μ b h n d hh hd]

/-- Existing compiler routing boundary expressed entirely in raw parameters. -/
structure CompilerPlan (p : Parameters) where
  serialAbsent : p.nativeSerial=Bool.false
  columns : p.nativeColumns∈[1,2,4,8]
  columns_bound : p.nativeColumns≤p.numWarps
  routes : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.Routes
  routing : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.LocalRouting p routes


def nativeDecode (I : Native.Instructions) (d : Nat) (x : Native.Expr) : BitVec 16 :=
  ((Native.lower (x (Native.physicalOwner d))).eval I.ops).setWidth 16

/-- Independent state-free native recurrence consumes only initial raw reads. -/
def nativeReference (I : Native.Instructions) (p : Parameters) (s : State Native.Expr)
    (b h n d : Nat) (selected : Selected p) (hd : d<p.head) : BitVec 16 :=
  nativeDecode I d (KernelOnly.Specification.outputValue (Native.modelFor p) p
    (KernelOnly.operandReader s) (n/p.blockM) b h (lane p selected n d hd))

theorem algorithm_at (P : Primitives Native.Expr) (p : Parameters) (before : State Native.Expr)
    (x b h : Nat) (hh : h<p.heads) :
    Whole.answer P p (start (grid p) (x,b*p.heads+h,0) before)=
      KernelOnly.Specification.outputValue P p (KernelOnly.operandReader before) x b h := by
  rw [KernelOnly.source_algorithm]
  have hp : 0<p.heads := by omega
  have hd : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hm : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  simp [start,ids,hd,hm] <;> rfl

/-- Complete selected source/native grid on raw typed memory, independent
native recurrence and full typed-cell frame, for either empty or live grids. -/
theorem native (I : Native.Instructions) (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (_plan : CompilerPlan p)
    (before : State Native.Expr) (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>Native.algebraFor p) (fun _=>Native.modelFor p) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        nativeDecode I d ((after.mem p.output (logicalAddress p b h n d)).read .fp16)=
          nativeReference I p before b h n d g.selected hd) ∧
      (∀ r off,OutsideLogical p r off → after.mem r off=before.mem r off) := by
  obtain ⟨after,run,values,frame⟩:=complete (Native.algebraFor p) (Native.modelFor p) p g separate before work order
  refine ⟨after,run,?_,frame⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd,Cell.read_same,algorithm_at _ p before (n/p.blockM) b h hh]
  rfl

theorem empty_kernel {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (p : Parameters) (g : Guard p) (empty : ¬Whole.Nonempty p) (before : State α)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    work=[] ∧ runPrograms (fun _=>M) (fun _=>P) p (grid p) work before=some before := by
  have nil : work=[] := List.nil_perm.mp (by simpa [no_grid p g empty] using order)
  exact ⟨nil,by simp [nil,runPrograms]⟩

#axiomsClean complete
#axiomsClean real
#axiomsClean native
#axiomsClean empty_kernel
#specNonCircular nativeReference avoiding [FP.Structural.exec,runPrograms,Whole.answer]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.RawKernel

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.RawKernel
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
open VeriTile.Bench.Repairs.AttentionKernelAligned.BF16 Source Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- No Tensor metadata: only raw shape, tile and stride/index conditions. -/
theorem no_grid (p : Parameters) (g : Guard p) (empty : ¬Whole.Nonempty p) :
    programs (grid p)=[] := by
  have bm:=g.selected.blockM_pos p
  by_cases hb : p.batches=0
  · simp [programs,grid,hb]
  by_cases hh : p.heads=0
  · simp [programs,grid,hh]
  have hn : p.length=0 := by unfold Whole.Nonempty at empty;omega
  have divide : (p.length+p.blockM-1)/p.blockM=0 := Nat.div_eq_of_lt (by omega)
  simp [programs,grid,divide]

/-- Complete raw grid, every logical store and the whole exterior-cell frame. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p)
    (before : State α) (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>M) (fun _=>P) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        after.mem p.output (logicalAddress p b h n d)=.mk .bf16
          (answer P p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)
            (lane p g.selected n d hd))) ∧
      (∀ r off,OutsideLogical p r off → after.mem r off=before.mem r off) := by
  by_cases live : Whole.Nonempty p
  · have unique : Unique p (grid p) work before := by
      intro a ha b hb i j hi hj same
      exact grid_unique p g before a (order.mem_iff.mpr ha) b (order.mem_iff.mpr hb) i j hi hj same
    obtain ⟨after,run,values,frame⟩:=run_correct (fun _=>M) (fun _=>P) p (separate live)
      (g.selected.blockN_pos p) (grid p) work before unique
    refine ⟨after,run,?_,?_⟩
    · intro b hb h hh n hn d hd
      have active : Whole.active p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)
          (lane p g.selected n d hd) := by
        change n/p.blockM*p.blockM+n%p.blockM<p.length
        simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
      have covers:=(grid_covers p g.selected b h n hb hh hn).1
      have value:=values _ (order.mem_iff.mp covers) _ active
      rw [physical_address p g.selected before b h n d hh hd] at value
      exact value
    · intro r off outside
      apply frame
      rcases outside with outside|outside
      · exact Or.inl outside
      · right
        intro a ha i hi he
        obtain ⟨hx,hy,hz⟩:=(mem_programs (grid p) a.1 a.2.1 a.2.2).mp (order.mem_iff.mpr ha)
        obtain ⟨_,hB,hH⟩:=head_decode p a.2.1 hy
        exact outside _ hB _ hH _ hi _ i.2.1.isLt he
  · have nil : work=[] := List.nil_perm.mp (by simpa [no_grid p g live] using order)
    refine ⟨before,by simp [nil,runPrograms],?_,fun _ _ _=>rfl⟩
    intro b hb h hh n hn d hd
    exact False.elim (live ⟨by omega,by omega,by omega⟩)

/-- Independent Real bias-attention formula on raw initial input arrays. -/
theorem real (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p)
    (before : State Real.R) (μ : RegionName → Nat → ℝ) (reads : Real.Reads p before μ)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.model) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.head,
        after.mem p.output (logicalAddress p b h n d)=.mk .bf16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,OutsideLogical p r off → after.mem r off=before.mem r off) := by
  obtain ⟨after,run,values,frame⟩:=complete Real.algebra Real.model p g separate before work order
  refine ⟨after,run,?_,frame⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd]
  have hreads : Real.Reads p (start (grid p) (n/p.blockM,b*p.heads+h,0) before) μ := reads
  have active : Source.row p (start (grid p) (n/p.blockM,b*p.heads+h,0) before)
      (lane p g.selected n d hd).1<p.length := by
    change n/p.blockM*p.blockM+n%p.blockM<p.length
    simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
  change Cell.mk .bf16 (Source.outputValue Real.model p _ _)=_
  rw [Real.answer_real p _ μ hreads _ active (g.selected.blockN_pos p),
    Real.expected_at p g.selected before μ b h n d hh hd]

/-- Existing compiler routing boundary expressed entirely in raw parameters. -/
structure CompilerPlan (p : Parameters) where
  columns : p.nativeColumns∈[1,2,4,8]
  columns_bound : p.nativeColumns≤p.numWarps
  routes : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.Routes
  routing : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.LocalRouting p routes
  serialSource : Fin 128 → Nat
  denominatorSource : Fin 16 → Fin 8 → Nat
  serialRouting : p.nativeSerial=Bool.true →
    (∀ j : Fin 128,j.val<p.blockN → serialSource j=j.val) ∧
    (∀ row col,denominatorSource row col=row.val)


def nativeDecode (I : Native.Instructions) (d : Nat) (x : Native.Expr) : BitVec 16 :=
  ((Native.lower (x (Native.physicalOwner d))).eval I.ops).setWidth 16

/-- Independent state-free native recurrence consumes only initial raw reads. -/
def nativeReference (I : Native.Instructions) (p : Parameters) (s : State Native.Expr)
    (b h n d : Nat) (selected : Selected p) (hd : d<p.head) : BitVec 16 :=
  nativeDecode I d (KernelOnly.Specification.outputValue (Native.modelFor p) p
    (KernelOnly.operandReader s) (n/p.blockM) b h (lane p selected n d hd))

theorem algorithm_at (P : Primitives Native.Expr) (p : Parameters) (before : State Native.Expr)
    (x b h : Nat) (hh : h<p.heads) :
    Whole.answer P p (start (grid p) (x,b*p.heads+h,0) before)=
      KernelOnly.Specification.outputValue P p (KernelOnly.operandReader before) x b h := by
  rw [KernelOnly.source_algorithm]
  have hp : 0<p.heads := by omega
  have hd : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hh]
    omega
  have hm : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  simp [start,ids,hd,hm] <;> rfl

/-- Complete selected source/native grid on raw typed memory, independent
native recurrence and full typed-cell frame, for either empty or live grids. -/
theorem native (I : Native.Instructions) (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (_plan : CompilerPlan p)
    (before : State Native.Expr) (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>Native.algebraFor p) (fun _=>Native.modelFor p) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        nativeDecode I d ((after.mem p.output (logicalAddress p b h n d)).read .bf16)=
          nativeReference I p before b h n d g.selected hd) ∧
      (∀ r off,OutsideLogical p r off → after.mem r off=before.mem r off) := by
  obtain ⟨after,run,values,frame⟩:=complete (Native.algebraFor p) (Native.modelFor p) p g separate before work order
  refine ⟨after,run,?_,frame⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd,Cell.read_same,algorithm_at _ p before (n/p.blockM) b h hh]
  rfl

theorem empty_kernel {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (p : Parameters) (g : Guard p) (empty : ¬Whole.Nonempty p) (before : State α)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    work=[] ∧ runPrograms (fun _=>M) (fun _=>P) p (grid p) work before=some before := by
  have nil : work=[] := List.nil_perm.mp (by simpa [no_grid p g empty] using order)
  exact ⟨nil,by simp [nil,runPrograms]⟩

#axiomsClean complete
#axiomsClean real
#axiomsClean native
#axiomsClean empty_kernel
#specNonCircular nativeReference avoiding [FP.Structural.exec,runPrograms,Whole.answer]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.RawKernel

