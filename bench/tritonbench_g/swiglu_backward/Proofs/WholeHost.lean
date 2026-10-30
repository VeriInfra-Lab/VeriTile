import bench.tritonbench_g.swiglu_backward.Proofs.WholeGrid
import bench.tritonbench_g.swiglu_fwd.Proofs.WholeHost

namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
open VeriTile.Triton FP.Structural
noncomputable section
open scoped Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

inductive Config where
  | b32 | b64 | b128 | b256 | b512 | b1024
  deriving DecidableEq, Repr

def Config.block : Config → Nat
  | .b32 => 32 | .b64 => 64 | .b128 => 128
  | .b256 => 256 | .b512 => 512 | .b1024 => 1024

theorem Config.positive (c : Config) : 0 < c.block := by cases c <;> decide

structure Launch (c : Layout) where
  config : Config
  order : List (Nat × Nat)
  schedule : order.Perm (grid c config.block)

def runLaunch [Inhabited α] (M : Algebra α) (c : Layout) (R : Bool)
    (launch : Launch c) (s : State α) : Option (State α) :=
  runTiles M c launch.config.block R launch.order s

def runTrials [Inhabited α] (M : Algebra α) (c : Layout) (R : Bool) :
    List (Launch c) → State α → Option (State α)
  | [], s => some s
  | launch :: rest, s => (runLaunch M c R launch s).bind (runTrials M c R rest)

theorem trials_frame [Inhabited α] (M : Algebra α) (c : Layout) (R : Bool)
    (own : Ownership c R) (trials : List (Launch c)) (s : State α) :
    ∃ final, runTrials M c R trials s = some final ∧
      ∀ region a, WholeOutside c R region a → final.mem region a = s.mem region a := by
  induction trials generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons launch rest ih =>
    obtain ⟨middle,ran,_,frame⟩ := grid_result M c launch.config.block R
      launch.config.positive own launch.order launch.schedule s
    obtain ⟨final,tail,tailFrame⟩ := ih middle
    refine ⟨final,?_,?_⟩
    · simp only [runTrials,runLaunch,ran,Option.bind_some,tail]
    · intro region a outside
      exact (tailFrame region a outside).trans (frame region a outside)

def tunedLaunch [Inhabited α] (M : Algebra α) (c : Layout) (R : Bool)
    (trials : List (Launch c)) (selected : Launch c) (s : State α) : Option (State α) :=
  (runTrials M c R trials s).bind (runLaunch M c R selected)

theorem tuned_result [Inhabited α] (M : Algebra α) (c : Layout) (R : Bool)
    (own : Ownership c R) (trials : List (Launch c)) (selected : Launch c) (s : State α) :
    ∃ final, tunedLaunch M c R trials selected s = some final ∧
      (∀ ch, enabled R ch → ∀ r q, r < c.rows → q < c.cols →
        final.mem (c.output ch) (r*c.stride ch+q) = Cell.mk .real (scalar M ch
          ((s.mem c.X (r*c.sx+q)).read .real) ((s.mem c.Y (r*c.sy+q)).read .real)
          ((s.mem c.DOUT (r*c.sd+q)).read .real))) ∧
      (∀ region a, WholeOutside c R region a → final.mem region a = s.mem region a) := by
  obtain ⟨middle,tried,trialFrame⟩ := trials_frame M c R own trials s
  obtain ⟨final,ran,values,frame⟩ := grid_result M c selected.config.block R
    selected.config.positive own selected.order selected.schedule middle
  refine ⟨final,?_,?_,?_⟩
  · simp only [tunedLaunch,runLaunch,tried,Option.bind_some,ran]
  · intro ch on r q hr hq
    rw [values ch on r q hr hq,
      trialFrame c.X _ (fun ch on => Or.inl (own.xSeparate ch on)),
      trialFrame c.Y _ (fun ch on => Or.inl (own.ySeparate ch on)),
      trialFrame c.DOUT _ (fun ch on => Or.inl (own.dSeparate ch on))]
  · intro region a outside
    exact (frame region a outside).trans (trialFrame region a outside)

abbrev PhysicalMemory := SwigluFwd.Whole.PhysicalMemory
abbrev Tensor := SwigluFwd.Whole.Tensor
abbrev Matrix := SwigluFwd.Whole.Matrix

structure Call where
  input : Tensor
  upstream : Tensor
  gradient : Option Tensor
  output : Option Tensor
  recompute : Bool

/-- Ordinary contiguous/reshape contract on original logical input values.
The two source operands have independent normalization and reshape paths. -/
structure Flattened (α : Type) (input : Tensor) (entry memory : PhysicalMemory α) where
  normalized : Tensor
  matrix : Matrix
  normalizedShape : normalized.shape = input.shape
  normalizedDevice : normalized.cudaDevice = input.cudaDevice
  matrixDevice : matrix.cudaDevice = input.cudaDevice
  noCopy : input.lastStride = 1 → normalized = input
  copy : input.lastStride ≠ 1 → input.shape.prod ≠ 0 → 1 < input.shape.getLastD 0 → normalized.lastStride = 1
  normalizedValues : ∀ k, k < input.shape.prod → normalized.read memory k = input.read entry k
  rowsCols : matrix.rows*matrix.cols = input.shape.prod
  columns : matrix.cols = input.shape.getLastD 0
  lastStride : matrix.rows = 0 ∨ matrix.cols ≤ 1 ∨ matrix.lastStride = 1
  reshapeValues : ∀ r q, r < matrix.rows → q < matrix.cols →
    matrix.read memory r q = normalized.read memory (r*matrix.cols+q)

/-- Torch may retain arbitrary last stride for empty tensors and singleton
columns even after contiguous(). Those layouts are valid: no live offset
uses a nonzero column in the singleton case. -/
theorem Flattened.unit_offset (f : Flattened α input entry memory) (r q : Nat)
    (hr : r < f.matrix.rows) (hq : q < f.matrix.cols) : q*f.matrix.lastStride = q := by
  rcases f.lastStride with empty | singleton | unit
  · omega
  · have : q = 0 := by omega
    simp [this]
  · simp [unit]

theorem Flattened.reads (f : Flattened α input entry memory) (r q : Nat)
    (hr : r < f.matrix.rows) (hq : q < f.matrix.cols) :
    f.matrix.read memory r q = input.read entry (r*f.matrix.cols+q) := by
  rw [f.reshapeValues r q hr hq]
  apply f.normalizedValues
  rw [← f.rowsCols]
  have h := Nat.mul_le_mul_right f.matrix.cols (Nat.succ_le_of_lt hr)
  nlinarith

/-- Standard allocator/reshape metadata for a destination; no computed value
is supplied. Reshape copying remains allowed, as in the original wrapper. -/
structure Destination (α : Type) (supplied : Option Tensor) (entry memory : PhysicalMemory α)
    (preexisting : Nat → Prop) (rows cols device : Nat) where
  matrix : Matrix
  suppliedPreexisting : ∀ t, supplied = some t → preexisting t.allocation
  emptyShape : supplied = none → matrix.rows = rows ∧ matrix.cols = cols
  emptyDevice : supplied = none → matrix.cudaDevice = device
  emptyFresh : supplied = none → ¬ preexisting matrix.allocation
  reshapeShape : ∀ t, supplied = some t → matrix.rows*matrix.cols = t.shape.prod ∧
    matrix.cols = t.shape.getLastD 0
  reshapeValues : ∀ t, supplied = some t → ∀ r q, r < matrix.rows → q < matrix.cols →
    matrix.read memory r q = t.read entry (r*matrix.cols+q)

structure Prepared (α : Type) (call : Call) (entry : PhysicalMemory α) (preexisting : Nat → Prop) where
  batchShape : List Nat
  ncols : Nat
  memory : PhysicalMemory α
  inputPreexisting : preexisting call.input.allocation
  upstreamPreexisting : preexisting call.upstream.allocation
  preparationFrame : ∀ allocation a, preexisting allocation → memory allocation a = entry allocation a
  sourceShape : call.input.shape = batchShape ++ [2*ncols]
  xy : Flattened α call.input entry memory
  dout : Flattened α call.upstream entry memory
  inputRows : xy.matrix.rows = batchShape.prod
  inputCols : xy.matrix.cols = 2*ncols
  grad : Destination α call.gradient entry memory preexisting batchShape.prod (2*ncols) call.input.cudaDevice
  out : Destination α (if call.recompute then call.output else none) entry memory preexisting
    batchShape.prod ncols call.input.cudaDevice

/-- Assertions occur at the same branches as Python: dout shape, supplied
dxy shape, both chunk strides, then optional output shape/stride. -/
def checks (p : Prepared α call entry preexisting) : Bool :=
  decide (p.dout.matrix.rows = p.xy.matrix.rows ∧ p.dout.matrix.cols = p.ncols) &&
  (if call.gradient.isSome then decide (p.grad.matrix.rows = p.xy.matrix.rows ∧
    p.grad.matrix.cols = p.xy.matrix.cols) else Bool.true) &&
  decide (p.grad.matrix.lastStride = 1) && decide (p.grad.matrix.lastStride = 1) &&
  (if call.recompute then
    (if call.output.isSome then decide (p.out.matrix.rows = p.xy.matrix.rows ∧
      p.out.matrix.cols = p.ncols) else Bool.true) && decide (p.out.matrix.lastStride = 1)
   else Bool.true)

def Prepared.view (p : Prepared α call entry preexisting) : Channel → Matrix
  | .dx => p.grad.matrix
  | .dy => {p.grad.matrix with base := p.grad.matrix.base+p.ncols}
  | .out => p.out.matrix

def Prepared.location (p : Prepared α call entry preexisting) (ch : Channel) (r q : Nat) : Nat × Nat :=
  ((p.view ch).allocation,(p.view ch).base+r*(p.view ch).rowStride+q)

def Prepared.layout (p : Prepared α call entry preexisting) : Layout :=
  ⟨"X","Y","DOUT","DX","DY","OUT",p.batchShape.prod,p.ncols,
    p.xy.matrix.rowStride,p.xy.matrix.rowStride,p.dout.matrix.rowStride,
    p.grad.matrix.rowStride,p.grad.matrix.rowStride,p.out.matrix.rowStride⟩

structure Domain (p : Prepared α call entry preexisting) where
  positiveColumns : 0 < p.ncols
  sourceChecks : checks p = Bool.true
  upstreamShape : p.dout.matrix.rows = p.batchShape.prod ∧ p.dout.matrix.cols = p.ncols
  gradientShape : p.grad.matrix.rows = p.batchShape.prod ∧ p.grad.matrix.cols = 2*p.ncols
  outputShape : call.recompute = Bool.true → p.out.matrix.rows = p.batchShape.prod ∧ p.out.matrix.cols = p.ncols
  sameDevice : p.dout.matrix.cudaDevice = call.input.cudaDevice ∧
    p.grad.matrix.cudaDevice = call.input.cudaDevice ∧
    (call.recompute = Bool.true → p.out.matrix.cudaDevice = call.input.cudaDevice)
  capacity : Nat → Nat
  originalBounds : ∀ t ∈ [call.input,call.upstream], ∀ k, k < t.shape.prod → t.offset k < capacity t.allocation
  inputBounds : ∀ r q, r < p.batchShape.prod → q < 2*p.ncols →
    p.xy.matrix.address r q < capacity p.xy.matrix.allocation
  upstreamBounds : ∀ r q, r < p.batchShape.prod → q < p.ncols →
    p.dout.matrix.address r q < capacity p.dout.matrix.allocation
  outputBounds : ∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols →
    (p.location ch r q).2 < capacity (p.location ch r q).1
  outputUnique : ∀ ch ch' r q r' q', enabled call.recompute ch → enabled call.recompute ch' →
    r < p.batchShape.prod → q < p.ncols → r' < p.batchShape.prod → q' < p.ncols →
    p.location ch r q = p.location ch' r' q' → ch = ch' ∧ r = r' ∧ q = q'
  inputOwnership : ∀ ch, enabled call.recompute ch → ∀ r q r' q',
    r < p.batchShape.prod → q < p.ncols → r' < p.batchShape.prod → q' < 2*p.ncols →
    p.location ch r q ≠ (p.xy.matrix.allocation,p.xy.matrix.address r' q')
  upstreamOwnership : ∀ ch, enabled call.recompute ch → ∀ r q r' q',
    r < p.batchShape.prod → q < p.ncols → r' < p.batchShape.prod → q' < p.ncols →
    p.location ch r q ≠ (p.dout.matrix.allocation,p.dout.matrix.address r' q')

theorem view_stride (p : Prepared α call entry preexisting) (ch : Channel) :
    (p.view ch).rowStride = p.layout.stride ch := by cases ch <;> rfl

theorem Domain.ownership (p : Prepared α call entry preexisting) (d : Domain p) :
    Ownership p.layout call.recompute := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro ch ch' eq
    cases ch <;> cases ch' <;> simp_all [Prepared.layout,Layout.output]
  · intro ch _; cases ch <;> simp [Prepared.layout,Layout.output]
  · intro ch _; cases ch <;> simp [Prepared.layout,Layout.output]
  · intro ch _; cases ch <;> simp [Prepared.layout,Layout.output]
  · intro ch on r q r' q' hr hq hr' hq' equal
    have loc : p.location ch r q = p.location ch r' q' := by
      simp only [Prepared.location,view_stride,Nat.add_assoc,equal]
    exact (d.outputUnique ch ch r q r' q' on on hr hq hr' hq' loc).2

def Prepared.state [Inhabited α] (p : Prepared α call entry preexisting) : State α where
  mem := fun region a => Cell.mk .real
    (if region = ("X" : RegionName) then p.memory p.xy.matrix.allocation (p.xy.matrix.base+a)
     else if region = ("Y" : RegionName) then p.memory p.xy.matrix.allocation (p.xy.matrix.base+p.ncols+a)
     else if region = ("DOUT" : RegionName) then p.memory p.dout.matrix.allocation (p.dout.matrix.base+a)
     else if region = ("DX" : RegionName) then p.memory p.grad.matrix.allocation (p.grad.matrix.base+a)
     else if region = ("DY" : RegionName) then p.memory p.grad.matrix.allocation (p.grad.matrix.base+p.ncols+a)
     else p.memory p.out.matrix.allocation (p.out.matrix.base+a))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem prepared_reads [Inhabited α] (p : Prepared α call entry preexisting) (d : Domain p)
    (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    ((p.state.mem p.layout.X (r*p.layout.sx+q)).read .real =
      call.input.read entry (r*(2*p.ncols)+q)) ∧
    ((p.state.mem p.layout.Y (r*p.layout.sy+q)).read .real =
      call.input.read entry (r*(2*p.ncols)+(p.ncols+q))) ∧
    ((p.state.mem p.layout.DOUT (r*p.layout.sd+q)).read .real =
      call.upstream.read entry (r*p.ncols+q)) := by
  have hx := p.xy.reads r q (by rwa [p.inputRows]) (by rw [p.inputCols]; omega)
  have hy := p.xy.reads r (p.ncols+q) (by rwa [p.inputRows]) (by rw [p.inputCols]; omega)
  have hd := p.dout.reads r q (by rwa [d.upstreamShape.1]) (by rwa [d.upstreamShape.2])
  refine ⟨?_,?_,?_⟩
  · simpa [Prepared.state,Prepared.layout,SwigluFwd.Whole.Matrix.read,
      SwigluFwd.Whole.Matrix.address,p.xy.unit_offset r q (by rwa [p.inputRows]) (by rw [p.inputCols]; omega),p.inputCols,Nat.add_assoc] using hx
  · simpa [Prepared.state,Prepared.layout,SwigluFwd.Whole.Matrix.read,
      SwigluFwd.Whole.Matrix.address,p.xy.unit_offset r (q+p.ncols) (by rwa [p.inputRows]) (by rw [p.inputCols]; omega),p.inputCols,
      Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using hy
  · simpa [Prepared.state,Prepared.layout,SwigluFwd.Whole.Matrix.read,
      SwigluFwd.Whole.Matrix.address,p.dout.unit_offset r q (by rwa [d.upstreamShape.1]) (by rwa [d.upstreamShape.2]),d.upstreamShape.2,Nat.add_assoc] using hd

/-- A finite logical write has one physical location. The witness is used only
to transport a value already computed by the complete source interpreter. -/
def Written (p : Prepared α call entry preexisting) (allocation a : Nat) : Prop :=
  ∃ ch r q, enabled call.recompute ch ∧ r < p.batchShape.prod ∧ q < p.ncols ∧
    (allocation,a) = p.location ch r q

def Prepared.commit [Inhabited α] (p : Prepared α call entry preexisting)
    (final : State α) : PhysicalMemory α := fun allocation a =>
  if h : Written p allocation a then
    let ch := h.choose
    let r := h.choose_spec.choose
    let q := h.choose_spec.choose_spec.choose
    (final.mem (p.layout.output ch) (r*p.layout.stride ch+q)).read .real
  else p.memory allocation a

theorem commit_value [Inhabited α] (p : Prepared α call entry preexisting) (d : Domain p)
    (final : State α) (ch : Channel) (on : enabled call.recompute ch) (r q : Nat)
    (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    p.commit final (p.location ch r q).1 (p.location ch r q).2 =
      (final.mem (p.layout.output ch) (r*p.layout.stride ch+q)).read .real := by
  have written : Written p (p.location ch r q).1 (p.location ch r q).2 :=
    ⟨ch,r,q,on,hr,hq,Prod.mk.eta⟩
  unfold Prepared.commit
  rw [dif_pos written]
  have witness := written.choose_spec.choose_spec.choose_spec
  have same := d.outputUnique written.choose ch written.choose_spec.choose
    written.choose_spec.choose_spec.choose r q witness.1 on witness.2.1 witness.2.2.1 hr hq
    (witness.2.2.2.symm.trans (Prod.mk.eta))
  rw [same.2.2,same.2.1,same.1]

def PhysicalOutside (p : Prepared α call entry preexisting) (allocation a : Nat) : Prop :=
  ¬ Written p allocation a

theorem commit_frame [Inhabited α] (p : Prepared α call entry preexisting) (final : State α)
    (allocation a : Nat) (outside : PhysicalOutside p allocation a) :
    p.commit final allocation a = p.memory allocation a := by
  simp only [Prepared.commit,PhysicalOutside] at *
  exact dif_neg outside

theorem entry_frame [Inhabited α] (p : Prepared α call entry preexisting) (final : State α)
    (allocation a : Nat) (existing : preexisting allocation) (outside : PhysicalOutside p allocation a) :
    p.commit final allocation a = entry allocation a :=
  (commit_frame p final allocation a outside).trans (p.preparationFrame allocation a existing)

theorem live_inputs_preserved [Inhabited α] (p : Prepared α call entry preexisting) (d : Domain p)
    (final : State α) :
    (∀ r q, r < p.batchShape.prod → q < 2*p.ncols →
      p.xy.matrix.read (p.commit final) r q = p.xy.matrix.read p.memory r q) ∧
    (∀ r q, r < p.batchShape.prod → q < p.ncols →
      p.dout.matrix.read (p.commit final) r q = p.dout.matrix.read p.memory r q) := by
  constructor
  · intro r q hr hq
    apply commit_frame
    rintro ⟨ch,ro,qo,on,hro,hqo,same⟩
    exact d.inputOwnership ch on ro qo r q hro hqo hr hq same.symm
  · intro r q hr hq
    apply commit_frame
    rintro ⟨ch,ro,qo,on,hro,hqo,same⟩
    exact d.upstreamOwnership ch on ro qo r q hro hqo hr hq same.symm

def Prepared.returnShape (p : Prepared α call entry preexisting) : Channel → List Nat
  | .dx | .dy => p.batchShape ++ [p.grad.matrix.cols]
  | .out => p.batchShape ++ [p.out.matrix.cols]

def entryRun [Inhabited α] (M : Algebra α) (p : Prepared α call entry preexisting)
    (trials : List (Launch p.layout)) (selected : Launch p.layout) : Option (State α) :=
  if checks p then tunedLaunch M p.layout call.recompute trials selected p.state else none

theorem entry_result [Inhabited α] (M : Algebra α) (p : Prepared α call entry preexisting)
    (d : Domain p) (trials : List (Launch p.layout)) (selected : Launch p.layout) :
    ∃ final, entryRun M p trials selected = some final ∧
      (∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols →
        p.commit final (p.location ch r q).1 (p.location ch r q).2 = scalar M ch
          (call.input.read entry (r*(2*p.ncols)+q))
          (call.input.read entry (r*(2*p.ncols)+(p.ncols+q)))
          (call.upstream.read entry (r*p.ncols+q))) ∧
      (∀ region a, WholeOutside p.layout call.recompute region a →
        final.mem region a = p.state.mem region a) := by
  obtain ⟨final,ran,values,frame⟩ := tuned_result M p.layout call.recompute
    (Domain.ownership p d) trials selected p.state
  refine ⟨final,?_,?_,frame⟩
  · simp only [entryRun,d.sourceChecks,if_true,ran]
  · intro ch on r q hr hq
    rw [commit_value p d final ch on r q hr hq]
    have value := values ch on r q hr hq
    have reads := prepared_reads p d r q hr hq
    rw [reads.1,reads.2.1,reads.2.2] at value
    rw [value]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
