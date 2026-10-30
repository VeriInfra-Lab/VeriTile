import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.IO

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReadProjection
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Separate internal reader IDs only. The source AST depends on strides, not
these IDs. They never impose a separation premise on the physical A/B inputs. -/
def views (b : Buffers) : Buffers :=
  ⟨{b.A with storage:=0},{b.B with storage:=1},{b.C with storage:=2}⟩

theorem source_same (p : Problem) (b : Buffers) : source p (views b)=source p b := rfl

theorem layout (p : Problem) (b : Buffers) (h : Layout p b) : Layout p (views b) where
  aShape := h.aShape
  bShape := h.bShape
  cShape := h.cShape
  aType := h.aType
  bType := h.bType
  cType := h.cType
  aBound := ⟨h.aBound.rows,h.aBound.columns,h.aBound.rowStride,h.aBound.columnStride,
    h.aBound.offset,h.aBound.span,h.aBound.capacity,h.aBound.physical,h.aBound.pointer⟩
  bBound := ⟨h.bBound.rows,h.bBound.columns,h.bBound.rowStride,h.bBound.columnStride,
    h.bBound.offset,h.bBound.span,h.bBound.capacity,h.bBound.physical,h.bBound.pointer⟩
  cBound := ⟨h.cBound.rows,h.cBound.columns,h.cBound.rowStride,h.cBound.columnStride,
    h.cBound.offset,h.cBound.span,h.cBound.capacity,h.cBound.physical,h.cBound.pointer⟩
  outputStrides := h.outputStrides
  aSeparate := by simp [views]
  bSeparate := by simp [views]

/-- Read existing regional values without replacing any actual source state.
Below-view offsets are kept separate so subtraction cannot alias offset zero. -/
def decode (b : Buffers) (s : State α) : Memory α := fun point =>
  if point.1=0 ∧ b.A.offset≤point.2 then (s.mem "A" (point.2-b.A.offset)).read .real
  else if point.1=1 ∧ b.B.offset≤point.2 then (s.mem "B" (point.2-b.B.offset)).read .real
  else if point.1=2 ∧ b.C.offset≤point.2 then (s.mem "C" (point.2-b.C.offset)).read .fp16
  else default

@[simp] theorem decode_a (b : Buffers) (s : State α) (o : Nat) :
    decode b s (0,b.A.offset+o)=(s.mem "A" o).read .real := by simp [decode]
@[simp] theorem decode_b (b : Buffers) (s : State α) (o : Nat) :
    decode b s (1,b.B.offset+o)=(s.mem "B" o).read .real := by simp [decode]
@[simp] theorem decode_c (b : Buffers) (s : State α) (o : Nat) :
    decode b s (2,b.C.offset+o)=(s.mem "C" o).read .fp16 := by simp [decode]

/-- All payload dependence is through decoded A/B reads and the axis-0 PID.
This reuses the existing complete arithmetic recurrence, not a second source. -/
theorem output_congr (M : Algebra α) (D : DotAcc α) (p : Problem) (b : Buffers)
    (s t : State α) (pid : s.pids 0=t.pids 0)
    (ha : ∀ o, (s.mem "A" o).read .real=(t.mem "A" o).read .real)
    (hb : ∀ o, (s.mem "B" o).read .real=(t.mem "B" o).read .real) :
    outputValue M D p b s=outputValue M D p b t := by
  have aeq : aValue M p b s=aValue M p b t := by
    funext k i
    simp only [aValue,aOffset,row,pidM,firstM,pid,groupSize,ha]
  have beq : bValue M p b s=bValue M p b t := by
    funext k i
    simp only [bValue,bOffset,col,pidN,pid,groupSize,firstM,hb]
  have updates : update M D p b s=update M D p b t := by
    funext k acc
    simp only [update,aeq,beq]
  funext i
  simp only [outputValue,accumulation,updates]

/-- A normal form used only to apply the retained physical projection lemma.
Actual regional input cells may have arbitrary tags, as before decoded IO. -/
def normalized (b : Buffers) (s : State α) :=
  programState (views b) (decode b s) (s.pids 0) s

@[simp] theorem normalized_pid (b : Buffers) (s : State α) : (normalized b s).pids 0=s.pids 0 := by
  simp [normalized,programState]
@[simp] theorem normalized_a (b : Buffers) (s : State α) (o : Nat) :
    ((normalized b s).mem "A" o).read .real=(s.mem "A" o).read .real := by
  simp [normalized,programState,views,Cell.read_same]
@[simp] theorem normalized_b (b : Buffers) (s : State α) (o : Nat) :
    ((normalized b s).mem "B" o).read .real=(s.mem "B" o).read .real := by
  simp [normalized,programState,views,Cell.read_same]
@[simp] theorem normalized_c (b : Buffers) (s : State α) (o : Nat) :
    ((normalized b s).mem "C" o).read .fp16=(s.mem "C" o).read .fp16 := by
  simp [normalized,programState,views,Cell.read_same]

/-- Explicit physical-to-logical channel binding from the *same* caller
memory. This allows A and B to share storage and retain arbitrary offsets. -/
def physicalInputs (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers)
    (memory : Memory α) : ChannelValues α (IO.inputs g) :=
  fun i => match i with
    | ⟨0,_⟩ => fun j => memory ((b.views g).A.address j.1.val j.2.1.val)
    | ⟨1,_⟩ => fun j => memory ((b.views g).B.address j.1.val j.2.1.val)
    | ⟨_+2,h⟩ => False.elim (by simp at h)

theorem physical_binding (M : Algebra α) (ops : Operations α)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (memory : Memory α)
    (pid : Nat) (scratch : State α) :
    (IO.matmulIO M ops g b).InputBinding (physicalInputs g b memory)
      (programState (b.views g) memory pid scratch) := by
  intro i j _
  fin_cases i
  · change ((programState (b.views g) memory pid scratch).mem "A"
      (j.1.val*b.A.rowStride+j.2.1.val*b.A.columnStride)).read .real=
      memory ((b.views g).A.address j.1.val j.2.1.val)
    simp [programState,Tensor.address,KernelBuffers.Buffers.views,KernelBuffers.Buffer.view,
      Cell.read_same,Nat.add_assoc]
  · change ((programState (b.views g) memory pid scratch).mem "B"
      (j.1.val*b.B.rowStride+j.2.1.val*b.B.columnStride)).read .real=
      memory ((b.views g).B.address j.1.val j.2.1.val)
    simp [programState,Tensor.address,KernelBuffers.Buffers.views,KernelBuffers.Buffer.view,
      Cell.read_same,Nat.add_assoc]

#axiomsClean output_congr
#axiomsClean physical_binding
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReadProjection
