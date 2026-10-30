import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumericProof
namespace VeriTile.Bench.Repairs.Conv2d.KernelNumeric
open VeriTile.Triton FP.Structural KernelBinding KernelGrid
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16384
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Only the zero field is read by the independent array definitions. -/
def arrayISA (zero : α) : ISA α α :=
  ⟨zero,fun _ _ c=>c,id,fun _ _ c=>c,fun _ _ c=>c⟩
def memory (o : Operations α) (t : State (Value α)) : Memory α :=
  fun r a=>scalar o (KernelOperands.memory t r a)
def convert (o : Operations α) (half : Bool) (x : α) : α := if half then o.half x else x

/-- Independent initial-array instruction graph. The source's padded lanes
and h/w/channel-block order are retained, with explicit operand conversion.
The graph contains no source AST, program id, register, output heap, or result. -/
def reference (o : Operations α) (s : Shape) (f : Flags) (m : Memory α) (x : Conv2d.Cell) : α :=
  foldRange (fun h acc=>foldRange (fun w acc=>foldRange
    (fun c acc=>o.lower32
      (fun e=>convert o f.fp16 (refA s (arrayISA o.zero) m x.group x.spatial h w (c+e.val)))
      (fun e=>convert o f.fp16 (refB s (arrayISA o.zero) m x.group x.feature h w (c+e.val))) acc)
    0 s.cg 32 acc) 0 s.kw 1 acc) 0 s.kh 1 o.zero

theorem input_reference (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (pid : Program) (hg : pid.2.2<s.groups) (t : State (Value α))
    (h w c : Nat) (i : Fin 128) (k : Fin 32) :
    input o f.fp16 (parameters s f) (atProgram pid t) h w c (i,k,PUnit.unit)=
      convert o f.fp16 (refA s (arrayISA o.zero) (memory o t) pid.2.2 (pid.1*128+i.val) h w (c+k.val)) := by
  unfold input
  have load:=KernelOperands.input_load s v f pid t (arrayISA (.scalar o.zero)) h w c i k
  change KernelOperands.input (.scalar o.zero) (parameters s f) (atProgram pid t) h w c (i,k,PUnit.unit)=_ at load
  rw [load]
  have eq : scalar o (aLoad s (arrayISA (.scalar o.zero)) (KernelOperands.memory t)
      pid.2.2 (pid.1*128+i.val) h w (c+k.val))=
      aLoad s (arrayISA o.zero) (memory o t) pid.2.2 (pid.1*128+i.val) h w (c+k.val) := by
    unfold aLoad
    split <;> rfl
  rw [eq,aLoad_reference s _ _ _ _ _ _ _ hg]
  rfl

theorem weight_reference (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (pid : Program) (t : State (Value α)) (h w c : Nat) (k j : Fin 32) :
    weight o f.fp16 (parameters s f) (atProgram pid t) h w c (k,j,PUnit.unit)=
      convert o f.fp16 (refB s (arrayISA o.zero) (memory o t) pid.2.2 (pid.2.1*32+j.val) h w (c+k.val)) := by
  unfold weight
  have load:=KernelOperands.weight_load s v f pid t (arrayISA (.scalar o.zero)) h w c k j
  change KernelOperands.weight (.scalar o.zero) (parameters s f) (atProgram pid t) h w c (k,j,PUnit.unit)=_ at load
  rw [load]
  have eq : scalar o (bLoad s (arrayISA (.scalar o.zero)) (KernelOperands.memory t)
      pid.2.2 (pid.2.1*32+j.val) h w (c+k.val))=
      bLoad s (arrayISA o.zero) (memory o t) pid.2.2 (pid.2.1*32+j.val) h w (c+k.val) := by
    unfold bLoad
    split <;> rfl
  rw [eq,bLoad_reference]
  rfl

theorem lane_reference (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (pid : Program) (hg : pid.2.2<s.groups) (t : State (Value α)) (i : Index) :
    laneReference o s f pid t i=reference o s f (memory o t) (cell pid i) := by
  unfold laneReference reference
  simp only [input_reference o s v f pid hg t,weight_reference o s v f pid t,cell,cellOf,lane]

theorem point_reference (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (t : State (Value α)) (x : Conv2d.Cell) (hx : s.Active x) :
    KernelGrid.point (algebra o) (halfDot o) (defaultDot o) s f t x=
      .scalar (reference o s f (memory o t) x) := by
  unfold KernelGrid.point
  rw [source_reference o s v f,lane_reference o s v f _ hx.1,cell_index]

#specNonCircular reference avoiding [KernelSource.kernel,KernelGrid.launch,KernelGrid.value,KernelOperands.input,KernelOperands.weight]
#axiomsClean input_reference
#axiomsClean weight_reference
#axiomsClean point_reference
end
end VeriTile.Bench.Repairs.Conv2d.KernelNumeric
