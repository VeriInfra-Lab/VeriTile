import bench.tritonbench_g.rms_matmul_rbe.Proofs.PrefixValues
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Loop

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def parameters (p : Arguments) (epsilon theta : α) : RmsMatmulRbe.Parameters α :=
  ⟨p.K,p.headDim,p.start,epsilon,theta⟩

def rawRow {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (batch row : Nat) : RmsMatmulRbe.RowInput α where
  x := fun j => (s.mem p.X (batch*p.sxb+row*p.sxm+j*p.sxk)).read .real
  rms := fun j =>
    let half := if fp8 then kernelCast a .f8e5 .fp16
      (a.e5Bits .f8e5 (Float8E5Bits.ofInt ((s.mem p.RMS (j*p.srms)).read .int)))
      else (s.mem p.RMS (j*p.srms)).read .fp16
    kernelCast a .fp16 .real half
  weight := fun n j =>
    let half := if fp8 then kernelCast a .real .fp16 (kernelCast a .f8e5 .real
      (a.e5Bits .f8e5 (Float8E5Bits.ofInt ((s.mem p.W (j*p.swk+n*p.swn)).read .int))))
      else (s.mem p.W (j*p.swk+n*p.swn)).read .fp16
    kernelCast a .fp16 .real half

/-- Local scalar conversion facts for masked zeros. These statements concern
one literal or one conversion chain each, never a tile, loop or output. -/
structure ZeroLaws (a : Arithmetic α) : Prop where
  literal : a.literal .real 0=a.zero
  half : kernelCast a .fp16 .real
    (kernelCast a .real .fp16 (a.literal .real 0))=a.zero
  rmsE5 : kernelCast a .fp16 .real (kernelCast a .f8e5 .fp16
    (a.e5Bits .f8e5 (Float8E5Bits.ofInt 0)))=a.zero
  weightE5 : kernelCast a .fp16 .real (kernelCast a .real .fp16
    (kernelCast a .f8e5 .real (a.e5Bits .f8e5 (Float8E5Bits.ofInt 0))))=a.zero

def valuesAt {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) : LoopValues α :=
  loopValues a fp8 p s (initialValues a p s) t

theorem valuesAt_keys {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (k : Fin 64) :
    (valuesAt a fp8 p s t).keys (k,PUnit.unit)=k.val := by
  induction t with
  | zero => rfl
  | succ t ih => exact ih

theorem valuesAt_xptrs {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (i : TileIndex [16,64]) :
    (valuesAt a fp8 p s t).xptrs i=
      (p.X,s.pids 0*p.sxb+((programRow p s*16+i.1.val)%p.M)*p.sxm+
        (t*64+i.2.1.val)*p.sxk) := by
  induction t with
  | zero => simp [valuesAt,loopValues,initialValues]
  | succ t ih =>
    change (((valuesAt a fp8 p s t).xptrs i).1,
      ((valuesAt a fp8 p s t).xptrs i).2+64*p.sxk)=_
    rw [ih]
    congr 1
    ring

theorem valuesAt_wptrs {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (i : TileIndex [64,64]) :
    (valuesAt a fp8 p s t).wptrs i=
      (p.W,(t*64+i.1.val)*p.swk+((programCol p s*64+i.2.1.val)%p.N)*p.swn) := by
  induction t with
  | zero => simp [valuesAt,loopValues,initialValues]
  | succ t ih =>
    change (((valuesAt a fp8 p s t).wptrs i).1,
      ((valuesAt a fp8 p s t).wptrs i).2+64*p.swk)=_
    rw [ih]
    congr 1
    ring

theorem valuesAt_rmsptrs {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (i : TileIndex [1,64]) :
    (valuesAt a fp8 p s t).rmsptrs i=(p.RMS,(t*64+i.2.1.val)*p.srms) := by
  induction t with
  | zero => simp [valuesAt,loopValues,initialValues]
  | succ t ih =>
    change (((valuesAt a fp8 p s t).rmsptrs i).1,
      ((valuesAt a fp8 p s t).rmsptrs i).2+64*p.srms)=_
    rw [ih]
    congr 1
    ring

#axiomsClean valuesAt_keys
#axiomsClean valuesAt_xptrs
#axiomsClean valuesAt_wptrs
#axiomsClean valuesAt_rmsptrs
#specNonCircular rawRow avoiding [valuesAt,loopValues,run,bodyExecution]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
