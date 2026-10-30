import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.RawInput
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeLoop

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def nativeValuesAt {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) : LoopValues α :=
  nativeLoopValues a fp8 p s (initialValues a p s) t

theorem nativeValuesAt_keys {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (k : Fin 64) :
    (nativeValuesAt a fp8 p s t).keys (k,PUnit.unit)=k.val := by
  induction t with
  | zero => rfl
  | succ t ih => exact ih

theorem nativeValuesAt_xptrs {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (i : TileIndex [16,64]) :
    (nativeValuesAt a fp8 p s t).xptrs i=
      (p.X,s.pids 0*p.sxb+((programRow p s*16+i.1.val)%p.M)*p.sxm+
        (t*64+i.2.1.val)*p.sxk) := by
  induction t with
  | zero => simp [nativeValuesAt,nativeLoopValues,initialValues]
  | succ t ih =>
    change (((nativeValuesAt a fp8 p s t).xptrs i).1,
      ((nativeValuesAt a fp8 p s t).xptrs i).2+64*p.sxk)=_
    rw [ih]
    congr 1
    ring

theorem nativeValuesAt_wptrs {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (i : TileIndex [64,64]) :
    (nativeValuesAt a fp8 p s t).wptrs i=
      (p.W,(t*64+i.1.val)*p.swk+((programCol p s*64+i.2.1.val)%p.N)*p.swn) := by
  induction t with
  | zero => simp [nativeValuesAt,nativeLoopValues,initialValues]
  | succ t ih =>
    change (((nativeValuesAt a fp8 p s t).wptrs i).1,
      ((nativeValuesAt a fp8 p s t).wptrs i).2+64*p.swk)=_
    rw [ih]
    congr 1
    ring

theorem nativeValuesAt_rmsptrs {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s : State α) (t : Nat) (i : TileIndex [1,64]) :
    (nativeValuesAt a fp8 p s t).rmsptrs i=(p.RMS,(t*64+i.2.1.val)*p.srms) := by
  induction t with
  | zero => simp [nativeValuesAt,nativeLoopValues,initialValues]
  | succ t ih =>
    change (((nativeValuesAt a fp8 p s t).rmsptrs i).1,
      ((nativeValuesAt a fp8 p s t).rmsptrs i).2+64*p.srms)=_
    rw [ih]
    congr 1
    ring

#axiomsClean nativeValuesAt_keys
#axiomsClean nativeValuesAt_xptrs
#axiomsClean nativeValuesAt_wptrs
#axiomsClean nativeValuesAt_rmsptrs
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
