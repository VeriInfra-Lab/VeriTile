import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOSafetyNativeBody

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem loop_keys (a : Arithmetic α) (native fp8 : Bool) (p : Arguments) (s : State α)
    (t : Nat) (k : Fin 64) :
    (IOBytes.loop native a fp8 p s (initialValues a p s) t).keys (k,PUnit.unit)=k.val := by
  cases native
  · exact valuesAt_keys a fp8 p s t k
  · exact nativeValuesAt_keys a fp8 p s t k

theorem loop_xptrs (a : Arithmetic α) (native fp8 : Bool) (p : Arguments) (s : State α)
    (t : Nat) (i : TileIndex [16,64]) :
    (IOBytes.loop native a fp8 p s (initialValues a p s) t).xptrs i=
      (p.X,s.pids 0*p.sxb+((programRow p s*16+i.1.val)%p.M)*p.sxm+(t*64+i.2.1.val)*p.sxk) := by
  cases native
  · exact valuesAt_xptrs a fp8 p s t i
  · exact nativeValuesAt_xptrs a fp8 p s t i

theorem loop_wptrs (a : Arithmetic α) (native fp8 : Bool) (p : Arguments) (s : State α)
    (t : Nat) (i : TileIndex [64,64]) :
    (IOBytes.loop native a fp8 p s (initialValues a p s) t).wptrs i=
      (p.W,(t*64+i.1.val)*p.swk+((programCol p s*64+i.2.1.val)%p.N)*p.swn) := by
  cases native
  · exact valuesAt_wptrs a fp8 p s t i
  · exact nativeValuesAt_wptrs a fp8 p s t i

theorem loop_rmsptrs (a : Arithmetic α) (native fp8 : Bool) (p : Arguments) (s : State α)
    (t : Nat) (i : TileIndex [1,64]) :
    (IOBytes.loop native a fp8 p s (initialValues a p s) t).rmsptrs i=(p.RMS,(t*64+i.2.1.val)*p.srms) := by
  cases native
  · exact valuesAt_rmsptrs a fp8 p s t i
  · exact nativeValuesAt_rmsptrs a fp8 p s t i

theorem loop_bounds (a : Arithmetic α) (native fp8 : Bool) (p : Arguments) (s : State α)
    (capacity : RegionName → Nat) (bound : Bounds p s capacity) (t : Nat) :
    LoopBounds p (IOBytes.loop native a fp8 p s (initialValues a p s) t) t capacity := by
  constructor
  · intro i active
    rw [loop_xptrs]
    have key:=loop_keys a native fp8 p s t i.2.1
    change _<p.K-t*64 at active
    rw [key] at active
    exact bound.x i.1 _ (by omega)
  · intro i active
    rw [loop_wptrs]
    have key:=loop_keys a native fp8 p s t i.1
    change _<p.K-t*64 at active
    rw [key] at active
    exact bound.w i.2.1 _ (by omega)
  · intro i active
    rw [loop_rmsptrs]
    have key:=loop_keys a native fp8 p s t i.2.1
    change _<p.K-t*64 at active
    rw [key] at active
    exact bound.rms _ (by omega)

theorem iteration_safe (a : Arithmetic α) (native fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : IOBytes.InputBytes fp8 p s)
    (regions : IOBytes.PointerRegions p v) (t : Nat) (capacity : RegionName → Nat)
    (bound : LoopBounds p v t capacity) :
    traceSafeList a.model capacity (IOBytes.code native fp8 outputFP8 p)
      (s.setReg "step" .nat [] (fun _=>t)) := by
  cases native
  · exact source_body a fp8 outputFP8 p s v h bytes regions t capacity bound
  · exact native_body a fp8 outputFP8 p s v h bytes regions t capacity bound

theorem loop_safe (a : Arithmetic α) (native fp8 outputFP8 : Bool)
    (p : Arguments) (initial : State α) (bytes : IOBytes.InputBytes fp8 p initial)
    (capacity : RegionName → Nat) (bound : Bounds p initial capacity)
    (count cur : Nat) (current : State α) (le : cur≤count) (same : current.mem=initial.mem)
    (h : LoopRegisters (IOBytes.loop native a fp8 p initial (initialValues a p initial) cur) current) :
    rangeSafe a.model capacity "step" cur count 1 (IOBytes.code native fp8 outputFP8 p) current := by
  have aux : ∀ left cur (current : State α),count-cur=left → cur≤count → current.mem=initial.mem →
      LoopRegisters (IOBytes.loop native a fp8 p initial (initialValues a p initial) cur) current →
      rangeSafe a.model capacity "step" cur count 1 (IOBytes.code native fp8 outputFP8 p) current := by
    intro left
    induction left using Nat.strong_induction_on with
    | h left ih=>
      intro cur current hc le hm hr
      rw [rangeSafe_unfold,if_neg (by decide)]
      by_cases less : cur<count
      · rw [if_pos less]
        have byteCurrent : IOBytes.InputBytes fp8 p current := by
          intro fp r member off
          simpa only [hm] using bytes fp r member off
        have regions:=IOBytes.loop_regions native a fp8 p initial (initialValues a p initial)
          (IOBytes.initial_regions a p initial) cur
        let next:=IOBytes.iteration a native fp8 outputFP8 p current _ hr byteCurrent regions cur
        refine ⟨iteration_safe a native fp8 outputFP8 p current _ hr byteCurrent regions cur capacity
          (loop_bounds a native fp8 p initial capacity bound cur),?_⟩
        rw [next.property]
        have nextMem : next.val.mem=initial.mem :=
          (IOBytes.iteration_frame a native fp8 outputFP8 p current _ hr byteCurrent regions cur).trans hm
        have nextRegs : LoopRegisters
            (IOBytes.loop native a fp8 p initial (initialValues a p initial) (cur+1)) next.val := by
          rw [IOBytes.loop_succ,←IOBytes.values_memory native a fp8 p initial current hm]
          exact IOBytes.iteration_values a native fp8 outputFP8 p current _ hr byteCurrent regions cur
        exact ih (count-(cur+1)) (by omega) (cur+1) next.val rfl (by omega) nextMem nextRegs
      · simp only [if_neg less]
  exact aux _ _ _ rfl le same h

#axiomsClean loop_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
