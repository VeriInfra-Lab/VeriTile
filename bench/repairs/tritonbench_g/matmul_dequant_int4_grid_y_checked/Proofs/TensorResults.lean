import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.NativeResults

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.TensorBridge
open VeriTile.Triton FP.Structural Tensor Native
open MatmulDequantInt4.Direct
noncomputable section

/-- Public complete Real kernel theorem for either admitted storage profile,
all eight tunings, actual Tensor coordinates, and the complete storage frame. -/
theorem real (f : Source.Profile) (p : PackedInfo) (o : Info) (t : Arguments p o)
    (checked : checkPacked p 1 f.hostDtype=.ok ()) (output : OutputABI p f o)
    (i : Fin 8) (heap : Heap ℝ) :
    ∃ final,KernelGrid.run f Spec.real (params p i) (entry f p o t heap)=some final ∧
      Post p o t (Spec.mathematical p o t heap) heap (physical f p o t heap final) := by
  obtain ⟨state,executed,values,frame,packed⟩ := Tensor.whole f Spec.real p o t
    ((checkPacked_iff _ _ _).mp checked) output i heap
  refine ⟨state,executed,?_,frame,packed⟩
  intro k n hk hn
  have hv := values k n hk hn
  simpa only [dif_pos hk,dif_pos hn,Spec.result_eq,Spec.real_result] using hv

/-- Independent ordered native target on all packed words and every scale
payload, including NaNs, infinities, signed zeros and subnormals. -/
def target (f : Source.Profile) (I : Instructions) (p : PackedInfo) (o : Info)
    (t : Arguments p o) (heap : Heap Word) (k n : Nat) : Word :=
  match f with
  | .half => (I.mul16 (I.convert16 .signed (Spec.difference p o t heap k n))
      ((Spec.scale p o t heap k n).truncate 16)).zeroExtend 32
  | .single => I.mul32 (I.convert32 .signed (Spec.difference p o t heap k n))
      (Spec.scale p o t heap k n)

theorem target_eq (f : Source.Profile) (I : Instructions) (conversion : f=.half → Conversion I)
    (p : PackedInfo) (o : Info) (t : Arguments p o) (heap : Heap Word) (k n : Nat) :
    Spec.scalarResult f (scalar I) p o t heap k n=target f I p o t heap k n := by
  have bounds := Spec.difference_bounds p o t heap k n
  cases f with
  | half => simp [Spec.scalarResult,scalar,target,Source.Profile.dtype,
      conversion rfl _ bounds.1 bounds.2]
  | single => rfl

/-- Complete source launch and actual Tensor/frame theorem. Numerical
assumptions are confined to the named finite scalar conversion port. -/
theorem native (f : Source.Profile) (I : Instructions) (conversion : f=.half → Conversion I)
    (p : PackedInfo) (o : Info) (t : Arguments p o)
    (checked : checkPacked p 1 f.hostDtype=.ok ()) (output : OutputABI p f o)
    (i : Fin 8) (heap : Heap Word) :
    ∃ final,KernelGrid.run f (scalar I) (params p i) (entry f p o t heap)=some final ∧
      Post p o t (target f I p o t heap) heap (physical f p o t heap final) := by
  obtain ⟨state,executed,values,frame,packed⟩ := Tensor.whole f (scalar I) p o t
    ((checkPacked_iff _ _ _).mp checked) output i heap
  refine ⟨state,executed,?_,frame,packed⟩
  intro k n hk hn
  have hv := values k n hk hn
  simpa only [dif_pos hk,dif_pos hn,Spec.result_eq,target_eq f I conversion] using hv

#axiomsClean real
#axiomsClean native
#specNonCircular target avoiding [KernelGrid.run,Source.source,Source.execution,Native.scalar,Spec.scalarResult]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.TensorBridge
