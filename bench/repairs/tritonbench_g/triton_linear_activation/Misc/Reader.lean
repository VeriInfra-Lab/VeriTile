import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Machine
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.SourceLink
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.ConditionalCorrect
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.ConditionalFPEquiv
namespace VeriTile.Bench.Repairs.TritonLinearActivation.Reader
noncomputable section
set_option maxHeartbeats 4000000

def request (batched bias saved : Bool) (mode : Mode) : Request :=
  ⟨if batched then .batched 2 3 else .matrix 17,33,35,0,Fin.val,Fin.val,bias,saved,mode,some "fast_gelu"⟩

theorem admitted (batched bias saved : Bool) (mode : Mode) : (request batched bias saved mode).Valid := by
  constructor
  · cases batched
    · change SourceBounds ⟨17,33,35⟩
      norm_num [SourceBounds]
    · change SourceBounds ⟨6,33,35⟩
      norm_num [SourceBounds]
  · intro i j ne hi hj equal
    exact ne (Fin.ext equal)

def allocation (r : Request) (before : Memory α) (storage : r.storage=Fin.val)
    (object : r.object=Fin.val) : Allocation r {n | n<4} before where
  storage := fun i => 10+i.val
  object := fun i => 20+i.val
  distinct := by intro i j eq; apply Fin.ext; change 10+i.val=10+j.val at eq; omega
  objectDistinct := by intro i j eq; apply Fin.ext; change 20+i.val=20+j.val at eq; omega
  fresh := by intro i; change ¬10+i.val<4; omega
  objectFresh := by intro i j; rw [object]; have := j.isLt; omega
  inputPresent := by intro i; rw [storage]; exact i.isLt
  initial := before
  frame := by intro p hp; rfl

theorem all_configs_modes_real (batched bias saved : Bool) (mode : Mode) (c : Fin 98) (before : Memory ℝ)
    (trials : List (Fin 98)) :
    RealPost (request batched bias saved mode) (allocation _ before rfl rfl)
      (execute (allocation (request batched bias saved mode) before rfl rfl) realArithmetic c 7 trials) 7 :=
  correct _ c (admitted batched bias saved mode) 7 trials

theorem all_configs_modes_fp (batched bias saved : Bool) (mode : Mode) (c : Fin 98)
    (before : Memory Word) (p : LocalProfile) (trials : List (Fin 98)) :
    FPPost (c:=c) p (allocation (request batched bias saved mode) before rfl rfl)
      (execute (allocation (request batched bias saved mode) before rfl rfl) (fpArithmetic p) c 7 trials) 7 :=
  reference_fp_post _ _ c (admitted batched bias saved mode) 7 trials

#specNonCircular densePre avoiding [run, sourceAccumulation, accumulation, execute]
#specNonCircular denseOutput avoiding [run, sourceAccumulation, accumulation, execute, activation]
#auditStmt correct
#auditStmt fp_equiv
#print_spec correct
#print_spec reference_fp_post
#print_spec fp_equiv
#axiomsClean admitted
#axiomsClean all_configs_modes_real
#axiomsClean all_configs_modes_fp
#axiomsClean correct
#axiomsClean fp_equiv
#axiomsClean fp_output_bytes
#axiomsClean fp_saved_bytes
#axiomsClean accumulation_real
#axiomsClean grouped_covered
#axiomsClean grouped_injective
#axiomsClean source_store_inverse
#axiomsClean source_store_unique
#axiomsClean active_source_accumulation
#axiomsClean run_frozen
#axiomsClean run_output
#axiomsClean run_saved
#axiomsClean run_frame
#axiomsClean run_empty
#axiomsClean source_scalar_bounds
#axiomsClean output_machine_exact
#axiomsClean input_machine_exact
#axiomsClean loaded_input_bound
#axiomsClean Code.native_refines
#axiomsClean native_grid
#axiomsClean CopyRegions.output
#axiomsClean CopyRegions.frame
#axiomsClean Allocation.prepared_input
#axiomsClean Allocation.separate
#axiomsClean Allocation.preactivation
#axiomsClean Allocation.final_output
#axiomsClean Allocation.final_saved
#axiomsClean Allocation.final_frame
#axiomsClean Allocation.host_metadata
#axiomsClean Allocation.context_amp
#axiomsClean Allocation.native_host
#axiomsClean native_fp_host_memory
#axiomsClean densePre_zero_k
#axiomsClean no_launch
#axiomsClean run_absorbs
#axiomsClean trial_absorbed
#axiomsClean Allocation.native_autotuned
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Reader
