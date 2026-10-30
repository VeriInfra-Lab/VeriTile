import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Protocol

namespace VeriTile.Bench.Repairs.UniformSampling.Correct
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Independent real projection of the selected FP32 graph: scalar payload
meaning is supplied only at the two boundary arguments. Philox remains exact. -/
def model (denote : Float64Bits→ℝ) : Arithmetic ℝ where
  extract:=denote
  uniform32:=Spec.unit
  sub32:=fun a b=>a-b
  fma32:=fun a b c=>a*b+c
  neg32:=fun a=>-a

theorem scalar (denote : Float64Bits→ℝ) (scalars : Scalars) (seed start B i : Nat) :
    Native.result (model denote) scalars seed start B i=
      Spec.target (denote scalars.lower) (denote scalars.upper) seed start B i := by
  unfold Native.result affine model Spec.target
  dsimp
  ring

/-- Whole successful wrapper: independent Real target at every logical output,
exact returned object, exact selected-generator transition, and device restore.
All caller backing bytes outside output windows are covered separately on both
success and failure by `Host.all_outcomes_frame`. Empty is `Host.empty`.
No GPU/runtime implementation theorem is asserted. -/
theorem returned (api : API ℝ Store Byte) (denote : Float64Bits→ℝ) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (lib : Host.Runtime) (stops : Host.Stops)
    (s : Host.State Store) (out : Nat) (hn:0<n)
    (ok:((Host.entry api (model denote) scalars v scratch which lib stops).run s).result=.ok out) :
    ∃ r g,reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g) ∧
      out=v.handle ∧
      (∀ i,i<n→logical api (model denote)
        ((Host.entry api (model denote) scalars v scratch which lib stops).run s).state.memory v i=
          Spec.target (denote scalars.lower) (denote scalars.upper) r.seed r.start (block n) i) ∧
      ((Host.entry api (model denote) scalars v scratch which lib stops).run s).state.generators=
        Function.update s.generators which g ∧
      ((Host.entry api (model denote) scalars v scratch which lib stops).run s).state.device=s.device := by
  obtain ⟨r,g,hr,ho,hvalues,hgen,hdev⟩:=Host.returned api (model denote) scalars v scratch which lib stops s out hn
    (fun x=>neg_neg x) ok
  refine ⟨r,g,hr,ho,?_,hgen,hdev⟩
  intro i hi
  rw [hvalues i hi,scalar]

#axiomsClean returned
#guard_msgs (drop info) in
#specNonCircular Spec.target avoiding [Host.entry,Native.run,Execution.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Correct
