import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.PairIO
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOGeometry

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open IOData
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

structure JobEq (left right : Invocation α) : Prop where
  run : ∀ s,left.run s=right.run s
  safe : ∀ cap s,left.Safe cap s ↔ right.Safe cap s

theorem JobEq.refl (j : Invocation α) : JobEq j j := ⟨fun _=>rfl,fun _ _=>Iff.rfl⟩

theorem forward_job (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 128) : JobEq (IOData.forwardJob M D shape scale p) (forwardJob M D shape scale p) :=
  ⟨fun _=>forward_run (withDot M D) _ _,fun _ _=>forward_safe (withDot M D) _ _ _⟩

theorem query_job (M : Algebra α) (D : DotPrimitive α) (shape : Shape) (scale : ℝ)
    (p : Program shape 64) : JobEq (IOData.queryJob M D shape scale p) (queryJob M D shape scale p) :=
  ⟨fun _=>query_run M D _ _,fun _ _=>query_safe M D _ _ _⟩

theorem related (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :
    List.Forall₂ JobEq (nativeJobs direction d scale) (jobs direction d scale) := by
  cases direction with
  | forward =>
    change List.Forall₂ JobEq (List.map _ (programs d.shape 128)) (List.map _ (programs d.shape 128))
    generalize programs d.shape 128=ps
    induction ps with
    | nil=>exact .nil
    | cons p rest ih=>exact .cons (forward_job _ _ _ _ _) ih
  | backward =>
    change List.Forall₂ JobEq (List.flatMap _ (programs d.shape 64)) (List.flatMap _ (programs d.shape 64))
    generalize programs d.shape 64=ps
    induction ps with
    | nil=>exact .nil
    | cons p rest ih=>
      simp only [List.flatMap_cons,List.cons_append,List.nil_append]
      exact .cons (query_job _ _ _ _ _) (.cons (JobEq.refl _) ih)

theorem schedule_run {left right : List (Invocation α)} (h : List.Forall₂ JobEq left right)
    (s : State α) : runSchedule left s=runSchedule right s := by
  induction h generalizing s with
  | nil=>rfl
  | cons h _ ih=>
    simp only [runSchedule,h.run]
    cases _ : Invocation.run _ s <;> simp [Option.bind,ih]

theorem schedule_safe {left right : List (Invocation α)} (h : List.Forall₂ JobEq left right)
    (cap : RegionName → Nat) (s : State α) (hs : ScheduleSafe cap left s) :
    ScheduleSafe cap right s := by
  induction h generalizing s with
  | nil=>trivial
  | cons h _ ih=>
    refine ⟨(h.safe cap s).mp hs.1,?_⟩
    intro next ran
    exact ih next (hs.2 next ((h.run s).trans ran))

theorem success_frame (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ)
    (xs : ChannelValues NativeStage.Expr (inputPorts d.shape)) (s : State NativeStage.Expr) :
    ∃ final,runSchedule (nativeJobs direction d scale) s=some final ∧
      (io direction d scale).Frame xs s final := by
  cases direction with
  | forward =>
    obtain ⟨final,ran,_,_,frame⟩:=IOGraph.forward_grid (NativeStage.algebra .forward d)
      (NativeStage.dots .forward d.shape) d.shape scale (programs d.shape 128) s
    refine ⟨final,(IOGraph.forward_schedule _ _ _ _ _ s).trans ran,?_⟩
    intro R off outside
    exact frame R off (fun p _=>forward_outside d.shape _ xs R off outside p)
  | backward =>
    obtain ⟨final,ran,_,_,frame⟩:=IOGraph.backward_grid (NativeStage.algebra .query d)
      (NativeStage.dots .query d.shape) (NativeStage.algebra .keyValue d)
      (NativeStage.dots .keyValue d.shape) d.shape scale (programs d.shape 64) s
    refine ⟨final,(IOGraph.backward_schedule _ _ _ _ _ _ _ s).trans ran,?_⟩
    intro R off outside
    exact frame R off (fun p _=>backward_outside d.shape _ xs R off outside p)

theorem equivalent (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :
    (io direction d scale).Equivalent (variant direction d scale) := by
  intro xs _ cap hb initial _
  have hs:=native_safe direction d scale xs cap hb initial
  have hr:=related direction d scale
  obtain ⟨final,ran,frame⟩:=success_frame direction d scale xs initial
  refine ⟨hs,schedule_safe hr cap initial hs,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run hr initial).symm.trans ran
  · intro o j active
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
