import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section
open scoped Classical
set_option maxHeartbeats 5000000

def InputBinding (c : Call) (read : Nat→Word) : Prop :=
  (∀ k,logicalRead read c.inputs.q k=VeriTile.Torch.FP32.logicalValue read c.inputs.q c.inputs.qLaws k) ∧
  (∀ k,logicalRead read c.inputs.cosine k=VeriTile.Torch.FP32.logicalValue read c.inputs.cosine c.inputs.cosineLaws k) ∧
  (∀ k,logicalRead read c.inputs.sine k=VeriTile.Torch.FP32.logicalValue read c.inputs.sine c.inputs.sineLaws k)

theorem input_binding (c : Call) (admit : Admitted c) (read : Nat→Word) : InputBinding c read := by
  obtain ⟨q,cosine,sine⟩ := KernelBinding.admitted_payload c.inputs c.geometry admit.metadata
  exact ⟨logical_word read _ c.inputs.qLaws q,logical_word read _ c.inputs.cosineLaws cosine,
    logical_word read _ c.inputs.sineLaws sine⟩

/-- Independent pair graph over the actual Tensor logical coordinates. -/
def tensorScalar (op : Operations α) (embed : β→α) (c : Call) (before : Nat→β)
    (row head col : Nat) : α :=
  scalarResult op c.geometry c.backward
    (fun k=>embed (logicalRead before c.inputs.q k))
    (fun k=>embed (logicalRead before c.inputs.cosine k))
    (fun k=>embed (logicalRead before c.inputs.sine k)) row head col

theorem scalar_binding {α β : Type} [Inhabited α] (op : Operations α) (embed : β→α)
    (c : Call) (admit : Admitted c) (before : Nat→β) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads) (hc : col<c.geometry.dimension) :
    scalarResult op c.geometry c.backward
      (fun o=>((state embed before c.inputs).mem "Q" o).read .real)
      (fun o=>((state embed before c.inputs).mem "C" o).read .real)
      (fun o=>((state embed before c.inputs).mem "S" o).read .real) row head col=
      tensorScalar op embed c before row head col := by
  have payload := KernelBinding.admitted_payload c.inputs c.geometry admit.metadata
  obtain ⟨_,cs,ss,cl,sl⟩ := KernelBinding.admitted_shapes c.inputs c.geometry admit.metadata
  have qread (k : Nat) (hk : k<c.geometry.dimension) :
      ((state embed before c.inputs).mem "Q" (c.geometry.offset row head k)).read .real=
        embed (logicalRead before c.inputs.q (c.geometry.offset row head k)) := by
    have bound := offset_lt c.geometry row head k hr hh hk
    rw [←q_size c admit] at bound
    simp only [state,Cell.read_same,↓reduceIte,logicalRead,tensor_address _ c.inputs.qLaws payload.1 _ bound]
  have coeff (j : Nat) (hj : j<c.geometry.half) :
      ((state embed before c.inputs).mem "C" (row%c.geometry.sequence*c.geometry.half+j)).read .real=
        embed (logicalRead before c.inputs.cosine (row%c.geometry.sequence*c.geometry.half+j)) ∧
      ((state embed before c.inputs).mem "S" (row%c.geometry.sequence*c.geometry.half+j)).read .real=
        embed (logicalRead before c.inputs.sine (row%c.geometry.sequence*c.geometry.half+j)) := by
    have bound := KernelBinding.coefficient_bound c.geometry row j hr hj
    have cb : row%c.geometry.sequence*c.geometry.half+j<c.inputs.cosine.metadata.numel := by
      rw [KernelBinding.table_numel _ _ _ cs]
      exact lt_of_lt_of_le bound (Nat.mul_le_mul_right _ cl)
    have sb : row%c.geometry.sequence*c.geometry.half+j<c.inputs.sine.metadata.numel := by
      rw [KernelBinding.table_numel _ _ _ ss]
      exact lt_of_lt_of_le bound (Nat.mul_le_mul_right _ sl)
    constructor
    · simp [state,logicalRead,tensor_address _ c.inputs.cosineLaws payload.2.1 _ cb]
    · simp [state,logicalRead,tensor_address _ c.inputs.sineLaws payload.2.2 _ sb]
  unfold tensorScalar scalarResult
  split
  · rename_i active
    let lane := if col<c.geometry.half then col else col-c.geometry.half
    have hl : lane<c.geometry.half := by dsimp [lane]; split <;> omega
    have dim : 2*c.geometry.half≤c.geometry.dimension := by unfold Request.half; omega
    have q1 := qread lane (by omega)
    have q2 := qread (lane+c.geometry.half) (by omega)
    obtain ⟨cosine,sine⟩ := coeff lane hl
    dsimp only
    rw [q1,q2,cosine,sine]
  · exact qread col hc

/-- Complete source execution and all physical/logical outputs in one result.
Only scalar embedding/projection identity is needed for the untouched frame. -/
theorem tensor_execution {α β : Type} [Inhabited α] (op : Operations α)
    (embed : β→α) (project : α→β) (retract : ∀ v,project (embed v)=v)
    (c : Call) (admit : Admitted c) (before : Nat→β) (row group : Nat)
    (launch : c.geometry.Launched row group) :
    ∃ final,FP.Structural.exec op.backend c.source
        (programState (state embed before c.inputs) row group)=some final ∧
      Kernel.Post c.geometry row group (state embed before c.inputs) final
        (fun head col=>tensorScalar op embed c before row head col) ∧
      (∀ peer head col,peer<c.geometry.rows → head<c.geometry.heads → col<c.geometry.dimension →
        logicalRead (physical project c.inputs before final) c.inputs.q (c.geometry.offset peer head col)=
          if affected c row group peer head col then project (tensorScalar op embed c before row head col)
          else logicalRead before c.inputs.q (c.geometry.offset peer head col)) ∧
      (∀ address,¬WrittenAddress c row group address → physical project c.inputs before final address=before address) := by
  classical
  obtain ⟨final,ran,post⟩ := executes op c admit (state embed before c.inputs) row group launch
  have logical : Kernel.Post c.geometry row group (state embed before c.inputs) final
      (fun head col=>tensorScalar op embed c before row head col) := by
    refine ⟨?_,post.frame⟩
    intro head col hlo hhi hc
    have dim : 2*c.geometry.half≤c.geometry.dimension := by unfold Request.half; omega
    rw [post.values head col hlo hhi hc,scalar_binding op embed c admit before row head col
      launch.2.1 (lt_of_lt_of_le hhi (min_le_right _ _)) (by omega)]
  exact ⟨final,ran,logical,physical_tensor embed project retract c admit row group before final _ logical,
    physical_frame embed project retract c row group before final _ logical⟩

/-- Independent exact-Real formula; storage/compute casts have their ideal
interpretation here only. Backward uses the current coefficient input words. -/
def realValue (c : Call) (before : Nat→ℝ) (row head col : Nat) : ℝ :=
  Spec.result c.geometry c.backward (logicalRead before c.inputs.q)
    (logicalRead before c.inputs.cosine) (logicalRead before c.inputs.sine) row head col

/-- Independent native scalar graph: same-FP32 casts, coefficient-first mul,
real ternary FMA, and source backward sub(+0,sine) are all retained. -/
def nativeValue (p : Numerics.Primitives) (c : Call) (before : Nat→Word) (row head col : Nat) : Word :=
  Numerics.result p c.geometry c.backward (logicalRead before c.inputs.q)
    (logicalRead before c.inputs.cosine) (logicalRead before c.inputs.sine) row head col

theorem real_binding (c : Call) (before : Nat→ℝ) (row head col : Nat) :
    tensorScalar ideal id c before row head col=realValue c before row head col :=
  ideal_result _ _ _ _ _ _ _ _

theorem native_binding (p : Numerics.Primitives) (c : Call) (before : Nat→Word) (row head col : Nat) :
    (tensorScalar (Numerics.operations p) Numerics.leaf c before row head col).word=
      nativeValue p c before row head col :=
  Numerics.source_profile _ _ _ _ _ _ _ _ _

#specNonCircular realValue avoiding [Call.source,FP.Structural.exec]
#specNonCircular nativeValue avoiding [Call.source,FP.Structural.exec]
#axiomsClean tensor_execution
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
