import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOProjection
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Linearization
noncomputable section
set_option maxHeartbeats 1500000
variable {α : Type} [Inhabited α]

/-- Select the finite public input windows from the same old typed memory.
No values are copied or converted by execution; this is logical indexing. -/
def values (h : Call) (data : Data α) : ChannelValues α (IO.inputs h data.indices) :=
  Fin.cases (fun j => data.indices j.1.val)
    (Fin.cases (fun j => ofFloat (dtype h.storage) (data.input (inputOffset h j.1.val j.2.1.val)))
      (Fin.cases (fun j => ofFloat (dtype h.storage) (data.weights (weightOffset h j.1.val j.2.1.val j.2.2.1.val)))
        (Fin.cases (fun j => ofFloat h.outputType (data.output (outputOffset h j.1.val j.2.1.val)))
          (fun i => Fin.elim0 i))))

def projected (h : Call) (data : Data α) : Data α :=
  IOValues.data h data.indices data.scaling (values h data)

theorem input (h : Call) (data : Data α) (b : Fin h.batches) (k : Fin h.K) :
    (projected h data).input (inputOffset h b.val k.val) = data.input (inputOffset h b.val k.val) := by
  change IOValues.input h data.indices (values h data) (inputOffset h b.val k.val) = _
  rw [IOValues.input_value]
  exact IOPrograms.to_from_float (dtype h.storage) _

theorem weight (h : Call) (data : Data α) (slot : Fin (IO.slots h data.indices)) (n : Fin h.N) (k : Fin h.K) :
    (projected h data).weights (weightOffset h slot.val n.val k.val) = data.weights (weightOffset h slot.val n.val k.val) := by
  change IOValues.weights h data.indices (values h data) (weightOffset h slot.val n.val k.val) = _
  rw [IOValues.weight_value]
  exact IOPrograms.to_from_float (dtype h.storage) _

theorem output (h : Call) (data : Data α) (b : Fin h.batches) (n : Fin h.N) :
    (projected h data).output (outputOffset h b.val n.val) = data.output (outputOffset h b.val n.val) := by
  change IOValues.output h data.indices (values h data) (outputOffset h b.val n.val) = _
  rw [IOValues.output_value]
  exact IOPrograms.to_from_float h.outputType _

theorem loaded (M : Algebra α) (h : Call) (data : Data α) (b : Fin h.batches) (split c : Nat) (e : Fin 256) :
    FPValues.loaded M h (projected h data) b.val split c e = FPValues.loaded M h data b.val split c e := by
  unfold FPValues.loaded
  split
  · rename_i inside
    exact input h data b ⟨rank split c e.val,inside⟩
  · rfl

theorem weight_loaded (M : Algebra α) (h : Call) (data : Data α) (b : Fin h.batches)
    (split c : Nat) (n : Fin h.BN) (e : Fin 256) :
    FPValues.weight M h (projected h data) (data.indices b.val).toNat split c n e =
      FPValues.weight M h data (data.indices b.val).toNat split c n e := by
  unfold FPValues.weight
  split
  · rename_i inside
    exact weight h data ⟨(data.indices b.val).toNat,IO.selected_inside h data.indices b⟩
      ⟨n.val,inside.1⟩ ⟨rank split c e.val,inside.2⟩
  · rfl

theorem products (M : Algebra α) (h : Call) (data : Data α) (b : Fin h.batches) (split c : Nat) :
    FPValues.products M h (projected h data) b.val (data.indices b.val).toNat split c =
      FPValues.products M h data b.val (data.indices b.val).toNat split c := by
  funext j
  obtain ⟨n,e,u⟩ := j
  simp only [FPValues.products,loaded,weight_loaded]

theorem sums (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) (b : Fin h.batches) (split c : Nat) :
    FPValues.sums M P h (projected h data) b.val (data.indices b.val).toNat split c =
      FPValues.sums M P h data b.val (data.indices b.val).toNat split c := by
  unfold FPValues.sums
  rw [products]

theorem accumulator (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) (b : Fin h.batches) (split c : Nat) :
    FPValues.accumulator M P h (projected h data) b.val (data.indices b.val).toNat split c =
      FPValues.accumulator M P h data b.val (data.indices b.val).toNat split c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    funext n
    simp only [FPValues.accumulator,ih,sums]

theorem payload (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) (b : Fin h.batches) (split : Nat) :
    FPValues.payload M P h (projected h data) b.val (data.indices b.val).toNat split =
      FPValues.payload M P h data b.val (data.indices b.val).toNat split := by
  funext n
  simp only [FPValues.payload,accumulator]
  rfl

theorem step (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) (b : Fin h.batches) (n : Fin h.N) :
    FPWorkset.step M P h (projected h data) b n = FPWorkset.step M P h data b n := by
  funext split old
  change P.atomic h.outputType old (FPValues.payload M P h (projected h data) b.val (data.indices b.val).toNat split.val _) = _
  rw [payload]
  rfl

/-- The original runtime's trace and its endpoint survive finite IO indexing
unchanged, including arbitrary original padding and exterior memory. -/
def run (M : Algebra α) (P : Primitives α) (h : Call) (orders : Orders h) (data : Data α)
    (original : FPWorkset.Run M P h orders data) : FPWorkset.Run M P h orders (projected h data) where
  output := original.output
  cells := by
    intro b n
    have trace := original.cells b n
    change Trace (FPWorkset.step M P h (projected h data) b n)
      (events h orders data.indices b n)
      ((projected h data).output (outputOffset h b.val n.val)) (original.output b n)
    rw [step,output]
    exact trace

#axiomsClean payload
#axiomsClean run
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOProjection
