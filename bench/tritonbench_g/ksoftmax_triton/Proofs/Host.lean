import bench.tritonbench_g.ksoftmax_triton.Proofs.Ksoftmax_Contract
import bench.tritonbench_g.ksoftmax_triton.Proofs.GridValue

namespace VeriTile.Bench.Repairs.Ksoftmax.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- Ordinary next_power_of_2 metadata, independent of values and execution. -/
structure DepthLibrary where
  depth : Nat → Nat
  positive : ∀ K, 0<depth K
  contract : ∀ K, 0<K → Native.DepthContract K (depth K)
  primitive : ∀ K, K≤2^30 → depth K≤2^30

def decodeMode (mode : Option String) : Forward.MaskMode :=
  if mode=some "qk" then .qk else if mode=some "bk" then .bk else .absent

def body {α : Type} [Inhabited α] (M : Algebra α) (depth : DepthLibrary)
    (op : Operation) (x : TensorMetadata) (s : State α) : State α :=
  if x.numel=0 then s else
    runRows M op "dst" "left" "right" (batches x) (rows x) (columns x)
      (depth.depth (columns x)) (depth.positive _) (programs (batches x) (rows x)) s

def softmax {α : Type} [Inhabited α] (M : Algebra α) (depth : DepthLibrary)
    (y x : TensorMetadata) (mask : ArgumentMetadata) (log : Bool) (mode : Option String)
    (causal : Bool) (s : State α) : Except InputError (Unit × State α) := do
  checkForward y x mask mode Bool.true
  return ((),body M depth (.forward log causal (decodeMode mode)) x s)

def softmaxBackward {α : Type} [Inhabited α] (M : Algebra α) (depth : DepthLibrary)
    (dst grad saved : TensorMetadata) (log causal : Bool) (s : State α) : Except InputError (Unit × State α) := do
  checkBackward dst grad saved Bool.true
  return ((),body M depth (.backward log causal) saved s)

theorem forward_return {α : Type} [Inhabited α] (M : Algebra α) (depth : DepthLibrary)
    (y x : TensorMetadata) (mask : ArgumentMetadata) (log : Bool) (mode : Option String) (causal : Bool)
    (h : ForwardContract y x mask mode) (s : State α) :
    softmax M depth y x mask log mode causal s =
      .ok ((),body M depth (.forward log causal (decodeMode mode)) x s) := by
  simp [softmax,(checkForward_iff y x mask mode Bool.true).mpr ⟨rfl,h⟩]

theorem backward_return {α : Type} [Inhabited α] (M : Algebra α) (depth : DepthLibrary)
    (d g o : TensorMetadata) (log causal : Bool) (h : BackwardContract d g o) (s : State α) :
    softmaxBackward M depth d g o log causal s = .ok ((),body M depth (.backward log causal) o s) := by
  simp [softmaxBackward,(checkBackward_iff d g o Bool.true).mpr ⟨rfl,h⟩]

theorem body_value {α : Type} [Inhabited α] (M : Algebra α) (depth : DepthLibrary)
    (op : Operation) (x : TensorMetadata) (rank : x.shape.length=3) (s : State α)
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (hi : i.val<columns x) :
    (body M depth op x s).mem "dst" (Native.flat (rows x) (columns x) b r i.val)=
      gridValue M op (rows x) (columns x) b r
        (fun j => (s.mem "left" j).read .real) (fun j => (s.mem "right" j).read .real) i := by
  have bound := Native.active_bound b.isLt r.isLt hi
  have nonempty : x.numel≠0 := by rw [count_three x rank]; omega
  simp only [body,nonempty,if_false]
  exact grid_observe M op "dst" "left" "right" _ _ _ _ _ s b r i hi (by decide) (by decide)

theorem body_frame {α : Type} [Inhabited α] (M : Algebra α) (depth : DepthLibrary)
    (op : Operation) (x : TensorMetadata) (rank : x.shape.length=3) (s : State α) (region : RegionName) (j : Nat)
    (outside : region≠"dst" ∨ x.numel≤j) : (body M depth op x s).mem region j=s.mem region j := by
  unfold body
  split
  · rfl
  · apply rows_frame M op "dst" "left" "right" _ _ _ _ _ _ s region j
    simpa only [count_three x rank] using outside

/-- All valid columns have a lane; no coverage assumption occurs in wrappers. -/
theorem column_lane (depth : DepthLibrary) (K j : Nat) (hj : j<K) : j<depth.depth K :=
  lt_of_lt_of_le hj (depth.contract K (by omega)).covers

#axiomsClean forward_return
#axiomsClean backward_return
#axiomsClean body_value
#axiomsClean body_frame
#axiomsClean column_lane
end
end VeriTile.Bench.Repairs.Ksoftmax.Host
