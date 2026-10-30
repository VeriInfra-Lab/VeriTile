import bench.tritonbench_g.triton_attention.Proofs.ForwardLocality
import bench.tritonbench_g.triton_attention.Proofs.PreprocessLocality
import bench.tritonbench_g.triton_attention.Proofs.BackwardLocality
import bench.tritonbench_g.triton_attention.Proofs.NativeGraphMemory

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
open VeriTile Triton FP.Structural Source
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000

theorem fold (limit : Nat) (step : Grid.Program→Memory α→Memory α)
    (work : List Grid.Program)
    (preserves : ∀ pid∈work,∀ a b,Agree limit a b → Agree limit (step pid a) (step pid b))
    (a b : Memory α) (h : Agree limit a b) :
    Agree limit (work.foldl (fun m pid=>step pid m) a) (work.foldl (fun m pid=>step pid m) b) := by
  induction work generalizing a b with
  | nil=>exact h
  | cons pid work ih=>
    apply ih (fun id member=>preserves id (List.mem_cons_of_mem pid member))
    exact preserves pid (List.mem_cons_self ..) a b h

theorem whole (F P B : Primitives α) (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : ∀ pid∈fw,Grid.Legal (Grid.forwardGrid c) pid)
    (hp : ∀ pid∈pw,Grid.Legal (Grid.preprocessGrid c) pid)
    (hb : ∀ pid∈bw,Grid.Legal (Grid.backwardGrid c) pid)
    (a b : Memory α) (h : Agree c.elements a b) :
    Agree c.elements (Grid.wholeReference F P B c scale fw pw bw a)
      (Grid.wholeReference F P B c scale fw pw bw b) := by
  unfold Grid.wholeReference
  apply fold c.elements (Grid.backwardStep B c scale) bw
    (fun pid member a b h=>backward_step B c scale pid (hb pid member) a b h)
  apply fold c.elements (Grid.preprocessStep P c) pw
    (fun pid member a b h=>preprocess_step P c pid (hp pid member) a b h)
  exact fold c.elements (Grid.forwardStep F c scale) fw
    (fun pid member a b h=>forward_step F c scale pid (hf pid member) a b h) a b h

theorem native_store (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program)
    (legal : Grid.Legal (Native.gridFor c kind) pid) (a b : Memory Native.Graph)
    (h : Agree c.elements a b) :
    Agree c.elements (Native.Selected.store one kind c scale owners pid a)
      (Native.Selected.store one kind c scale owners pid b) := by
  have embedded : Agree c.elements (Native.Selected.embed a) (Native.Selected.embed b) :=
    mapped (fun g _=>Native.Stage.ready g) c.elements a b h
  intro r o ho
  unfold Native.Selected.store Native.Selected.finish
  apply congrArg (Native.mapCell (fun e : Native.Expr=>Native.lowerScalar one kind (e (Native.storeOwner c owners r o))))
  cases kind with
  | forward=>exact forward_step (Native.modelScalar one .forward) c scale pid legal _ _ embedded r o ho
  | preprocess=>exact preprocess_step (Native.modelScalar one .preprocess) c pid legal _ _ embedded r o ho
  | backward=>exact backward_step (Native.modelScalar one .backward) c scale pid legal _ _ embedded r o ho

theorem native_whole (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (fw pw bw : List Grid.Program)
    (hf : ∀ pid∈fw,Grid.Legal (Grid.forwardGrid c) pid)
    (hp : ∀ pid∈pw,Grid.Legal (Grid.preprocessGrid c) pid)
    (hb : ∀ pid∈bw,Grid.Legal (Grid.backwardGrid c) pid)
    (a b : Memory Native.Graph) (h : Agree c.elements a b) :
    Agree c.elements (Native.Selected.stores one c scale owners fw pw bw a)
      (Native.Selected.stores one c scale owners fw pw bw b) := by
  unfold Native.Selected.stores
  apply fold c.elements (Native.Selected.store one .backward c scale owners) bw
    (fun pid member a b h=>native_store one .backward c scale owners pid (hb pid member) a b h)
  apply fold c.elements (Native.Selected.store one .preprocess c scale owners) pw
    (fun pid member a b h=>native_store one .preprocess c scale owners pid (hp pid member) a b h)
  exact fold c.elements (Native.Selected.store one .forward c scale owners) fw
    (fun pid member a b h=>native_store one .forward c scale owners pid (hf pid member) a b h) a b h

#axiomsClean whole
#axiomsClean native_whole
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
