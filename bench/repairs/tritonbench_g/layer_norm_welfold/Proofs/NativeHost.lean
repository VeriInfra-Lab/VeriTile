import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceGraph
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.LoweringPlan
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Native
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
local instance (p : Prop) : Decidable p := Classical.propDecidable p

/-- Every observable store of the complete selected launch, including S=0. -/
inductive Port (S D : Nat) where
  | mean : Fin S→Port S D
  | inverse : Fin S→Port S D
  | output : Fin S→Fin D→Port S D

def Port.row : Port S D→Nat
  | .mean row | .inverse row | .output row _ => row.val

def Port.dtype : Port S D→TileDType
  | .mean _ | .inverse _ => .real
  | .output _ _ => .bf16

def Port.cell {α : Type} (port : Port S D) (value : α) : Cell α :=
  match port with
  | .mean _ | .inverse _ => .mk .real value
  | .output _ _ => .mk .bf16 value

def Port.address (a : Allocations) : Port S D→RegionName×Nat
  | .mean row => (a.mean,row.val)
  | .inverse row => (a.inverse,row.val)
  | .output row col => (a.output,col.val+D*row.val)

theorem Port.address_injective (a : Allocations) (live : RegionName→Prop) (fresh : a.Fresh live) :
    Function.Injective (Port.address a : Port S D→RegionName×Nat) := by
  intro x y h
  cases x with
  | mean r =>
    cases y with
    | mean q => exact congrArg Port.mean (Fin.ext (Prod.mk.inj h).2)
    | inverse q => exact False.elim (fresh.mean_inverse (Prod.mk.inj h).1)
    | output q c => exact False.elim (fresh.mean_output (Prod.mk.inj h).1)
  | inverse r =>
    cases y with
    | mean q => exact False.elim (fresh.mean_inverse (Prod.mk.inj h).1.symm)
    | inverse q => exact congrArg Port.inverse (Fin.ext (Prod.mk.inj h).2)
    | output q c => exact False.elim (fresh.inverse_output (Prod.mk.inj h).1)
  | output r c =>
    cases y with
    | mean q => exact False.elim (fresh.mean_output (Prod.mk.inj h).1.symm)
    | inverse q => exact False.elim (fresh.inverse_output (Prod.mk.inj h).1.symm)
    | output q e =>
      obtain ⟨hr,hc⟩ := row_column_injective D r.val q.val c.val e.val c.isLt e.isLt (Prod.mk.inj h).2
      cases Fin.ext hr
      cases Fin.ext hc
      rfl

def Port.source (c : Configuration) : Port S D→Expr
  | .mean _ => meanExpr D c.block c.tree
  | .inverse _ => inverseExpr D c.block c.tree
  | .output _ col => outputExpr D c.block col.val c.tree

/-- Compiler/ABI input is a value-erased graph with an exact source recipe and
local construction trace for each actual store port. It contains no values,
semantic equality field, input-dependent output oracle, or reduction law.
The source-to-PTX graph binding remains an ordinary compiler boundary, audited
for the saved profiles; the theorem is parametric in every such local plan. -/
structure CompiledCall (S D : Nat) (c : Configuration) where
  code : Port S D→Expr
  plan : ∀ port, LoweringPlan (port.source c) (code port)

/-- Parallel unique-address store semantics of the graph launch. Private source
registers are irrelevant to the returned tensor observations. -/
def executeNative {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (c : Configuration) (program : CompiledCall S D c) : Returned α :=
  let memory : RegionName→Nat→Cell α := fun q off =>
    if hit : ∃ port : Port S D, port.address a=(q,off) then
      let port := Classical.choose hit
      port.cell (eval p (inputs M s X.region W.region Bias.region D port.row) (program.code port))
    else s.mem q off
  ⟨{s with mem := memory},(outputTensor a S D,X,meanTensor a S,inverseTensor a S)⟩

/-- A pure structural readback for each graph output: fresh allocations give
exactly one port per store address. No scalar numerical hypothesis is used. -/
theorem executeNative_port {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (c : Configuration) (program : CompiledCall S D c) (live : RegionName→Prop)
    (fresh : a.Fresh live) (port : Port S D) :
    (executeNative M p s X W Bias a S D c program).state.mem (port.address a).1 (port.address a).2 =
      port.cell (eval p (inputs M s X.region W.region Bias.region D port.row) (program.code port)) := by
  have hit : ∃ other : Port S D,other.address a=((port.address a).1,(port.address a).2) := ⟨port,rfl⟩
  dsimp only [executeNative]
  rw [dif_pos hit]
  have he : Classical.choose hit=port := Port.address_injective a live fresh (Classical.choose_spec hit)
  simp only [he]

theorem source_port {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (bindings : SourceBindings M p) (s t : State α) (X W Bias : Tensor)
    (a : Allocations) (S D : Nat) (c : Configuration)
    (effect : GridEffect M s t X.region W.region Bias.region a.output a.mean a.inverse D c.block c.tree (List.range S))
    (port : Port S D) :
    t.mem (port.address a).1 (port.address a).2 = port.cell
      (eval p (inputs M s X.region W.region Bias.region D port.row) (port.source c)) := by
  cases port with
  | mean row => simpa only [Port.address,Port.cell,Port.row,Port.source,mean_eval M p bindings] using effect.mean row.val (List.mem_range.mpr row.isLt)
  | inverse row => simpa only [Port.address,Port.cell,Port.row,Port.source,inverse_eval M p bindings] using effect.rstd row.val (List.mem_range.mpr row.isLt)
  | output row col => simpa only [Port.address,Port.cell,Port.row,Port.source,output_eval M p bindings] using effect.output row.val (List.mem_range.mpr row.isLt) col.val col.isLt

theorem no_port_frame (a : Allocations) (S D : Nat) (q : RegionName) (off : Nat)
    (noHit : ¬∃ port : Port S D,port.address a=(q,off)) : gridMiss a.output a.mean a.inverse S D q off := by
  intro row hr
  refine ⟨?_,?_,?_⟩
  · by_cases hq : q=a.mean
    · right; intro ho
      exact noHit ⟨.mean ⟨row,hr⟩,by simp [Port.address,hq,ho]⟩
    · exact .inl hq
  · by_cases hq : q=a.inverse
    · right; intro ho
      exact noHit ⟨.inverse ⟨row,hr⟩,by simp [Port.address,hq,ho]⟩
    · exact .inl hq
  · by_cases hq : q=a.output
    · right; intro col hc ho
      exact noHit ⟨.output ⟨row,hr⟩ ⟨col,hc⟩,by simp [Port.address,hq,ho]⟩
    · exact .inl hq

/-- Complete source execution is joined to every native mean/rstd/Y store,
all tuple identities and the full caller-memory frame. The only numeric ports
are SourceBindings' scalar primitive equalities and LocalLaws at plan leaves. -/
theorem host_native_complete {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α→α→Bool)
    (p : Primitives α) (bindings : SourceBindings M p) (s : State α) (X W Bias : Tensor)
    (a : Allocations) (S D : Nat) (domain : InputContract s X W Bias S D)
    (live : RegionName→Prop) (hX : live X.region) (hW : live W.region) (hBias : live Bias.region)
    (fresh : a.Fresh live) (trials : List Configuration) (last : Configuration)
    (program : CompiledCall S D last)
    (laws : ∀ port, LocalLaws p (inputs M s X.region W.region Bias.region D port.row)
      (fun step => step∈(program.plan port).steps)) :
    ∃ ret, executeHost M eq32 s X W Bias a S D trials last=some ret ∧
      ret.tensors=(executeNative M p s X W Bias a S D last program).tensors ∧
      ret.state.mem=(executeNative M p s X W Bias a S D last program).state.mem ∧
      ReturnedCorrect M eq32 s X W Bias a S D last (executeNative M p s X W Bias a S D last program) ∧
      (∀ q, live q→∀ off,(executeNative M p s X W Bias a S D last program).state.mem q off=s.mem q off) := by
  obtain ⟨ret,hr,ht,hf⟩ := host_complete M eq32 s X W Bias a S D domain live hX hW hBias fresh trials last
  have hm : ret.state.mem=(executeNative M p s X W Bias a S D last program).state.mem := by
    funext q off
    dsimp only [executeNative]
    by_cases hit : ∃ port : Port S D,port.address a=(q,off)
    · rw [dif_pos hit]
      let port := Classical.choose hit
      have hp : port.address a=(q,off) := Classical.choose_spec hit
      have hs := source_port M p bindings s ret.state X W Bias a S D last ht.2 port
      rw [hp] at hs
      have he := (program.plan port).sound p (inputs M s X.region W.region Bias.region D port.row)
        (laws port) (fun _ h => h)
      simpa only [he] using hs
    · rw [dif_neg hit]
      exact ht.2.frame q off (fun row hrow => no_port_frame a S D q off hit row (List.mem_range.mp hrow))
  refine ⟨ret,hr,ht.1,hm,⟨rfl,?_,?_,?_,?_⟩,?_⟩
  · intro row hrow; rw [←hm]; exact ht.2.mean row hrow
  · intro row hrow; rw [←hm]; exact ht.2.rstd row hrow
  · intro row hrow col hc; rw [←hm]; exact ht.2.output row hrow col hc
  · intro q off hmiss; rw [←hm]; exact ht.2.frame q off hmiss
  · intro q hq off; rw [←hm]; exact hf q hq off

#axiomsClean Port.address_injective
#axiomsClean executeNative_port
#axiomsClean source_port
#axiomsClean no_port_frame
#axiomsClean host_native_complete
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Native
