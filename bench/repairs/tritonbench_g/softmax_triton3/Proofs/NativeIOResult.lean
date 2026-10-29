import bench.repairs.tritonbench_g.softmax_triton3.Proofs.NativeSource
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.StructuralSafety
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.NativeIOResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
noncomputable section

/-- The complete selected source on encoded logical words. Arbitrary other
memory/register cells retain their full original type and payload. -/
theorem execute (inst : Instructions) (V : Nat) (out inp : RegionName)
    (mask : Option RegionName) (rows n B p : Nat)
    (positive : 0<B) (cover : n≤B) (valid : p<rows)
    (s : State Node) (xs : Nat → Word) (ms : Option (Nat → Word))
    (input : ∀ c,c<n → (s.mem inp (p*n+c)).read .real=atom (xs c))
    (masks : ∀ r,mask=some r → ∀ c,c<n →
      (s.mem r (p*n+c)).read .real=atom ((ms.getD (fun _ => 0)) c))
    (matching : mask.isSome=ms.isSome) :
    ∃ t,Floating.evaluate (algebra inst V) out inp mask rows n B p s=some t ∧
      (∀ c,c<n → mapCell (fun value => value (owner V B c)) (t.mem out (p*n+c))=
        .mk .real (row inst V n B xs ms c)) ∧
      (∀ r i,(r≠out ∨ ∀ c,c<n → i≠p*n+c) → t.mem r i=s.mem r i) := by
  obtain ⟨t,run⟩ := Floating.terminates (algebra inst V) out inp mask rows n B p positive s
  have inputs : ∀ c,c<n → Floating.access.read s inp (p*n+c)=atom (xs c) := input
  have maskInputs : ∀ r,mask=some r → ∀ c,c<n → Floating.access.read s r (p*n+c)=
      ((ms.map (fun f j => atom (f j))).getD (fun _ => Floating.access.read s inp 0)) c := by
    intro r hr c hc
    cases ms with
    | none => cases mask <;> simp_all
    | some values => exact masks r hr c hc
  have result := (Floating.source (algebra inst V)).row out inp mask rows n B p s t
    (fun c => atom (xs c)) (ms.map (fun f c => atom (f c))) positive cover valid
    inputs maskInputs (by simpa using matching) run
  refine ⟨t,run,?_,result.2⟩
  intro c hc
  have cb : c<B := lt_of_lt_of_le hc cover
  have typed := Floating.row_value (algebra inst V) out inp (mask.getD inp) n B
    mask.isSome positive (Floating.launchState rows p s) t run ⟨c,cb⟩ hc
  have typedRead : t.mem out (p*n+c)=.mk .real ((t.mem out (p*n+c)).read .real) := by
    simpa only [Floating.launchState,↓reduceIte,Cell.read_same] using
      typed.trans (congrArg (fun cell : Cell Node => Cell.mk .real (cell.read .real)) typed).symm
  rw [typedRead]
  change Cell.mk .real (((t.mem out (p*n+c)).read .real) (owner V B c))=_
  have value := result.1 c hc
  change (t.mem out (p*n+c)).read .real=_ at value
  rw [value,source_graph_lowering inst V n B xs ms c cb]

#axiomsClean execute
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.NativeIOResult
