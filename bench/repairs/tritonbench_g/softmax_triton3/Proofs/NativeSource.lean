import bench.repairs.tritonbench_g.softmax_triton3.Proofs.NativeGraph
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.NativeSource
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
noncomputable section
open Classical
set_option maxHeartbeats 4000000
abbrev liftState := SoftmaxTriton1.NativeGraph.liftState

/-- Only active output cells are projected from the temporary per-thread source
carrier. Every other cell retains its complete raw-word Cell and tag. -/
def projectState (V B : Nat) (out : RegionName) (n p : Nat) (s : State Word) (t : State Node) : State Word :=
  {s with mem := fun r i =>
    if r = out ∧ p*n ≤ i ∧ i < p*n+n then
      Cell.mk TileDType.real (((t.mem r i).read .real) (owner V B (i-p*n)))
    else s.mem r i}

def evaluate (inst : Instructions) (V : Nat) : Evaluator (State Word) := fun out inp mask rows n B p s =>
  (Floating.evaluate (algebra inst V) out inp mask rows n B p (liftState s)).map (projectState V B out n p s)

theorem project_active (V B n p c : Nat) (out : RegionName) (hc : c<n)
    (s : State Word) (t : State Node) :
    (projectState V B out n p s t).mem out (p*n+c)=
      Cell.mk TileDType.real (((t.mem out (p*n+c)).read .real) (owner V B c)) := by
  simp [projectState,show p*n+c<p*n+n by omega]

theorem source (inst : Instructions) (V : Nat) : Source Floating.access (evaluate inst V) (row inst V) := by
  constructor
  intro out inp mask rows n B p s t xs ms hB cover hp input masks matching run
  cases he : Floating.evaluate (algebra inst V) out inp mask rows n B p (liftState s) with
  | none => simp [evaluate,he] at run
  | some nodes =>
    have eq : projectState V B out n p s nodes=t := by simpa [evaluate,he] using run
    subst t
    have src := (Floating.source (algebra inst V)).row out inp mask rows n B p (liftState s) nodes
      (fun c => atom (xs c)) (ms.map (fun f c => atom (f c))) hB cover hp
      (by intro c hc;exact congrArg atom (input c hc))
      (by
        intro r hr c hc
        cases ms with
        | none => cases mask <;> simp_all
        | some f => exact congrArg atom (masks r hr c hc))
      (by simpa using matching) he
    constructor
    · intro c hc
      change ((projectState V B out n p s nodes).mem out (p*n+c)).read .real=_
      rw [project_active V B n p c out hc]
      change ((nodes.mem out (p*n+c)).read .real) (owner V B c)=_
      have val := src.1 c hc
      change (nodes.mem out (p*n+c)).read .real=_ at val
      rw [val]
      exact source_graph_lowering inst V n B xs ms c (lt_of_lt_of_le hc cover)
    · intro r i miss
      have off : ¬(r = out ∧ p*n ≤ i ∧ i < p*n+n) := by
        rintro ⟨hr,lo,hi⟩
        rcases miss with ne|no
        · exact ne hr
        · exact no (i-p*n) (by omega) (by omega)
      simp [Floating.access,SoftmaxTriton1.Floating.access,projectState,off]

theorem terminates (inst : Instructions) (V : Nat) (out inp : RegionName) (mask : Option RegionName)
    (rows n B p : Nat) (hB : 0<B) (s : State Word) :
    ∃ t,evaluate inst V out inp mask rows n B p s=some t := by
  obtain ⟨t,h⟩ := Floating.terminates (algebra inst V) out inp mask rows n B p hB (liftState s)
  exact ⟨projectState V B out n p s t,by simp [evaluate,h]⟩

#axiomsClean source
#axiomsClean terminates
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
