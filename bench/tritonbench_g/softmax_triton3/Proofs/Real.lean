import bench.tritonbench_g.softmax_triton3.Proofs.Model
import bench.tritonbench_g.softmax_triton3.Proofs.Mathematical_Real
import bench.tritonbench_g.softmax_triton1.Proofs.KernelMath
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.RealModel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.TritonBenchG.SoftmaxTriton3 (softmaxSpec)
noncomputable section
def access : Access BlockState MemCell ℝ where
  cells := BlockState.mem
  decode := fun c => match c.readAs .real with | some (some v) => v | _ => 0

def formula : Formula ℝ := fun n _ xs mask j =>
  let logits := fun k => xs k+(mask.getD (fun _ => 0)) k
  Real.exp (logits j)/∑ k ∈ Finset.range n,Real.exp (logits k)

def launchState (rows p : Nat) (s : BlockState) : BlockState :=
  {s with pids := fun ax => if ax=0 then p else 0
          numPids := fun ax => if ax=0 then rows else 1}

def evaluate : Evaluator BlockState := fun out inp mask rows n B p s =>
  exec (softmax_kernel out inp (mask.getD inp) n n B mask.isSome).toAlgKernel (launchState rows p s)

theorem source : Source access evaluate formula := by
  constructor
  intro out inp mask rows n B p s t xs ms hB cover hp input masks matching run
  cases mask with
  | none =>
    cases ms with
    | some ms => simp at matching
    | none =>
      obtain ⟨u,hu,hv,hf⟩ := VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel_region_run
        out inp inp n n B hB (launchState rows p s) (fun j => xs j.val)
        (by intro j hj;exact input j.val hj)
      have eq : u=t := Option.some.inj (hu.symm.trans run)
      subst t
      constructor
      · intro c hc
        let i : Fin B := ⟨c,lt_of_lt_of_le hc cover⟩
        have h := hv i hc
        rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmaxSpec_none_eq_formula n B hB _ i hc,
          SoftmaxTriton1.RealModel.padded_sum_eq n B cover (fun j => Real.exp (xs j))] at h
        simpa [formula,i,access,Access.read,BlockState.readMem,launchState,BlockState.pid_eq] using h
      · intro r a miss
        apply hf r a
        rcases miss with ne|off
        · exact Or.inl ne
        · exact Or.inr (fun j hj => off j.val hj)
  | some mask =>
    cases ms with
    | none => simp at matching
    | some ms =>
      obtain ⟨u,hu,hv,hf⟩ := VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel_masked_region_run
        out inp mask n n B hB (launchState rows p s) (fun j => xs j.val) (fun j => ms j.val)
        (by intro j hj;exact input j.val hj) (by intro j hj;exact masks mask rfl j.val hj)
      have eq : u=t := Option.some.inj (hu.symm.trans run)
      subst t
      constructor
      · intro c hc
        let i : Fin B := ⟨c,lt_of_lt_of_le hc cover⟩
        have h := hv i hc
        rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmaxSpec_mask_eq_formula n B hB _ _ i hc,
          SoftmaxTriton1.RealModel.padded_sum_eq n B cover (fun j => Real.exp (xs j+ms j))] at h
        simpa [formula,i,access,Access.read,BlockState.readMem,launchState,BlockState.pid_eq] using h
      · intro r a miss
        apply hf r a
        rcases miss with ne|off
        · exact Or.inl ne
        · exact Or.inr (fun j hj => off j.val hj)

theorem terminates (out inp : RegionName) (mask : Option RegionName) (rows n B p : Nat)
    (hB : 0<B) (s : BlockState) : ∃ t,evaluate out inp mask rows n B p s=some t := by
  cases mask with
  | none =>
    obtain ⟨t,h,_,_⟩ := VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel_region_run
      out inp inp n n B hB (launchState rows p s)
      (fun j => s.readMem inp (p*n+j.val)) (by intros;rfl)
    exact ⟨t,h⟩
  | some mask =>
    obtain ⟨t,h,_,_⟩ := VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel_masked_region_run
      out inp mask n n B hB (launchState rows p s)
      (fun j => s.readMem inp (p*n+j.val)) (fun j => s.readMem mask (p*n+j.val))
      (by intros;rfl) (by intros;rfl)
    exact ⟨t,h⟩

#specNonCircular formula avoiding [evaluate,softmax_kernel,softmaxSpec]
#axiomsClean source
#axiomsClean terminates
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.RealModel
