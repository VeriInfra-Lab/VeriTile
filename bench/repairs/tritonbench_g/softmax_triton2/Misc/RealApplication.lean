import bench.repairs.tritonbench_g.softmax_triton2.Proofs.Real

/-! Optional checked Tensor application of the independent Real row result. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton2.RealProof
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

def access : Access BlockState MemCell ℝ :=
  ⟨BlockState.mem,fun c => match c.readAs .real with | some (some v) => v | _ => 0⟩

theorem checked (x : Object) (h : InputContract x.metadata) (out : RegionName) (ob : Nat)
    (read : Nat → ℝ) (s : BlockState) (binding : InputMemory access read x s)
    (pid : s.pid < rows x.metadata) :
    ∃ t,
      exec (SoftmaxTriton2.softmax_kernel out x.region x.tensor.storageOffset ob
        (cols x.metadata) 1 (cols x.metadata) 1 (cols x.metadata)
        (blockSize (cols x.metadata))).toAlgKernel s = some t ∧
      (∀ c, c < cols x.metadata →
        t.readMem out (ob+s.pid*cols x.metadata+c) =
          reference (cols x.metadata) (logicalValue Neg.neg read x s.pid) c) ∧
      (∀ r a, (∀ c, c < cols x.metadata →
        ¬(out=r ∧ ob+s.pid*cols x.metadata+c=a)) → t.mem r a=s.mem r a) := by
  let n := cols x.metadata
  let B := blockSize n
  have hB := (blockSize_bounds n).1
  have cover := (blockSize_bounds n).2
  obtain ⟨t,he⟩ := row_exec out x.region x.tensor.storageOffset ob n 1 n 1 n B hB s
  refine ⟨t,he,?_,?_⟩
  · intro c hc
    let i : Fin B := ⟨c,lt_of_lt_of_le hc cover⟩
    have value := row_value out x.region x.tensor.storageOffset ob n 1 n 1 n B hB
      (by intro i j _ _ eq; apply Fin.ext; simpa using eq) s t he i hc
    have same : ∀ j : Fin B, j.val < n →
        s.readMem x.region (x.tensor.storageOffset+s.pid*n+j.val*1) =
          logicalValue Neg.neg read x s.pid j.val := by
      intro j hj
      simpa [n,access,Access.read,BlockState.readMem] using
        input_read access Neg.neg read x h s binding s.pid j.val pid hj
    rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr n B _
      (fun j => logicalValue Neg.neg read x s.pid j.val) same i,
      VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula n B hB _ i hc,
      RealModel.padded_sum_eq n B cover (fun j => Real.exp (logicalValue Neg.neg read x s.pid j))] at value
    simpa [reference,i,n] using value
  · intro r a miss
    exact row_frame out x.region x.tensor.storageOffset ob n 1 n 1 n B hB s t he r a
      (by intro i hi; simpa using miss i.val hi)

#axiomsClean checked
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.RealProof
