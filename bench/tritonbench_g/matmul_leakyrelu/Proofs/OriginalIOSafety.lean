import bench.tritonbench_g.matmul_leakyrelu.Proofs.OriginalFacts
import bench.tritonbench_g.matmul_leakyrelu.Proofs.OriginalSafety

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.OriginalIOSafety
open VeriTile.Triton FP.Structural Interface Original
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
variable {α : Type} [Inhabited α]

theorem program_safe (M : Algebra α) (D : FPValue.Dot α)
    (p : KernelInterface.Parameters) (order : Schedule p.request)
    (vals : Inputs α p) (bounds : RegionName → Nat)
    (backed : (io M D p order).InBounds bounds vals)
    (pid : Nat) (inside : pid<Whole.programs p.request) (s : State α) :
    (job M D p pid).Safe bounds s := by
  have rows : 0<p.rows := by
    by_contra hz
    have empty : p.rows=0 := by omega
    simpa [Whole.programs,Whole.tileRows,KernelInterface.Parameters.request,empty] using inside
  have cols : 0<p.columns := by
    by_contra hz
    have empty : p.columns=0 := by omega
    simpa [Whole.programs,Whole.tileColumns,KernelInterface.Parameters.request,empty] using inside
  change TraceSafe (FPValue.backend M D) bounds
    (FPValue.source p.request.parameters .single) (Whole.instanceState p.request pid s)
  apply OriginalSafety.source_safe
  · intro k i active
    have inner : k*32+i.2.1.val<p.inner := by
      change i.2.1.val<p.inner-k*32 at active
      omega
    let row := FPValue.aRow p.request.parameters (Whole.instanceState p.request pid s) i.1
    have hr : row<p.rows := Nat.mod_lt _ rows
    have b := backed.2.1 (0 : Fin 2) (⟨row,hr⟩,⟨k*32+i.2.1.val,inner⟩,()) trivial
    change row*p.strideAM+(k*32+i.2.1.val)*p.strideAK<bounds p.A at b
    change row*p.strideAM+i.2.1.val*p.strideAK+k*(32*p.strideAK)<bounds p.A
    convert b using 1 <;> ring
  · intro k i active
    have inner : k*32+i.1.val<p.inner := by
      change i.1.val<p.inner-k*32 at active
      omega
    let col := FPValue.bCol p.request.parameters (Whole.instanceState p.request pid s) i.2.1
    have hc : col<p.columns := Nat.mod_lt _ cols
    have b := backed.2.1 (1 : Fin 2) (⟨k*32+i.1.val,inner⟩,⟨col,hc⟩,()) trivial
    change (k*32+i.1.val)*p.strideBK+col*p.strideBN<bounds p.B at b
    change i.1.val*p.strideBK+col*p.strideBN+k*(32*p.strideBK)<bounds p.B
    convert b using 1 <;> ring
  · intro i active
    have b := backed.2.2.1 (0 : Fin 1) (⟨_,active.1⟩,⟨_,active.2⟩,()) trivial
    simpa only [Original.io,FPValue.cOffset,Whole.Request.parameters,
      KernelInterface.Parameters.request,Whole.outputOffset,Nat.one_mul] using b

theorem whole (M : Algebra α) (D : FPValue.Dot α)
    (p : KernelInterface.Parameters) (order : Schedule p.request)
    (vals : Inputs α p) (bounds : RegionName → Nat)
    (backed : (io M D p order).InBounds bounds vals) (s : State α) :
    ScheduleSafe bounds ((io M D p order).schedule vals) s := by
  have all : ∀ pid∈order.order,pid<Whole.programs p.request := by
    intro pid inside
    exact List.mem_range.mp (order.complete.mem_iff.mp inside)
  change ScheduleSafe bounds (order.order.map (job M D p)) s
  generalize he : order.order=ps at all ⊢
  clear he
  induction ps generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨program_safe M D p order vals bounds backed pid (all pid (by simp)) s,?_⟩
    intro next _
    exact ih next (fun q hq => all q (by simp [hq]))

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.OriginalIOSafety
