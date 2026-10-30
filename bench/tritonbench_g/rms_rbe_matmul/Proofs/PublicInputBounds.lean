import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicIOLayout
import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicSourceSafety

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem program_dimensions (B : Nat) (p : Parameters) (q : KernelGrid.Program)
    (member : q∈KernelGrid.programs B p) : q.1<B ∧ 0<p.M ∧ 0<p.N := by
  have h := List.mem_product.mp member
  have batch : q.1<B := List.mem_range.mp h.1
  have tile : q.2<KernelGrid.tilesM p*KernelGrid.tilesN p := List.mem_range.mp h.2
  refine ⟨batch,?_,?_⟩
  · by_contra h
    have zero : p.M=0 := by omega
    simp [KernelGrid.tilesM,zero] at tile
  · by_contra h
    have zero : p.N=0 := by omega
    simp [KernelGrid.tilesN,zero] at tile

theorem program_bounds (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues α (inputs fp8 B p)) (bounds : RegionName → Nat)
    (bounded : (io a fp8 B p).InBounds bounds xs) (state : State α)
    (q : KernelGrid.Program) (member : q∈KernelGrid.programs B p) :
    SourceBounds bounds p (KernelGrid.atProgram state q) := by
  obtain ⟨batch,rows,cols⟩ := program_dimensions B p q member
  constructor
  · intro cur i hi
    have hk : i.2.1.val+cur*64<p.K := by omega
    have hm : row p (KernelGrid.atProgram state q) i.1%p.M<p.M := Nat.mod_lt _ rows
    have h := bounded.2.1 (0 : Fin 3) (⟨q.1,batch⟩,⟨_,hm⟩,⟨_,hk⟩,PUnit.unit) trivial
    simpa only [io,read,inputs,Fin.cases_zero,xOffset,KernelGrid.atProgram,Nat.add_assoc] using h
  · intro cur i hi
    have hk : i.1.val+cur*64<p.K := by omega
    have hn : col p (KernelGrid.atProgram state q) i.2.1%p.N<p.N := Nat.mod_lt _ cols
    have h := bounded.2.1 (1 : Fin 3) (⟨_,hn⟩,⟨_,hk⟩,PUnit.unit) trivial
    change (i.1.val+cur*64)+(col p (KernelGrid.atProgram state q) i.2.1%p.N)*p.K<bounds "W" at h
    unfold wOffset
    omega
  · intro cur i hi
    have hk : i.2.1.val+cur*64<p.K := by omega
    have h := bounded.2.1 (2 : Fin 3) (⟨_,hk⟩,PUnit.unit) trivial
    exact h
  · intro i hi
    have active : row p (KernelGrid.atProgram state q) i.1<p.M ∧
        col p (KernelGrid.atProgram state q) i.2.1<p.N := by simpa [KernelSource.active] using hi
    have h := bounded.2.2.1 (0 : Fin 1)
      (⟨q.1,batch⟩,⟨_,active.1⟩,⟨_,active.2⟩,PUnit.unit) trivial
    exact h

theorem program_bytes (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues α (inputs fp8 B p)) (admitted : domain fp8 B p xs)
    (state : State α) (binding : (io a fp8 B p).InputBinding xs state)
    (q : KernelGrid.Program) (member : q∈KernelGrid.programs B p) (cur : Nat) :
    BlockBytes fp8 p (KernelGrid.atProgram state q) cur := by
  intro hf
  subst fp8
  obtain ⟨batch,rows,cols⟩ := program_dimensions B p q member
  change (∀ j, Float8E5Bits.isInt8 (xs (1 : Fin 3) j)=Bool.true) ∧
    (∀ j, Float8E5Bits.isInt8 (xs (2 : Fin 3) j)=Bool.true) at admitted
  constructor
  · intro i hi
    have hk : i.2.1.val+cur*64<p.K := by omega
    have h := binding (2 : Fin 3) (⟨_,hk⟩,PUnit.unit) trivial
    change (state.mem "RMS" (rOffset cur i)).read .int=xs (2 : Fin 3) (⟨_,hk⟩,PUnit.unit) at h
    change Float8E5Bits.isInt8 ((state.mem "RMS" (rOffset cur i)).read .int)=Bool.true
    rw [h]
    exact admitted.2 _
  · intro i hi
    have hk : i.1.val+cur*64<p.K := by omega
    have hn : col p (KernelGrid.atProgram state q) i.2.1%p.N<p.N := Nat.mod_lt _ cols
    have h := binding (1 : Fin 3) (⟨_,hn⟩,⟨_,hk⟩,PUnit.unit) trivial
    have address : wOffset p (KernelGrid.atProgram state q) cur i=
        (i.1.val+cur*64)+(col p (KernelGrid.atProgram state q) i.2.1%p.N)*p.K := by
      unfold wOffset
      omega
    change Float8E5Bits.isInt8 ((state.mem "W" (wOffset p (KernelGrid.atProgram state q) cur i)).read .int)=Bool.true
    rw [address]
    change (state.mem "W" ((i.1.val+cur*64)+(col p (KernelGrid.atProgram state q) i.2.1%p.N)*p.K)).read .int=
      xs (1 : Fin 3) (⟨_,hn⟩,⟨_,hk⟩,PUnit.unit) at h
    rw [h]
    exact admitted.1 _

#axiomsClean program_bounds
#axiomsClean program_bytes
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
