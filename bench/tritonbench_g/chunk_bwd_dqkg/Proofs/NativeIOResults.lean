import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeFormulaBinding
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealIOResults

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeIOResults
open VeriTile Triton FP.Structural Whole KernelNativeCarrier AddressBounds
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem preparation (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (a : CTA c) (s : State Node) :
    (IO.nativeJob isa route scale c a).initial s=
      Typed.prepare (nativeConfig c) s (KernelNative.nativeCTA a) := rfl

theorem schedule (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (order : List (CTA c)) (s : State Node) :
    runSchedule (order.map (IO.nativeJob isa route scale c)) s=
      Typed.runGrid (algebra isa route scale) dotModel (nativeConfig c) (order.map KernelNative.nativeCTA) s := by
  induction order generalizing s with
  | nil=>rfl
  | cons a rest ih=>
    simp only [List.map_cons,runSchedule,Typed.runGrid]
    change (FP.Structural.exec (FP.ProfiledRewrite.engine (Typed.withDot (algebra isa route scale) dotModel))
      (kernel (nativeConfig c)) (Typed.prepare (nativeConfig c) s (KernelNative.nativeCTA a))).bind _ =
      (FP.Structural.exec (FP.ProfiledRewrite.engine (Typed.withDot (algebra isa route scale) dotModel))
      (kernel (nativeConfig c)) (Typed.prepare (nativeConfig c) s (KernelNative.nativeCTA a))).bind _
    congr 1
    funext t
    exact ih t

def formula (isa : Instructions) (route : Routing) (scale : Word) (c : Config) :=
  fun vals=>IO.nativeResult c (Target.native isa route scale c (IO.nativeValues c vals))

theorem correct (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c))
    (vals : ChannelValues Node (IO.nativeInputs c)) (initial : State Node)
    (inputs : (IO.nativeIO isa route scale c order).InputBinding vals initial) :
    ∃ final,runSchedule (order.map (IO.nativeJob isa route scale c)) initial=some final ∧
      (IO.nativeIO isa route scale c order).OutputBinding vals (formula isa route scale c vals) final ∧
      (IO.nativeIO isa route scale c order).Frame vals initial final := by
  have ho' : (order.map KernelNative.nativeCTA).Perm (allCTAs (nativeConfig c)) := by
    rw [←KernelNative.all_native]
    exact ho.map _
  obtain ⟨final,ran,hq,hk,hg,frame⟩:=Typed.whole_result (algebra isa route scale) dotModel
    (nativeConfig c) initial (order.map KernelNative.nativeCTA) ho'
  rw [←KernelNative.all_native,KernelNative.native_writes] at frame
  have reads:=NativeFormulaBinding.logical_inputs isa route scale c order initial vals inputs
  let μ:=Target.inputs c (IO.nativeValues c vals)
  refine ⟨final,(schedule isa route scale c order initial).trans ran,?_,?_⟩
  · intro o j _
    change Fin 3 at o
    fin_cases o
    · change mapCell id (final.mem "dq" j.1.val)=Cell.mk .fp32 (Target.native isa route scale c (IO.nativeValues c vals) 0 j.1)
      have bound : j.1.val<c.heads*c.T*c.K:=by simpa [IO.nativeIO,IO.nativeOutputs,IO.outputSize,IO.outputRegion,extent] using j.1.isLt
      let p:=Target.qCoordinates c j.1.val bound
      have value:=hq p.1 p.2.1 p.2.2
      change final.mem "dq" ((p.1.val*c.T+p.2.1.val)*c.K+p.2.2.val)=_ at value
      rw [RealIOResults.q_coordinates c j.1.val bound] at value
      rw [value]
      apply congrArg (Cell.mk .fp32)
      change Typed.resultQ (algebra isa route scale) dotModel (nativeConfig c) initial
        (KernelNative.nativeCTA (qOwner c p.1 p.2.1 p.2.2)) (rowLane p.2.1.val) (keyLane c p.2.2.val)=_
      rw [(KernelNative.tiles isa route scale c initial (qOwner c p.1 p.2.1 p.2.2) _ _).1]
      simp only [KernelNative.tileQ,NativeFormulaBinding.accumulate isa route c initial μ reads]
      rw [(NativeFormulaBinding.answers isa route scale c initial μ reads _ _).1]
      rfl
    · change mapCell id (final.mem "dk" j.1.val)=Cell.mk .fp32 (Target.native isa route scale c (IO.nativeValues c vals) 1 j.1)
      have bound : j.1.val<c.heads*c.T*c.K:=by simpa [IO.nativeIO,IO.nativeOutputs,IO.outputSize,IO.outputRegion,extent] using j.1.isLt
      let p:=Target.qCoordinates c j.1.val bound
      have value:=hk p.1 p.2.1 p.2.2
      change final.mem "dk" ((p.1.val*c.T+p.2.1.val)*c.K+p.2.2.val)=_ at value
      rw [RealIOResults.q_coordinates c j.1.val bound] at value
      rw [value]
      apply congrArg (Cell.mk .fp32)
      change Typed.resultK (algebra isa route scale) dotModel (nativeConfig c) initial
        (KernelNative.nativeCTA (qOwner c p.1 p.2.1 p.2.2)) (rowLane p.2.1.val) (keyLane c p.2.2.val)=_
      rw [(KernelNative.tiles isa route scale c initial (qOwner c p.1 p.2.1 p.2.2) _ _).2.1]
      simp only [KernelNative.tileK,NativeFormulaBinding.accumulate isa route c initial μ reads]
      rw [(NativeFormulaBinding.answers isa route scale c initial μ reads _ _).2.1]
      rfl
    · change mapCell id (final.mem "dg" j.1.val)=Cell.mk .real (Target.native isa route scale c (IO.nativeValues c vals) 2 j.1)
      have bound : j.1.val<c.NK*c.heads*c.T:=by simpa [IO.nativeIO,IO.nativeOutputs,IO.outputSize,IO.outputRegion,extent] using j.1.isLt
      let p:=Target.gCoordinates c j.1.val bound
      have value:=hg p.1 p.2.1 p.2.2
      change final.mem "dg" ((p.1.val*c.heads+p.2.1.val)*c.T+p.2.2.val)=_ at value
      rw [RealIOResults.g_coordinates c j.1.val bound] at value
      rw [value]
      apply congrArg (Cell.mk .real)
      change Typed.resultG (algebra isa route scale) dotModel (nativeConfig c) initial
        (KernelNative.nativeCTA (gOwner c p.1 p.2.1 p.2.2)) (rowLane p.2.2.val)=_
      rw [(KernelNative.tiles isa route scale c initial (gOwner c p.1 p.2.1 p.2.2) _ (keyLane c 0)).2.2]
      simp only [KernelNative.tileG,NativeFormulaBinding.accumulate isa route c initial μ reads]
      rw [(NativeFormulaBinding.answers isa route scale c initial μ reads _ _).2.2]
      rfl
  · intro r i outside
    apply frame
    rintro ⟨a,_,writes⟩
    rcases (writes_cases c a r i).mp writes with ⟨rfl,row,e,active,rfl⟩|⟨rfl,row,e,active,rfl⟩|⟨rfl,row,active,rfl⟩
    · have h:=outside.1 (0 : Fin 3) (⟨_,q_address_bound c a row e active⟩,PUnit.unit) trivial
      exact h.elim (fun hn=>hn rfl) (fun hn=>hn rfl)
    · have h:=outside.1 (1 : Fin 3) (⟨_,q_address_bound c a row e active⟩,PUnit.unit) trivial
      exact h.elim (fun hn=>hn rfl) (fun hn=>hn rfl)
    · have h:=outside.1 (2 : Fin 3) (⟨_,g_address_bound c a row active⟩,PUnit.unit) trivial
      exact h.elim (fun hn=>hn rfl) (fun hn=>hn rfl)

#axiomsClean correct
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeIOResults
