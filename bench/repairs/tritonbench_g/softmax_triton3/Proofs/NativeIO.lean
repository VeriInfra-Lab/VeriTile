import bench.repairs.tritonbench_g.softmax_triton3.IO
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.NativeIOResult
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
noncomputable section
/-- Independent ordered native graph: source subtraction, optional mask add,
log2(e) multiply, exp2, selected reduction and division. -/
def reference (inst : Instructions) (profile : Profile) := NativeGraph.row inst profile.vector

def negateWord (x : Word) : Word := x ^^^ (BitVec.ofNat 32 (2^31))

/-- Extend a logical tile outside its shape; those columns are never active. -/
def tileWords {B : Nat} (xs : Values Word .real [B]) (c : Nat) : Word :=
  if h : c<B then xs (⟨c,h⟩,PUnit.unit) else 0

/-- State-free native formula on the declared logical input channels. -/
def nativeTarget (inst : Instructions) (profile : Profile) (out inp : RegionName)
    (mask : Option RegionName) (n B : Nat)
    (values : ChannelValues Word [⟨.real,[B],inp⟩,⟨.real,[B],mask.getD inp⟩]) :
    ChannelValues Word [⟨.real,[B],out⟩] :=
  Fin.cases (fun j => reference inst profile n B (tileWords (values ⟨0,by simp⟩))
    (if mask.isSome then some (tileWords (values ⟨1,by simp⟩)) else none) j.1.val)
    (fun o => Fin.elim0 o)

/-- The complete row source executes safely and yields the independent native
instruction graph on every FP32 word. Primitive instruction semantics remain
explicit in `inst`; no floating transformation law is assumed. -/
theorem correct (inst : Instructions) (profile : Profile)
    (out inp : RegionName) (mask : Option RegionName) (rows n B p : Nat)
    (positive : 0<B) (cover : n≤B) (valid : p<rows) :
    (FPEquiv.softmaxIO inst profile out inp mask rows n B p).Implements
      (nativeTarget inst profile out inp mask n B) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    have hin : ∀ j : Fin B,j.val<n → p*n+j.val<bounds inp :=
      fun j hj => by
        simpa [FPEquiv.softmaxIO] using hb.2.1 ⟨0,by simp [FPEquiv.softmaxIO]⟩ (j,PUnit.unit)
          (by simpa [FPEquiv.softmaxIO] using hj)
    have hm : mask.isSome=Bool.true → ∀ j : Fin B,j.val<n → p*n+j.val<bounds (mask.getD inp) :=
      fun h j hj => by
        simpa [FPEquiv.softmaxIO] using hb.2.1 ⟨1,by simp [FPEquiv.softmaxIO]⟩ (j,PUnit.unit)
          (by simpa [FPEquiv.softmaxIO] using And.intro hj h)
    have hout : ∀ j : Fin B,j.val<n → p*n+j.val<bounds out :=
      fun j hj => by
        simpa [FPEquiv.softmaxIO] using hb.2.2.1 ⟨0,by simp [FPEquiv.softmaxIO]⟩ (j,PUnit.unit)
          (by simpa [FPEquiv.softmaxIO] using hj)
    simpa only [FPEquiv.softmaxIO,scheduleSafe_singleton,Invocation.Safe,Invocation.initial,
      bindRegisters,List.foldl_nil,Option.getD_some] using
      Floating.source_trace_safe (algebra inst profile.vector) out inp (mask.getD inp)
        n B mask.isSome positive bounds (Floating.launchState rows p initial) hin hm hout
  · intro vals initial hi
    let xs := tileWords (vals ⟨0,by simp [FPEquiv.softmaxIO]⟩)
    let ms := if mask.isSome then some (tileWords (vals ⟨1,by simp [FPEquiv.softmaxIO]⟩)) else none
    have input : ∀ c,c<n → (initial.mem inp (p*n+c)).read .real=atom (xs c) := by
      intro c hc
      have cb : c<B := lt_of_lt_of_le hc cover
      simpa only [xs,tileWords,dif_pos cb,mapValue] using
        hi ⟨0,by simp [FPEquiv.softmaxIO]⟩ (⟨c,cb⟩,PUnit.unit) (by simpa [FPEquiv.softmaxIO] using hc)
    have masks : ∀ r,mask=some r → ∀ c,c<n →
        (initial.mem r (p*n+c)).read .real=atom ((ms.getD (fun _ => 0)) c) := by
      intro r hr c hc
      have cb : c<B := lt_of_lt_of_le hc cover
      have h := hi ⟨1,by simp [FPEquiv.softmaxIO]⟩ (⟨c,cb⟩,PUnit.unit) (by simpa [FPEquiv.softmaxIO,hr] using hc)
      simpa only [FPEquiv.softmaxIO,hr,Option.getD_some,ms,Option.isSome_some,↓reduceIte,
        tileWords,dif_pos cb,mapValue] using h
    have matching : mask.isSome=ms.isSome := by simp [ms];cases mask <;> rfl
    obtain ⟨t,run,values,frame⟩ := NativeIOResult.execute inst profile.vector out inp mask
      rows n B p positive cover valid initial xs ms input masks matching
    refine ⟨t,?_,?_,?_⟩
    · simpa only [FPEquiv.softmaxIO,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_some,Floating.evaluate,Floating.launchState] using run
    · intro o j hj
      fin_cases o
      exact values j.1.val hj
    · intro r offset outside
      apply frame r offset
      by_cases hr : r=out
      · right
        intro c hc he
        have h := outside.1 ⟨0,by simp [FPEquiv.softmaxIO]⟩
          (⟨c,lt_of_lt_of_le hc cover⟩,PUnit.unit) (by simpa [FPEquiv.softmaxIO] using hc)
        have h' : r≠out ∨ offset≠p*n+c := by simpa [FPEquiv.softmaxIO] using h
        exact (h'.resolve_left (fun ne => ne hr)) he
      · exact Or.inl hr

#specNonCircular nativeTarget avoiding [softmax_kernel,NativeGraph.evaluate,Floating.evaluate]
#axiomsClean correct
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.NativeIO
