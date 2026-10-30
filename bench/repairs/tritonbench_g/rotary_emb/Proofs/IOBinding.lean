import bench.repairs.tritonbench_g.rotary_emb.IO
import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedResult

namespace VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

@[simp] theorem invocation_initial (rest : Algebra A) (ops : Ops A) (shape : Shape)
    (pids : Nat → Nat) (s : State A) :
    (invocation rest ops shape pids).initial s={s with pids := pids} := rfl

theorem active_eq (shape : Shape) (s : State A) (key : Bool) (j : Lane shape) :
    Typed.active (Typed.layout shape) s key j=payloadActive shape s.pids key j := rfl

theorem offset_eq (shape : Shape) (s : State A) (key odd : Bool) (j : Lane shape) :
    Typed.offset (Typed.layout shape) s key odd j=payloadOffset shape s.pids key odd j := by
  cases key <;> cases odd <;>
    simp [Typed.offset,Typed.layout,Typed.token,Typed.head,Typed.dimension,payloadOffset,
      Nat.mul_assoc,Nat.add_assoc]

theorem coefficient_eq (shape : Shape) (s : State A) (sine odd : Bool) (j : Lane shape) :
    Typed.coefficient (Typed.layout shape) s sine odd j=
      coefficientOffset shape s.pids sine odd j := by
  cases sine <;> cases odd <;>
    simp [Typed.coefficient,Typed.layout,Typed.token,Typed.dimension,coefficientOffset,Nat.add_assoc]

theorem payload_read {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (pids : Nat → Nat) (values : Inputs A shape) (s : State A)
    (bound : (kernelIO rest ops shape pids).InputBinding values s)
    (key odd : Bool) (j : Lane shape) (live : payloadActive shape pids key j) :
    (s.mem (if key then "K" else "Q") (payloadOffset shape pids key odd j)).read .real=
      payload values key odd j := by
  cases key <;> cases odd
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨0,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨1,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨2,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨3,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)

theorem coefficient_read {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (pids : Nat → Nat) (values : Inputs A shape) (s : State A)
    (bound : (kernelIO rest ops shape pids).InputBinding values s)
    (sine odd : Bool) (j : Lane shape) (live : coefficientActive shape pids j) :
    (s.mem (if sine then "Sin" else "Cos") (coefficientOffset shape pids sine odd j)).read .real=
      coefficient values sine odd j := by
  cases sine <;> cases odd
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨4,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨5,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨6,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory,payload,coefficient,mapValue] using
      bound ⟨7,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)

theorem payload_bounds (rest : Algebra A) (ops : Ops A) (shape : Shape)
    (pids : Nat → Nat) (values : Inputs A shape) (bounds : RegionName → Nat)
    (backed : (kernelIO rest ops shape pids).InBounds bounds values)
    (key odd : Bool) (j : Lane shape) (live : payloadActive shape pids key j) :
    payloadOffset shape pids key odd j<bounds (if key then "K" else "Q") := by
  cases key <;> cases odd
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨0,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨1,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨2,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨3,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)

theorem coefficient_bounds (rest : Algebra A) (ops : Ops A) (shape : Shape)
    (pids : Nat → Nat) (values : Inputs A shape) (bounds : RegionName → Nat)
    (backed : (kernelIO rest ops shape pids).InBounds bounds values)
    (sine odd : Bool) (j : Lane shape) (live : coefficientActive shape pids j) :
    coefficientOffset shape pids sine odd j<bounds (if sine then "Sin" else "Cos") := by
  cases sine <;> cases odd
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨4,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨5,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨6,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)
  · simpa [kernelIO,memory] using
      backed.2.1 ⟨7,by simp [kernelIO,memory,inputChannels]⟩ j (by simpa [kernelIO,memory] using live)

/-- The old raw-state independent pair formula is bound to the eight logical
channels. Masked-off payloads need no binding; coefficient masks stay broader. -/
theorem value_pairFormula {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (pids : Nat → Nat) (values : Inputs A shape) (s : State A)
    (bound : (kernelIO rest ops shape pids).InputBinding values s)
    (key odd : Bool) (j : Lane shape) (live : payloadActive shape pids key j) :
    Typed.value ops (Typed.layout shape) {s with pids := pids} key odd j=
      Cell.mk .real (pairFormula ops values key odd j) := by
  have coeffLive : coefficientActive shape pids j := ⟨live.1,live.2.2⟩
  unfold Typed.value
  simp only [offset_eq,coefficient_eq]
  change Cell.mk .real (if odd then
    ops.add (ops.mul ((s.mem (if key then "K" else "Q") (payloadOffset shape pids key Bool.false j)).read .real)
      ((s.mem "Sin" (coefficientOffset shape pids Bool.true odd j)).read .real))
      (ops.mul ((s.mem (if key then "K" else "Q") (payloadOffset shape pids key Bool.true j)).read .real)
        ((s.mem "Cos" (coefficientOffset shape pids Bool.false odd j)).read .real))
    else ops.sub (ops.mul ((s.mem (if key then "K" else "Q") (payloadOffset shape pids key Bool.false j)).read .real)
      ((s.mem "Cos" (coefficientOffset shape pids Bool.false odd j)).read .real))
      (ops.mul ((s.mem (if key then "K" else "Q") (payloadOffset shape pids key Bool.true j)).read .real)
        ((s.mem "Sin" (coefficientOffset shape pids Bool.true odd j)).read .real)))=_
  rw [payload_read rest ops shape pids values s bound key Bool.false j live,
    payload_read rest ops shape pids values s bound key Bool.true j live]
  have cosine := coefficient_read rest ops shape pids values s bound Bool.false odd j coeffLive
  have sine := coefficient_read rest ops shape pids values s bound Bool.true odd j coeffLive
  simp only [Bool.false_eq_true,↓reduceIte] at cosine sine
  rw [cosine,sine]
  rfl

theorem outside_typed (rest : Algebra A) (ops : Ops A) (shape : Shape)
    (pids : Nat → Nat) (values : Inputs A shape) (s : State A) (r : RegionName) (offset : Nat)
    (outside : (kernelIO rest ops shape pids).Outside values r offset) :
    Typed.Outside shape {s with pids := pids} r offset := by
  intro key
  by_cases same : r=Typed.region (Typed.layout shape) key
  · right
    intro odd j live
    rw [offset_eq]
    cases key <;> cases odd
    · exact (outside.1 ⟨0,by simp [kernelIO,memory,outputChannels]⟩ j (by simpa [kernelIO,memory,active_eq] using live)).resolve_left
        (fun ne => ne same)
    · exact (outside.1 ⟨1,by simp [kernelIO,memory,outputChannels]⟩ j (by simpa [kernelIO,memory,active_eq] using live)).resolve_left
        (fun ne => ne same)
    · exact (outside.1 ⟨2,by simp [kernelIO,memory,outputChannels]⟩ j (by simpa [kernelIO,memory,active_eq] using live)).resolve_left
        (fun ne => ne same)
    · exact (outside.1 ⟨3,by simp [kernelIO,memory,outputChannels]⟩ j (by simpa [kernelIO,memory,active_eq] using live)).resolve_left
        (fun ne => ne same)
  · exact Or.inl same

end
end VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
