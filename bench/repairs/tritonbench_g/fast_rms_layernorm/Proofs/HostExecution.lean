import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostMetadata

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Host
open VeriTile.Triton Storage
noncomputable section
variable {α β : Type}

structure Tensor (α : Type) where
  shape : List Nat
  device : Nat
  storageId : Nat
  buffer : Native.Shared.Buffer
  values : Nat → α

structure Outcome (β ρ : Type) where
  bytes : Heap β
  result : Except Failure ρ

/-- A runtime error may leave arbitrary private values. Successful launch
supplies only Unit; the source evaluator still has to execute the program. -/
abbrev Gate (α : Type) := Except (Failure × CellHeap α) Unit

def interrupted (h : Call) (roles : List RegionName) (before changes : CellHeap α) : CellHeap α :=
  fun a => if a.1∈roles ∧ a.2<cells h a.1 then changes a else before a

def early {ρ : Type} (entry : Heap β) (failure : Failure) : Outcome β ρ :=
  ⟨entry,.error failure⟩

def finish {ρ : Type} {h : Call} {p : Phase} {live : Caller} (prep : Prepared h p live)
    (encode : α → Fin 4 → β) (entry : Heap β) (memory : CellHeap α) (result : Except Failure ρ) : Outcome β ρ :=
  ⟨Storage.commit prep.storage encode entry (fun r o => memory (r,o)),result⟩

def makeContext (r : Request α) (h : Call) {live : Caller} (p : Prepared h .forward live)
    (actual : CellHeap α) : Context α :=
  ⟨h,r.shape.dims,r.weightShape,r.inputDevice,actual,p.storage⟩

def outputTensor (r : Request α) (h : Call) {live : Caller} (p : Prepared h .forward live)
    (actual : CellHeap α) : Tensor α :=
  ⟨r.shape.dims,r.inputDevice,(p.storage.allocation "y" (by simp [buffers])).storageId,
    (p.storage.allocation "y" (by simp [buffers])).buffer,
    fun i => actual ("y",i)⟩

def gradientTensors (ctx : Context α) {live : Caller} (p : BackwardPrepared ctx live)
    (actual : CellHeap α) : Tensor α × Tensor α :=
  (⟨ctx.inputShape,ctx.device,(p.storage.allocation "dx" (by simp [buffers])).storageId,
     (p.storage.allocation "dx" (by simp [buffers])).buffer,fun i => actual ("dx",i)⟩,
   ⟨ctx.weightShape,ctx.device,(p.storage.allocation "dw" (by simp [buffers])).storageId,
     (p.storage.allocation "dw" (by simp [buffers])).buffer,fun i => actual ("dw",i)⟩)

/-- Source evaluator is instantiated with the shared typed program in each
independent reader. Preparation and runtime failures are ordinary outcomes. -/
def forward (execute : Call → CellHeap α → Option (CellHeap α)) (r : Request α)
    (scratch : CellHeap α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : ∀ h,Preparation h .forward live) (launch : Gate α)
    (viewX viewW : Except Failure Unit := .ok ()) : Outcome β (Context α × Tensor α) :=
  match preflight r with
  | .error failure => early entry failure
  | .ok h => match viewX with
    | .error failure => early entry failure
    | .ok _ => match viewW with
      | .error failure => early entry failure
      | .ok _ => if r.weightShape.prod≠h.cols then early entry .reshape else
        match prepare h with
        | .error failure => early entry failure
        | .ok p =>
          let initial:=packed r scratch
          match (if h.rows=0 then .ok () else launch) with
          | .error (failure,privateValues) => finish p encode entry (interrupted h ["y","r"] initial privateValues) (.error failure)
          | .ok _ => match execute h initial with
            | none => finish p encode entry initial (.error .internal)
            | some actual => finish p encode entry actual (.ok (makeContext r h p actual,outputTensor r h p actual))

def backward (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : Except Failure (BackwardPrepared ctx live)) (weightLaunch inputLaunch zeroFill : Gate α)
    (resolveG : Except Failure Unit := .ok ())
    (savedRead : Except Failure Unit := .ok ()) : Outcome β (Tensor α × Tensor α) :=
  match savedRead with
  | .error failure => early entry failure
  | .ok _ => match backwardDevice ctx g with
    | .error failure => early entry failure
    | .ok _ => match resolveG with
      | .error failure => early entry failure
      | .ok _ => if g.shape.prod≠ctx.call.cells then early entry .reshape else
        match prepare with
        | .error failure => early entry failure
        | .ok p =>
          let initial:=seed ctx g
          if ctx.call.rows=0 then
            match zeroFill with
            | .error (failure,privateValues) => finish p.toPrepared encode entry (interrupted ctx.call ["dw"] initial privateValues) (.error failure)
            | .ok _ =>
              let actual:=zeroWeight ctx.call zero initial
              finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual))
          else match weightLaunch with
            | .error (failure,privateValues) => finish p.toPrepared encode entry (interrupted ctx.call ["dw"] initial privateValues) (.error failure)
            | .ok _ => match weight ctx.call initial with
              | none => finish p.toPrepared encode entry initial (.error .internal)
              | some weighted => match inputLaunch with
                | .error (failure,privateValues) => finish p.toPrepared encode entry (interrupted ctx.call ["dx"] weighted privateValues) (.error failure)
                | .ok _ => match input ctx.call weighted with
                  | none => finish p.toPrepared encode entry weighted (.error .internal)
                  | some actual => finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual))

theorem finish_frame {ρ : Type} {h : Call} {phase : Phase} {live : Caller} (p : Prepared h phase live)
    (encode : α → Fin 4 → β) (entry : Heap β) (memory : CellHeap α) (result : Except Failure ρ)
    (addr : Nat) (active : live addr) : (finish p encode entry memory result).bytes addr=entry addr := by
  exact Storage.commit_frame h phase p.storage encode entry (fun r o => memory (r,o)) addr (p.fresh addr active)

theorem forward_frame (execute : Call → CellHeap α → Option (CellHeap α)) (r : Request α)
    (scratch : CellHeap α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : ∀ h,Preparation h .forward live) (launch : Gate α)
    (viewX viewW : Except Failure Unit) (addr : Nat) (active : live addr) :
    (forward execute r scratch live encode entry prepare launch viewX viewW).bytes addr=entry addr := by
  unfold forward
  repeat' (first | exact finish_frame _ _ _ _ _ addr active | rfl | ((try dsimp only); split))

theorem backward_frame (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : Except Failure (BackwardPrepared ctx live)) (weightLaunch inputLaunch zeroFill : Gate α)
    (resolveG : Except Failure Unit) (addr : Nat) (active : live addr)
    (savedRead : Except Failure Unit := .ok ()) :
    (backward weight input zero ctx g live encode entry prepare weightLaunch inputLaunch zeroFill resolveG savedRead).bytes addr=entry addr := by
  unfold backward
  repeat' (first | exact finish_frame _ _ _ _ _ addr active | rfl | ((try dsimp only); split))

theorem forward_success (execute : Call → CellHeap α → Option (CellHeap α)) (r : Request α)
    (scratch : CellHeap α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (prepare : ∀ h,Preparation h .forward live) (h : Call) (p : Prepared h .forward live)
    (valid : validate r=.ok h) (ready : prepare h=.ok p) (actual : CellHeap α)
    (run : execute h (packed r scratch)=some actual) :
    forward execute r scratch live encode entry prepare (.ok ())=
      finish p encode entry actual (.ok (makeContext r h p actual,outputTensor r h p actual)) := by
  obtain ⟨_,_,_,_,weightShape,_⟩:=validated r h valid
  simp [forward,preflight_of_valid r h valid,weightShape,ready,run]

theorem backward_empty (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (p : BackwardPrepared ctx live) (valid : backwardValidate ctx g=.ok ()) (empty : ctx.call.rows=0)
    (weightLaunch inputLaunch : Gate α) :
    backward weight input zero ctx g live encode entry (.ok p) weightLaunch inputLaunch (.ok ())=
      finish p.toPrepared encode entry (zeroWeight ctx.call zero (seed ctx g))
        (.ok (gradientTensors ctx p (zeroWeight ctx.call zero (seed ctx g)))) := by
  have checks:=backward_valid ctx g valid
  simp [backward,checks.1,checks.2,empty]

theorem backward_success (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α)
    (ctx : Context α) (g : Upstream α) (live : Caller) (encode : α → Fin 4 → β) (entry : Heap β)
    (p : BackwardPrepared ctx live) (valid : backwardValidate ctx g=.ok ()) (nonempty : ctx.call.rows≠0)
    (weighted actual : CellHeap α) (wrun : weight ctx.call (seed ctx g)=some weighted)
    (xrun : input ctx.call weighted=some actual) (zeroFill : Gate α) :
    backward weight input zero ctx g live encode entry (.ok p) (.ok ()) (.ok ()) zeroFill=
      finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual)) := by
  have checks:=backward_valid ctx g valid
  simp [backward,checks.1,checks.2,nonempty,wrun,xrun]

#axiomsClean forward_frame
#axiomsClean backward_frame
#axiomsClean forward_success
#axiomsClean backward_empty
#axiomsClean backward_success
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Host
