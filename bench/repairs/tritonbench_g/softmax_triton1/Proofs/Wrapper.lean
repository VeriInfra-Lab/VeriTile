import bench.repairs.tritonbench_g.softmax_triton1.Proofs.SourceContract
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Storage

namespace VeriTile.Bench.Repairs.SoftmaxTriton1

structure Returned {S C V} (a : Access S C V) (neg : V → V) (layout : Layout)
    (formula : Nat → Nat → (Nat → V) → Nat → V)
    (x y : Tensor) (s t : World S) : Prop where
  shape : SameShape x y
  valid : Valid layout t y
  normal : y.negative=Bool.false
  injective : y.Injective
  fresh : ¬s.live y.region
  frame : Preserves a s t
  values : ∀ r c, r<x.rows → c<x.cols →
    logical a neg t y r c=formula x.cols (blockSize x.cols) (logical a neg s x r) c

theorem wrapper_return {S C V} (a : Access S C V) (neg : V → V) (layout : Layout)
    (evaluate : Tensor → Tensor → Nat → Nat → S → Option S)
    (formula : Nat → Nat → (Nat → V) → Nat → V)
    (source : SourceContract a evaluate formula) (rt : Runtime S)
    (runtime : RuntimeContract a neg layout evaluate rt)
    (x y : Tensor) (s t : World S) (valid : Valid layout s x)
    (run : wrapper rt x s=.ok y t) : Returned a neg layout formula x y s t := by
  unfold wrapper at run
  cases hr : rt.resolve x s with
  | error e u => simp [hr,Result.bind] at run
  | ok v u =>
    have resolved := runtime.resolve x s v u valid hr
    cases ha : rt.allocate v u with
    | error e b => simp [hr,ha,Result.bind] at run
    | ok out b =>
      have allocated := runtime.allocate v u out b resolved.valid ha
      have shape : SameShape x out :=
        ⟨allocated.shape.rows.trans resolved.shape.rows,
         allocated.shape.cols.trans resolved.shape.cols,
         allocated.shape.device.trans resolved.shape.device⟩
      have frame := resolved.frame.trans allocated.frame
      have fresh : ¬s.live out.region := fun h => allocated.fresh (resolved.frame.live _ h)
      by_cases empty : x.rows=0 ∨ x.cols=0
      · have eq : out=y ∧ b=t := by simpa [hr,ha,Result.bind,empty] using run
        rcases eq with ⟨hout,hstate⟩
        subst out
        subst b
        refine ⟨shape,allocated.valid,allocated.normal,allocated.injective,fresh,frame,?_⟩
        intro r c hr hc
        rcases empty with h|h <;> omega
      · cases hc : rt.compile v out (blockSize x.cols) (numWarps (blockSize x.cols)) b with
        | error e c => simp [hr,ha,Result.bind,empty,hc] at run
        | ok unit c =>
          cases unit
          have eq := runtime.compile v out _ _ b c
            (allocated.frame.valid layout v resolved.valid) allocated.valid hc
          subst c
          cases hg : runRows (rt.invoke v out (blockSize x.cols)) (List.range x.rows) b with
          | error e d => simp [hr,ha,Result.bind,empty,hc,hg] at run
          | ok unit d =>
            cases unit
            have eq : out=y ∧ d=t := by simpa [hr,ha,Result.bind,empty,hc,hg] using run
            rcases eq with ⟨hout,hstate⟩
            subst out
            subst d
            let values := logical a neg s x
            let invariant := fun w : World S => w.live=b.live
            have input : Inputs (worldAccess a) v values b := by
              intro r j hrow hcol
              have eq := allocated.frame.cells v.region (v.index r j) resolved.valid.1
              change a.decode (a.cells b.program v.region (v.index r j))=values r j
              rw [eq]
              have coords := resolved.coordinates r j (by rwa [resolved.shape.rows] at hrow)
                (by rwa [resolved.shape.cols] at hcol)
              simpa [logical,resolved.normal,values,Access.read] using coords
            have vlive := (allocated.frame.valid layout v resolved.valid).1
            have separate : v.region≠y.region := fun eq => allocated.fresh (eq ▸ resolved.valid.1)
            have law : RowLaw (worldAccess a) v y values
                (fun r j => formula x.cols (blockSize x.cols) (values r) j)
                (rt.invoke v y (blockSize x.cols)) invariant := by
              constructor
              intro p w z hp inputs inv call
              have vx : Valid layout w v := ⟨by rw [inv];exact vlive,resolved.valid.2⟩
              have vy : Valid layout w y := ⟨by rw [inv];exact allocated.valid.1,allocated.valid.2⟩
              have localRun := runtime.invoke v y _ p w z vx vy resolved.normal allocated.normal allocated.injective hp call
              have src := source.row v y _ p w.program z.program values allocated.shape allocated.injective
                (blockSize_bounds x.cols).1 (by rw [resolved.shape.cols];exact (blockSize_bounds x.cols).2)
                hp inputs localRun.1
              refine ⟨?_,src.2,localRun.2.trans inv⟩
              intro j hj
              simpa [resolved.shape.cols] using src.1 j hj
            have complete := rows_return (worldAccess a) v y values
              (fun r j => formula x.cols (blockSize x.cols) (values r) j)
              (rt.invoke v y (blockSize x.cols)) invariant law
              ⟨allocated.shape.rows,allocated.shape.cols⟩ separate allocated.injective (List.range x.rows)
              (by intro p hp;rw [resolved.shape.rows];exact List.mem_range.mp hp)
              b t input rfl hg
            have finalLive : t.live=b.live := complete.2.2
            have finalFrame : Preserves a s t := by
              constructor
              · intro r h
                rw [finalLive]
                exact frame.live r h
              · intro r i h
                have nr : r≠y.region := fun eq => fresh (eq ▸ h)
                have off : OffRows y (List.range x.rows) r i := by
                  intro p hp j hj eq
                  exact nr eq.1.symm
                exact (complete.2.1 r i off).trans (frame.cells r i h)
            refine ⟨shape,⟨by rw [finalLive];exact allocated.valid.1,allocated.valid.2⟩,
              allocated.normal,allocated.injective,fresh,finalFrame,?_⟩
            intro r j hrow hcol
            have value := complete.1 r (List.mem_range.mpr hrow) j (by rwa [resolved.shape.cols])
            simpa [logical,allocated.normal,values,worldAccess,Access.read] using value

theorem invocation_addresses {S C V} (a : Access S C V) (neg : V → V) (layout : Layout)
    (evaluate : Tensor → Tensor → Nat → Nat → S → Option S) (rt : Runtime S)
    (runtime : RuntimeContract a neg layout evaluate rt)
    (x y : Tensor) (B p c : Nat) (s t : World S)
    (vx : Valid layout s x) (vy : Valid layout s y)
    (normalX : x.negative=Bool.false) (normalY : y.negative=Bool.false)
    (injective : y.Injective) (shape : SameShape x y)
    (hp : p<x.rows) (hc : c<x.cols) (cover : x.cols≤B)
    (call : rt.invoke x y B p s=.ok () t) :
    (Native.sourcePointer (layout.buffer x.region) x p c).toNat=
      (layout.buffer x.region).base+4*x.index p c ∧
    (Native.sourcePointer (layout.buffer y.region) y p c).toNat=
      (layout.buffer y.region).base+4*y.index p c ∧
    (Native.sourcePointer (layout.buffer x.region) x p c).toNat+4≤
      (layout.buffer x.region).base+(layout.buffer x.region).capacity ∧
    (Native.sourcePointer (layout.buffer y.region) y p c).toNat+4≤
      (layout.buffer y.region).base+(layout.buffer y.region).capacity := by
  have abi := runtime.local_abi x y B p s t vx vy normalX normalY injective hp call
  have cx : c<2^31 := lt_of_lt_of_le hc (le_trans cover abi.2)
  have ix := Native.source_pointer_exact _ x (layout.legal x vx.2).1
    (layout.legal x vx.2).2 p c hp hc abi.1 cx
  have iy := Native.source_pointer_exact _ y (layout.legal y vy.2).1
    (layout.legal y vy.2).2 p c (by rwa [shape.rows]) (by rwa [shape.cols]) abi.1 cx
  exact ⟨ix.1,iy.1,ix.2,iy.2⟩

#axiomsClean wrapper_return
end VeriTile.Bench.Repairs.SoftmaxTriton1
