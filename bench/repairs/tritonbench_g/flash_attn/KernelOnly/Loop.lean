import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Proofs.Reference

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

def foldRange (update : Nat→β→β) (cur stop stride : Nat) (a : β) : β :=
  if hz : stride=0 then a else if hc : cur<stop then
    foldRange update (cur+stride) stop stride (update cur a) else a
termination_by stop-cur
decreasing_by omega

theorem range_values (M : Algebra α) (idx : RegName) (body : List ComputeStmt)
    (update : Nat→β→β) (I : Nat→β→State α→Prop) (Done : β→State α→Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t,I n a t→n<stop→∃ u,run M body (t.setReg idx .nat [] (fun _=>n))=some u ∧
      I (n+stride) (update n a) u)
    (hdone : ∀ n a t,I n a t→Done a t) (h : I cur initial s) :
    ∃ final,FP.Structural.range M idx cur stop stride body s=some final ∧
      Done (foldRange update cur stop stride initial) final := by
  by_cases hz : stride=0
  · exact ⟨s,by simp [FP.Structural.range,hz],by simpa [foldRange,hz] using hdone cur initial s h⟩
  · by_cases hc : cur<stop
    · obtain ⟨next,he,hn⟩:=hstep cur initial s h hc
      obtain ⟨final,hf,hd⟩:=range_values M idx body update I Done
        (cur+stride) stop stride (update cur initial) next hstep hdone hn
      refine ⟨final,?_,?_⟩
      · rw [FP.Structural.range]
        simpa only [hz,hc,if_false,if_true,he,Option.bind_some] using hf
      · rw [foldRange]
        simpa only [hz,hc,if_false,if_true] using hd
    · exact ⟨s,by simp [FP.Structural.range,hz,hc],by simpa [foldRange,hz,hc] using hdone cur initial s h⟩
termination_by stop-cur
decreasing_by omega

def stop (p : Parameters) (s : State α) : Nat :=
  if p.IS_CAUSAL then min ((s.pids 0+1)*128) p.SEQLEN else p.SEQLEN

def accumulated (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α) (scale : α) : Acc p α :=
  foldRange (update M dot p s scale) 0 (stop p s) 64 (initialAcc M p)

theorem loopExec (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial current : State α) (ctx : Context M p initial scale 0 (initialAcc M p) current) :
    ∃ t,range (engine (withDot M dot)) "start_n" 0 (stop p initial) 64 (parts p).body current=some t ∧
      ∃ row,Context M p initial scale row (accumulated M dot p initial scale) t := by
  exact range_values (engine (withDot M dot)) "start_n" (parts p).body (update M dot p initial scale)
    (fun row a s=>Context M p initial scale row a s)
    (fun a s=>∃ row,Context M p initial scale row a s) 0 (stop p initial) 64 _ current
    (by
      intro row a s h _
      let t:=bodyExec M dot p scale initial s row a h
      exact ⟨t.val,t.property,body_context M dot p scale initial s row a h⟩)
    (by intro row a s h;exact ⟨row,h⟩) ctx

theorem entry_bounds (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α) (s : State α) :
    (entry M dot p scale s).val.regs .nat [] "lo"=some (fun _=>0) ∧
    (entry M dot p scale s).val.regs .nat [] "hi"=some (fun _=>stop p s) := by
  rcases p with ⟨qbs,qh,qr,qd,kbs,kh,kr,kd,vbs,vh,vr,vd,obs,oh,orr,od,bs,heads,len,dim,causal⟩
  cases causal <;> simp [entry,stop]
  split_ifs with h
  · exact funext (fun _=>(min_eq_left (le_of_lt h)).symm)
  · exact funext (fun _=>(min_eq_right (le_of_not_gt h)).symm)

#axiomsClean loopExec
#specNonCircular accumulated avoiding [run,FP.Structural.exec,bodyExec,loopExec]
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
