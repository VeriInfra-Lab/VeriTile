import bench.repairs.tritonbench_g.max_reduction.Proofs.PublicEquivalence

namespace VeriTile.Bench.Repairs.MaxReduction.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def rowResult {α β : Type} (e : Source.Engine β) (encode : α→β)
    (inp : RegionName) (M N BM BN p : Nat)
    (xs : ChannelValues α [⟨.real,[BM,BN],inp⟩]) (i : Fin BM) : β×Nat :=
  e.indexed BN (fun n => if p*BM+i.val<M ∧ n.val<N then encode (xs 0 (i,n,PUnit.unit)) else e.padding)

def dimensionResult {α β : Type} (e : Source.Engine β) (encode : α→β) (observe : β→α)
    (inp ov oi : RegionName) (M N BM BN p : Nat)
    (xs : ChannelValues α [⟨.real,[BM,BN],inp⟩]) : ChannelValues α [⟨.real,[BM],ov⟩,⟨.int,[BM],oi⟩] :=
  Fin.cases (fun i => observe (rowResult e encode inp M N BM BN p xs i.1).1)
    (Fin.cases (fun i => ((rowResult e encode inp M N BM BN p xs i.1).2 : Int)) (fun i => i.elim0))

theorem row_binding {α β : Type} [Inhabited β] (e : Source.Engine β) (encode : α→β) (observe : β→α)
    (inp ov oi : RegionName) (M N K BM BN p k : Nat)
    (xs : (dimensionIO e encode observe inp ov oi M N K BM BN p k).Inputs)
    (s : State β) (pins : (dimensionIO e encode observe inp ov oi M N K BM BN p k).InputBinding xs s)
    (i : Fin BM) :
    Source.rowValue e (fun j => (s.mem inp j).read .real) M N K BM BN p k i.val=
      rowResult e encode inp M N BM BN p xs i := by
  unfold Source.rowValue rowResult
  apply congrArg (e.indexed BN)
  funext n
  split_ifs with live
  · exact pins (0 : Fin 1) (i,n,PUnit.unit) live
  · rfl

theorem dimension_implements {α β : Type} [Inhabited β] (e : Source.Engine β)
    (encode : α→β) (observe : β→α) (inp ov oi : RegionName)
    (M N K BM BN p k : Nat) (hBN : 0<BN) (hK : 0<K) (regions : ov≠oi) :
    (dimensionIO e encode observe inp ov oi M N K BM BN p k).Implements
      (dimensionResult e encode observe inp ov oi M N BM BN p) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds covered s _
    exact dimension_safe e encode observe Bool.false inp ov oi M N K BM BN p k hBN xs bounds covered s
  · intro xs s pins
    let entry := (dimensionLaunch e Bool.false inp ov oi M N K BM BN p k).initial s
    obtain ⟨final,ran,memory⟩ := Source.dimension_exec e inp ov oi M N K BM BN entry hBN
    refine ⟨final,?_,?_,?_⟩
    · simpa [dimensionIO,dimensionLaunch,runSchedule,Invocation.run] using ran
    · intro o
      refine Fin.cases ?_ (Fin.cases ?_ (fun i => i.elim0)) o
      · intro j active
        have value := (Stores.values e inp ov oi M N K BM BN regions hK entry j active).1
        change mapCell observe (final.mem ov ((p*BM+j.1.val)*K+k))=
          Cell.mk .real (observe (rowResult e encode inp M N BM BN p xs j.1).1)
        have hp0 : entry.pids 0=p := rfl
        have hp1 : entry.pids 1=k := rfl
        have hm : entry.mem=s.mem := rfl
        rw [hp0,hp1,hm] at value
        rw [memory,value]
        have bound := row_binding e encode observe inp ov oi M N K BM BN p k xs s pins j.1
        change mapCell observe (Cell.mk .real
          (Source.rowValue e (fun j => (s.mem inp j).read .real) M N K BM BN p k j.1.val).1)=_
        rw [bound]
        rfl
      · intro j active
        have value := (Stores.values e inp ov oi M N K BM BN regions hK entry j active).2
        change mapCell observe (final.mem oi ((p*BM+j.1.val)*K+k))=
          Cell.mk .int ((rowResult e encode inp M N BM BN p xs j.1).2 : Int)
        have hp0 : entry.pids 0=p := rfl
        have hp1 : entry.pids 1=k := rfl
        have hm : entry.mem=s.mem := rfl
        rw [hp0,hp1,hm] at value
        rw [memory,value]
        have bound := row_binding e encode observe inp ov oi M N K BM BN p k xs s pins j.1
        change mapCell observe (Cell.mk .int
          ((Source.rowValue e (fun j => (s.mem inp j).read .real) M N K BM BN p k j.1.val).2 : Int))=_
        rw [bound]
        rfl
    · intro r o outside
      rw [memory]
      exact dimension_frame e encode observe inp ov oi M N K BM BN p k xs s r o outside

#axiomsClean dimension_implements
end
end VeriTile.Bench.Repairs.MaxReduction.PublicIO
