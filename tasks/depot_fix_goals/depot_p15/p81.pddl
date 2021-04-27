(define (problem depotprob4534) (:domain depot)
(:objects
	depot0 depot1 depot2 distributor0 distributor1 distributor2 truck0 truck1 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 pallet7 pallet8 pallet9 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 crate10 crate11 crate12 crate13 crate14 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 )
		(:init
		(at crate0 distributor2)
		(at crate1 distributor1)
		(at crate10 distributor1)
		(at crate11 depot0)
		(at crate13 distributor1)
		(at crate14 distributor0)
		(at crate2 distributor0)
		(at crate4 depot1)
		(at crate6 depot1)
		(at crate7 depot1)
		(at crate8 distributor0)
		(at crate9 distributor0)
		(at truck0 distributor0)
		(at truck1 depot0)
		(available hoist0)
		(available hoist2)
		(available hoist3)
		(available hoist4)
		(available hoist5)
		(clear crate0)
		(clear crate11)
		(clear crate13)
		(clear crate14)
		(clear crate2)
		(clear crate4)
		(clear crate9)
		(clear pallet2)
		(clear pallet6)
		(clear pallet8)
		(in crate3 truck1)
		(in crate5 truck1)
		(lifting hoist1 crate12)
		(on crate0 pallet5)
		(on crate1 pallet4)
		(on crate10 crate1)
		(on crate11 pallet0)
		(on crate13 crate10)
		(on crate14 crate8)
		(on crate2 pallet7)
		(on crate4 crate7)
		(on crate6 pallet1)
		(on crate7 crate6)
		(on crate8 pallet9)
		(on crate9 pallet3)
		(pallet pallet0)
		(surface pallet0)
		(at pallet0 depot0)
		(pallet pallet1)
		(surface pallet1)
		(at pallet1 depot1)
		(pallet pallet2)
		(surface pallet2)
		(at pallet2 depot2)
		(pallet pallet3)
		(surface pallet3)
		(at pallet3 distributor0)
		(pallet pallet4)
		(surface pallet4)
		(at pallet4 distributor1)
		(pallet pallet5)
		(surface pallet5)
		(at pallet5 distributor2)
		(pallet pallet6)
		(surface pallet6)
		(at pallet6 depot1)
		(pallet pallet7)
		(surface pallet7)
		(at pallet7 distributor0)
		(pallet pallet8)
		(surface pallet8)
		(at pallet8 depot2)
		(pallet pallet9)
		(surface pallet9)
		(at pallet9 distributor0)
		(truck truck0)
		(truck truck1)
		(hoist hoist0)
		(at hoist0 depot0)
		(hoist hoist1)
		(at hoist1 depot1)
		(hoist hoist2)
		(at hoist2 depot2)
		(hoist hoist3)
		(at hoist3 distributor0)
		(hoist hoist4)
		(at hoist4 distributor1)
		(hoist hoist5)
		(at hoist5 distributor2)
		(crate crate0)
		(surface crate0)
		(crate crate1)
		(surface crate1)
		(crate crate2)
		(surface crate2)
		(crate crate3)
		(surface crate3)
		(crate crate4)
		(surface crate4)
		(crate crate5)
		(surface crate5)
		(crate crate6)
		(surface crate6)
		(crate crate7)
		(surface crate7)
		(crate crate8)
		(surface crate8)
		(crate crate9)
		(surface crate9)
		(crate crate10)
		(surface crate10)
		(crate crate11)
		(surface crate11)
		(crate crate12)
		(surface crate12)
		(crate crate13)
		(surface crate13)
		(crate crate14)
		(surface crate14)
		(place depot0)
		(place depot1)
		(place depot2)
		(place distributor0)
		(place distributor1)
		(place distributor2)
		)

(:goal (and
		(on crate0 crate8)
		(on crate1 crate10)
		(on crate2 pallet0)
		(on crate3 pallet1)
		(on crate4 crate7)
		(on crate5 pallet5)
		(on crate6 pallet6)
		(on crate7 pallet4)
		(on crate8 pallet7)
		(on crate9 crate4)
		(on crate10 crate11)
		(on crate11 crate9)
		(on crate12 crate5)
		(on crate13 pallet8)
		(on crate14 pallet9)
	)
))

