(define (problem depotprob8765) (:domain depot)
(:objects
	depot0 depot1 depot2 distributor0 distributor1 distributor2 truck0 truck1 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 )
		(:init
		(at crate1 depot0)
		(at crate4 distributor0)
		(at crate5 distributor0)
		(at crate6 distributor2)
		(at crate7 distributor2)
		(at crate8 distributor2)
		(at crate9 distributor1)
		(at truck0 depot1)
		(at truck1 distributor1)
		(available hoist0)
		(available hoist3)
		(available hoist5)
		(clear crate1)
		(clear crate4)
		(clear crate8)
		(clear crate9)
		(clear pallet1)
		(clear pallet2)
		(lifting hoist1 crate0)
		(lifting hoist2 crate2)
		(lifting hoist4 crate3)
		(on crate1 pallet0)
		(on crate4 crate5)
		(on crate5 pallet3)
		(on crate6 pallet5)
		(on crate7 crate6)
		(on crate8 crate7)
		(on crate9 pallet4)
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
		(place depot0)
		(place depot1)
		(place depot2)
		(place distributor0)
		(place distributor1)
		(place distributor2)
		)

(:goal (and
		(on crate0 crate7)
		(on crate1 pallet4)
		(on crate2 pallet5)
		(on crate3 crate9)
		(on crate4 pallet0)
		(on crate5 pallet2)
		(on crate6 crate5)
		(on crate7 crate1)
		(on crate8 pallet3)
		(on crate9 crate2)
	)
))

