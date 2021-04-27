(define (problem depotprob4398) (:domain depot)
(:objects
	depot0 depot1 distributor0 distributor1 truck0 truck1 truck2 truck3 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 pallet7 crate0 crate1 crate2 crate3 crate4 crate5 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 hoist6 hoist7 )
		(:init
		(at crate2 distributor1)
		(at crate4 distributor0)
		(at truck0 distributor1)
		(at truck1 depot1)
		(at truck2 depot0)
		(at truck3 depot1)
		(available hoist0)
		(available hoist3)
		(available hoist4)
		(available hoist5)
		(available hoist6)
		(available hoist7)
		(clear crate2)
		(clear crate4)
		(clear pallet0)
		(clear pallet1)
		(clear pallet4)
		(clear pallet5)
		(clear pallet6)
		(clear pallet7)
		(in crate0 truck0)
		(in crate1 truck2)
		(lifting hoist1 crate5)
		(lifting hoist2 crate3)
		(on crate2 pallet3)
		(on crate4 pallet2)
		(pallet pallet0)
		(surface pallet0)
		(at pallet0 depot0)
		(pallet pallet1)
		(surface pallet1)
		(at pallet1 depot1)
		(pallet pallet2)
		(surface pallet2)
		(at pallet2 distributor0)
		(pallet pallet3)
		(surface pallet3)
		(at pallet3 distributor1)
		(pallet pallet4)
		(surface pallet4)
		(at pallet4 depot1)
		(pallet pallet5)
		(surface pallet5)
		(at pallet5 distributor1)
		(pallet pallet6)
		(surface pallet6)
		(at pallet6 depot1)
		(pallet pallet7)
		(surface pallet7)
		(at pallet7 distributor0)
		(truck truck0)
		(truck truck1)
		(truck truck2)
		(truck truck3)
		(hoist hoist0)
		(at hoist0 depot0)
		(hoist hoist1)
		(at hoist1 depot1)
		(hoist hoist2)
		(at hoist2 distributor0)
		(hoist hoist3)
		(at hoist3 distributor1)
		(hoist hoist4)
		(at hoist4 distributor1)
		(hoist hoist5)
		(at hoist5 depot1)
		(hoist hoist6)
		(at hoist6 depot1)
		(hoist hoist7)
		(at hoist7 distributor1)
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
		(place depot0)
		(place depot1)
		(place distributor0)
		(place distributor1)
		)

(:goal (and
		(on crate0 pallet3)
		(on crate2 pallet1)
		(on crate3 pallet0)
		(on crate4 crate3)
		(on crate5 pallet2)
	)
))

