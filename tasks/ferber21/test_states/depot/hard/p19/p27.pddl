(define (problem depotprob6178) (:domain depot)
(:objects
	depot0 depot1 depot2 depot3 distributor0 distributor1 distributor2 distributor3 truck0 truck1 truck2 truck3 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 pallet7 pallet8 pallet9 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 hoist6 hoist7 )
(:init
	(at crate0 depot3)
	(at crate2 depot3)
	(at crate3 distributor1)
	(at crate4 distributor2)
	(at crate5 distributor1)
	(at crate7 depot3)
	(at truck0 distributor2)
	(at truck1 distributor1)
	(at truck2 distributor1)
	(at truck3 depot3)
	(available hoist0)
	(available hoist1)
	(available hoist3)
	(available hoist4)
	(available hoist5)
	(available hoist6)
	(available hoist7)
	(clear crate2)
	(clear crate4)
	(clear crate5)
	(clear crate7)
	(clear pallet0)
	(clear pallet1)
	(clear pallet2)
	(clear pallet4)
	(clear pallet6)
	(clear pallet7)
	(in crate6 truck0)
	(lifting hoist2 crate1)
	(on crate0 pallet9)
	(on crate2 pallet3)
	(on crate3 pallet5)
	(on crate4 pallet8)
	(on crate5 crate3)
	(on crate7 crate0)
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
	(at pallet3 depot3)
	(pallet pallet4)
	(surface pallet4)
	(at pallet4 distributor0)
	(pallet pallet5)
	(surface pallet5)
	(at pallet5 distributor1)
	(pallet pallet6)
	(surface pallet6)
	(at pallet6 distributor2)
	(pallet pallet7)
	(surface pallet7)
	(at pallet7 distributor3)
	(pallet pallet8)
	(surface pallet8)
	(at pallet8 distributor2)
	(pallet pallet9)
	(surface pallet9)
	(at pallet9 depot3)
	(truck truck0)
	(truck truck1)
	(truck truck2)
	(truck truck3)
	(hoist hoist0)
	(at hoist0 depot0)
	(hoist hoist1)
	(at hoist1 depot1)
	(hoist hoist2)
	(at hoist2 depot2)
	(hoist hoist3)
	(at hoist3 depot3)
	(hoist hoist4)
	(at hoist4 distributor0)
	(hoist hoist5)
	(at hoist5 distributor1)
	(hoist hoist6)
	(at hoist6 distributor2)
	(hoist hoist7)
	(at hoist7 distributor3)
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
	(place depot0)
	(place depot1)
	(place depot2)
	(place depot3)
	(place distributor0)
	(place distributor1)
	(place distributor2)
	(place distributor3)
)

(:goal (and
		(on crate0 pallet6)
		(on crate1 pallet8)
		(on crate3 crate1)
		(on crate4 pallet5)
		(on crate5 crate7)
		(on crate6 pallet4)
		(on crate7 crate4)
	)
))

