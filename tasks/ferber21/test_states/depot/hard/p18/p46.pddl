(define (problem depotprob1916) (:domain depot)
(:objects
	depot0 depot1 distributor0 distributor1 truck0 truck1 truck2 truck3 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 pallet7 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 crate10 crate11 crate12 crate13 crate14 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 hoist6 hoist7 )
(:init
	(at crate0 depot0)
	(at crate1 depot1)
	(at crate10 distributor0)
	(at crate11 depot1)
	(at crate13 depot1)
	(at crate2 distributor1)
	(at crate3 distributor1)
	(at crate6 distributor1)
	(at crate9 distributor1)
	(at truck0 distributor1)
	(at truck1 depot0)
	(at truck2 distributor1)
	(at truck3 distributor1)
	(available hoist2)
	(available hoist4)
	(clear crate0)
	(clear crate1)
	(clear crate10)
	(clear crate13)
	(clear crate3)
	(clear pallet2)
	(clear pallet4)
	(clear pallet6)
	(lifting hoist0 crate7)
	(lifting hoist1 crate12)
	(lifting hoist3 crate14)
	(lifting hoist5 crate4)
	(lifting hoist6 crate5)
	(lifting hoist7 crate8)
	(on crate0 pallet0)
	(on crate1 pallet1)
	(on crate10 pallet5)
	(on crate11 pallet7)
	(on crate13 crate11)
	(on crate2 crate9)
	(on crate3 crate2)
	(on crate6 pallet3)
	(on crate9 crate6)
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
	(at pallet4 depot0)
	(pallet pallet5)
	(surface pallet5)
	(at pallet5 distributor0)
	(pallet pallet6)
	(surface pallet6)
	(at pallet6 distributor1)
	(pallet pallet7)
	(surface pallet7)
	(at pallet7 depot1)
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
	(at hoist4 distributor0)
	(hoist hoist5)
	(at hoist5 depot0)
	(hoist hoist6)
	(at hoist6 distributor0)
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
	(place distributor0)
	(place distributor1)
)

(:goal (and
		(on crate0 crate10)
		(on crate1 pallet6)
		(on crate2 crate12)
		(on crate4 pallet4)
		(on crate5 pallet2)
		(on crate6 pallet7)
		(on crate8 crate4)
		(on crate9 crate1)
		(on crate10 pallet1)
		(on crate11 pallet5)
		(on crate12 crate5)
		(on crate13 pallet3)
		(on crate14 pallet0)
	)
))

