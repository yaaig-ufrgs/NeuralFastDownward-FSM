(define (problem depotprob8715) (:domain depot)
(:objects
	depot0 depot1 depot2 depot3 depot4 depot5 distributor0 distributor1 distributor2 distributor3 distributor4 distributor5 truck0 truck1 truck2 truck3 truck4 truck5 pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 pallet7 pallet8 pallet9 pallet10 pallet11 pallet12 pallet13 pallet14 pallet15 pallet16 pallet17 pallet18 pallet19 crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 hoist6 hoist7 hoist8 hoist9 hoist10 hoist11 hoist12 hoist13 hoist14 )
		(:init
		(at crate1 depot0)
		(at crate5 depot1)
		(at crate7 depot4)
		(at crate8 distributor1)
		(at crate9 depot1)
		(at truck0 depot5)
		(at truck1 depot2)
		(at truck2 depot4)
		(at truck3 depot5)
		(at truck4 depot4)
		(at truck5 distributor0)
		(available hoist0)
		(available hoist1)
		(available hoist10)
		(available hoist11)
		(available hoist12)
		(available hoist2)
		(available hoist3)
		(available hoist4)
		(available hoist5)
		(available hoist6)
		(available hoist7)
		(clear crate1)
		(clear crate5)
		(clear crate7)
		(clear crate8)
		(clear crate9)
		(clear pallet1)
		(clear pallet10)
		(clear pallet11)
		(clear pallet13)
		(clear pallet14)
		(clear pallet16)
		(clear pallet17)
		(clear pallet18)
		(clear pallet2)
		(clear pallet3)
		(clear pallet5)
		(clear pallet6)
		(clear pallet7)
		(clear pallet8)
		(clear pallet9)
		(in crate0 truck1)
		(lifting hoist13 crate3)
		(lifting hoist14 crate6)
		(lifting hoist8 crate4)
		(lifting hoist9 crate2)
		(on crate1 pallet0)
		(on crate5 pallet15)
		(on crate7 pallet4)
		(on crate8 pallet12)
		(on crate9 pallet19)
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
		(at pallet4 depot4)
		(pallet pallet5)
		(surface pallet5)
		(at pallet5 depot5)
		(pallet pallet6)
		(surface pallet6)
		(at pallet6 distributor0)
		(pallet pallet7)
		(surface pallet7)
		(at pallet7 distributor1)
		(pallet pallet8)
		(surface pallet8)
		(at pallet8 distributor2)
		(pallet pallet9)
		(surface pallet9)
		(at pallet9 distributor3)
		(pallet pallet10)
		(surface pallet10)
		(at pallet10 distributor4)
		(pallet pallet11)
		(surface pallet11)
		(at pallet11 distributor5)
		(pallet pallet12)
		(surface pallet12)
		(at pallet12 distributor1)
		(pallet pallet13)
		(surface pallet13)
		(at pallet13 distributor5)
		(pallet pallet14)
		(surface pallet14)
		(at pallet14 depot2)
		(pallet pallet15)
		(surface pallet15)
		(at pallet15 depot1)
		(pallet pallet16)
		(surface pallet16)
		(at pallet16 depot1)
		(pallet pallet17)
		(surface pallet17)
		(at pallet17 distributor2)
		(pallet pallet18)
		(surface pallet18)
		(at pallet18 depot4)
		(pallet pallet19)
		(surface pallet19)
		(at pallet19 depot1)
		(truck truck0)
		(truck truck1)
		(truck truck2)
		(truck truck3)
		(truck truck4)
		(truck truck5)
		(hoist hoist0)
		(at hoist0 depot0)
		(hoist hoist1)
		(at hoist1 depot1)
		(hoist hoist2)
		(at hoist2 depot2)
		(hoist hoist3)
		(at hoist3 depot3)
		(hoist hoist4)
		(at hoist4 depot4)
		(hoist hoist5)
		(at hoist5 depot5)
		(hoist hoist6)
		(at hoist6 distributor0)
		(hoist hoist7)
		(at hoist7 distributor1)
		(hoist hoist8)
		(at hoist8 distributor2)
		(hoist hoist9)
		(at hoist9 distributor3)
		(hoist hoist10)
		(at hoist10 distributor4)
		(hoist hoist11)
		(at hoist11 distributor5)
		(hoist hoist12)
		(at hoist12 depot5)
		(hoist hoist13)
		(at hoist13 depot1)
		(hoist hoist14)
		(at hoist14 depot4)
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
		(place depot3)
		(place depot4)
		(place depot5)
		(place distributor0)
		(place distributor1)
		(place distributor2)
		(place distributor3)
		(place distributor4)
		(place distributor5)
		)

(:goal (and
		(on crate0 pallet2)
		(on crate1 pallet7)
		(on crate2 pallet11)
		(on crate3 pallet3)
		(on crate5 pallet5)
		(on crate6 pallet12)
		(on crate7 pallet18)
		(on crate8 pallet15)
	)
))

