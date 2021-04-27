
(define (problem network2new_all_18_6_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b17 b14 b4 b6 b15 b13 b8 b2 b11 b5 b0 b1 b7 b12 b9 b3 b16 - batch-atom
	a1 a2 a3 - area
	s12 s13 - pipe
	

  )
  		(:init
		(first b14 s13)
		(first b7 s12)
		(follow b1 b14)
		(follow b10 b1)
		(follow b12 b13)
		(follow b13 b7)
		(last b10 s13)
		(last b12 s12)
		(on b0 a2)
		(on b11 a2)
		(on b15 a3)
		(on b16 a2)
		(on b17 a1)
		(on b2 a2)
		(on b3 a1)
		(on b4 a2)
		(on b5 a1)
		(on b6 a2)
		(on b8 a1)
		(on b9 a2)
		(pop-updating s12)
		(pop-updating s13)
		;; all pipelines segments are in normal state
		;; interfaces restrictions
		(may-interface lco lco)
		(may-interface gasoleo gasoleo)
		(may-interface rat-a rat-a)
		(may-interface oca1 oca1)
		(may-interface oc1b oc1b)
		(may-interface lco gasoleo)
		(may-interface gasoleo lco)
		(may-interface lco oca1)
		(may-interface oca1 lco)
		(may-interface lco oc1b)
		(may-interface oc1b lco)
		(may-interface lco rat-a)
		(may-interface rat-a lco)
		(may-interface gasoleo rat-a)
		(may-interface rat-a gasoleo)
		(may-interface gasoleo oca1)
		(may-interface oca1 gasoleo)
		(may-interface gasoleo oc1b)
		(may-interface oc1b gasoleo)
		(may-interface oca1 oc1b)
		(may-interface oc1b oca1)
		;; network topology definition
		(connect a1 a2 s12)
		(connect a1 a3 s13)
		;; batch-atoms products
		(is-product b10 gasoleo)
		(is-product b17 rat-a)
		(is-product b14 rat-a)
		(is-product b4 rat-a)
		(is-product b6 lco)
		(is-product b15 oc1b)
		(is-product b13 oc1b)
		(is-product b8 rat-a)
		(is-product b2 oca1)
		(is-product b11 oca1)
		(is-product b5 gasoleo)
		(is-product b0 oc1b)
		(is-product b1 rat-a)
		(is-product b7 lco)
		(is-product b12 gasoleo)
		(is-product b9 oca1)
		(is-product b3 oca1)
		(is-product b16 gasoleo)
		;; batch-atoms initially located in areas
		;; batch-atoms initially located in pipes
		;; unitary pipeline segments
		(not-unitary s12)
		(not-unitary s13)
		)
  (:goal (and
    	(on b15 a1)
	(on b2 a1)
	(on b5 a1)
	(on b7 a2)
	(on b12 a2)
	(on b3 a2)
	
  ))
)

