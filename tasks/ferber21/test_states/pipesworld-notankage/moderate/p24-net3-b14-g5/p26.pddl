
(define (problem network3new_all_14_5_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b0 b1 b4 b6 b7 b12 b9 b3 b13 b8 b2 b11 b5 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 - pipe
	

  )
  		(:init
		(first b3 s12)
		(first b6 s34)
		(first b9 s13)
		(follow b0 b3)
		(follow b1 b8)
		(follow b5 b0)
		(follow b8 b9)
		(last b1 s13)
		(last b5 s12)
		(last b6 s34)
		(on b10 a3)
		(on b11 a4)
		(on b12 a4)
		(on b13 a3)
		(on b2 a2)
		(on b4 a4)
		(on b7 a3)
		(push-updating s12)
		(push-updating s13)
		;; all pipelines segments are in normal state
		(normal s34)
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
		(connect a3 a4 s34)
		;; batch-atoms products
		(is-product b10 oc1b)
		(is-product b0 gasoleo)
		(is-product b1 oc1b)
		(is-product b4 gasoleo)
		(is-product b6 oca1)
		(is-product b7 rat-a)
		(is-product b12 oca1)
		(is-product b9 oc1b)
		(is-product b3 gasoleo)
		(is-product b13 lco)
		(is-product b8 oc1b)
		(is-product b2 gasoleo)
		(is-product b11 oc1b)
		(is-product b5 gasoleo)
		;; batch-atoms initially located in areas
		;; batch-atoms initially located in pipes
		;; unitary pipeline segments
		(not-unitary s12)
		(not-unitary s13)
		(unitary s34)
		)
  (:goal (and
    	(on b4 a1)
	(on b7 a3)
	(on b12 a2)
	(on b9 a2)
	(on b11 a4)
	
  ))
)

