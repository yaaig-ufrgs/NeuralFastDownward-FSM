
(define (problem network4new_all_14_3_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b0 b1 b4 b6 b7 b12 b9 b3 b13 b8 b2 b11 b5 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 s23 - pipe
	

  )
  		(:init
		(first b0 s34)
		(first b13 s12)
		(first b4 s13)
		(first b5 s23)
		(follow b1 b4)
		(follow b10 b11)
		(follow b11 b12)
		(follow b12 b5)
		(follow b2 b13)
		(follow b6 b2)
		(follow b8 b1)
		(last b0 s34)
		(last b10 s23)
		(last b6 s12)
		(last b8 s13)
		(on b3 a4)
		(on b7 a1)
		(on b9 a4)
		(pop-updating s13)
		(pop-updating s23)
		(push-updating s12)
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
		(connect a2 a3 s23)
		;; batch-atoms products
		(is-product b10 oca1)
		(is-product b0 oca1)
		(is-product b1 gasoleo)
		(is-product b4 oc1b)
		(is-product b6 gasoleo)
		(is-product b7 oc1b)
		(is-product b12 gasoleo)
		(is-product b9 oc1b)
		(is-product b3 gasoleo)
		(is-product b13 lco)
		(is-product b8 lco)
		(is-product b2 rat-a)
		(is-product b11 gasoleo)
		(is-product b5 rat-a)
		;; batch-atoms initially located in areas
		;; batch-atoms initially located in pipes
		;; unitary pipeline segments
		(not-unitary s12)
		(not-unitary s13)
		(unitary s34)
		(not-unitary s23)
		)
  (:goal (and
    	(on b1 a2)
	(on b4 a4)
	(on b11 a1)
	
  ))
)

