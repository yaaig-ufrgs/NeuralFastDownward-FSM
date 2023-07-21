
(define (problem network4new_all_16_6_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b14 b4 b6 b15 b13 b8 b2 b11 b5 b0 b1 b7 b9 b12 b3 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 s23 - pipe
	

  )
  		(:init
		(first b2 s34)
		(first b3 s13)
		(first b7 s23)
		(first b8 s12)
		(follow b1 b6)
		(follow b10 b3)
		(follow b13 b14)
		(follow b14 b8)
		(follow b15 b1)
		(follow b4 b10)
		(follow b6 b7)
		(last b13 s12)
		(last b15 s23)
		(last b2 s34)
		(last b4 s13)
		(on b0 a3)
		(on b11 a1)
		(on b12 a4)
		(on b5 a3)
		(on b9 a4)
		(pop-updating s13)
		(push-updating s12)
		(push-updating s23)
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
		(is-product b14 oc1b)
		(is-product b4 lco)
		(is-product b6 rat-a)
		(is-product b15 rat-a)
		(is-product b13 oca1)
		(is-product b8 oca1)
		(is-product b2 lco)
		(is-product b11 gasoleo)
		(is-product b5 oca1)
		(is-product b0 gasoleo)
		(is-product b1 rat-a)
		(is-product b7 gasoleo)
		(is-product b9 lco)
		(is-product b12 gasoleo)
		(is-product b3 oca1)
		;; batch-atoms initially located in areas
		;; batch-atoms initially located in pipes
		;; unitary pipeline segments
		(not-unitary s12)
		(not-unitary s13)
		(unitary s34)
		(not-unitary s23)
		)
  (:goal (and
    	(on b4 a4)
	(on b6 a4)
	(on b15 a2)
	(on b2 a1)
	(on b0 a1)
	(on b1 a1)
	
  ))
)
