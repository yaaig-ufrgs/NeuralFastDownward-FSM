
(define (problem network4new_all_18_4_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b17 b14 b4 b6 b15 b13 b8 b2 b11 b5 b0 b1 b7 b12 b9 b3 b16 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 s23 - pipe
	

  )
  (:init
	(first b11 s13)
	(first b15 s23)
	(first b3 s12)
	(first b5 s34)
	(follow b10 b15)
	(follow b12 b14)
	(follow b14 b10)
	(follow b4 b7)
	(follow b7 b11)
	(follow b8 b3)
	(last b12 s23)
	(last b4 s13)
	(last b5 s34)
	(last b8 s12)
	(normal s12)
	(on b0 a1)
	(on b1 a1)
	(on b13 a4)
	(on b16 a2)
	(on b17 a2)
	(on b2 a2)
	(on b6 a3)
	(on b9 a1)
	(push-updating s13)
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
	(is-product b10 rat-a)
	(is-product b17 oc1b)
	(is-product b14 gasoleo)
	(is-product b4 rat-a)
	(is-product b6 lco)
	(is-product b15 gasoleo)
	(is-product b13 oca1)
	(is-product b8 gasoleo)
	(is-product b2 oc1b)
	(is-product b11 oca1)
	(is-product b5 lco)
	(is-product b0 rat-a)
	(is-product b1 oc1b)
	(is-product b7 lco)
	(is-product b12 lco)
	(is-product b9 gasoleo)
	(is-product b3 lco)
	(is-product b16 oc1b)
	;; batch-atoms initially located in areas
	;; batch-atoms initially located in pipes
	;; unitary pipeline segments
	(not-unitary s12)
	(not-unitary s13)
	(unitary s34)
	(not-unitary s23)
)
  (:goal (and
    	(on b2 a4)
	(on b1 a3)
	(on b7 a1)
	(on b16 a3)
	
  ))
)
