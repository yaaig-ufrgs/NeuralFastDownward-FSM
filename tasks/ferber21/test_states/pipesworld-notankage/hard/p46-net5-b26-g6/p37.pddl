
(define (problem network5new_all_26_6_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b21 b17 b14 b22 b4 b6 b15 b19 b20 b13 b8 b2 b11 b24 b5 b0 b1 b25 b18 b7 b12 b9 b3 b23 b16 - batch-atom
	a1 a2 a3 a4 a5 - area
	s12 s13 s34 s23 s15 - pipe
	

  )
  (:init
	(first b0 s34)
	(first b15 s12)
	(first b16 s13)
	(first b23 s15)
	(first b7 s23)
	(follow b1 b2)
	(follow b11 b9)
	(follow b12 b19)
	(follow b14 b23)
	(follow b17 b14)
	(follow b19 b7)
	(follow b2 b16)
	(follow b21 b5)
	(follow b24 b12)
	(follow b5 b15)
	(follow b9 b17)
	(last b0 s34)
	(last b1 s13)
	(last b11 s15)
	(last b21 s12)
	(last b24 s23)
	(on b10 a4)
	(on b13 a4)
	(on b18 a1)
	(on b20 a3)
	(on b22 a4)
	(on b25 a3)
	(on b3 a5)
	(on b4 a2)
	(on b6 a1)
	(on b8 a1)
	(pop-updating s12)
	(pop-updating s13)
	(pop-updating s15)
	(pop-updating s23)
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
	(connect a1 a5 s15)
	;; batch-atoms products
	(is-product b10 rat-a)
	(is-product b21 rat-a)
	(is-product b17 gasoleo)
	(is-product b14 oca1)
	(is-product b22 oc1b)
	(is-product b4 lco)
	(is-product b6 gasoleo)
	(is-product b15 rat-a)
	(is-product b19 gasoleo)
	(is-product b20 lco)
	(is-product b13 gasoleo)
	(is-product b8 lco)
	(is-product b2 gasoleo)
	(is-product b11 rat-a)
	(is-product b24 oca1)
	(is-product b5 rat-a)
	(is-product b0 lco)
	(is-product b1 lco)
	(is-product b25 lco)
	(is-product b18 oca1)
	(is-product b7 rat-a)
	(is-product b12 lco)
	(is-product b9 lco)
	(is-product b3 oca1)
	(is-product b23 oc1b)
	(is-product b16 oc1b)
	;; batch-atoms initially located in areas
	;; batch-atoms initially located in pipes
	;; unitary pipeline segments
	(not-unitary s12)
	(not-unitary s13)
	(unitary s34)
	(not-unitary s23)
	(not-unitary s15)
)
  (:goal (and
    	(on b17 a4)
	(on b6 a3)
	(on b20 a2)
	(on b8 a3)
	(on b12 a5)
	(on b9 a1)
	
  ))
)

