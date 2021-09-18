
(define (problem network5new_all_24_5_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b21 b17 b14 b22 b4 b6 b15 b19 b20 b13 b8 b2 b11 b5 b0 b1 b18 b7 b12 b9 b3 b23 b16 - batch-atom
	a1 a2 a3 a4 a5 - area
	s12 s13 s34 s23 s15 - pipe
	

  )
  (:init
	(first b1 s13)
	(first b16 s15)
	(first b4 s34)
	(first b5 s23)
	(first b8 s12)
	(follow b11 b3)
	(follow b15 b5)
	(follow b2 b11)
	(follow b20 b16)
	(follow b21 b1)
	(follow b22 b21)
	(follow b23 b9)
	(follow b3 b20)
	(follow b7 b15)
	(follow b9 b8)
	(last b2 s15)
	(last b22 s13)
	(last b23 s12)
	(last b4 s34)
	(last b7 s23)
	(normal s23)
	(on b0 a3)
	(on b10 a2)
	(on b12 a1)
	(on b13 a2)
	(on b14 a5)
	(on b17 a3)
	(on b18 a2)
	(on b19 a3)
	(on b6 a5)
	(pop-updating s12)
	(pop-updating s13)
	(push-updating s15)
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
	(is-product b10 oc1b)
	(is-product b21 oca1)
	(is-product b17 rat-a)
	(is-product b14 gasoleo)
	(is-product b22 oc1b)
	(is-product b4 rat-a)
	(is-product b6 oc1b)
	(is-product b15 oca1)
	(is-product b19 gasoleo)
	(is-product b20 oca1)
	(is-product b13 lco)
	(is-product b8 oc1b)
	(is-product b2 gasoleo)
	(is-product b11 gasoleo)
	(is-product b5 gasoleo)
	(is-product b0 gasoleo)
	(is-product b1 gasoleo)
	(is-product b18 oca1)
	(is-product b7 gasoleo)
	(is-product b12 rat-a)
	(is-product b9 oc1b)
	(is-product b3 oc1b)
	(is-product b23 oc1b)
	(is-product b16 gasoleo)
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
    	(on b22 a2)
	(on b13 a4)
	(on b8 a2)
	(on b11 a4)
	(on b3 a2)
	
  ))
)

