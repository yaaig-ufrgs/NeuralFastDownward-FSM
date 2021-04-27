
(define (problem network5new_all_22_2_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b21 b17 b14 b4 b6 b15 b19 b20 b13 b8 b2 b11 b5 b0 b1 b18 b7 b9 b12 b3 b16 - batch-atom
	a1 a2 a3 a4 a5 - area
	s12 s13 s34 s23 s15 - pipe
	

  )
  		(:init
		(first b0 s34)
		(first b10 s15)
		(first b13 s12)
		(first b20 s13)
		(first b6 s23)
		(follow b1 b14)
		(follow b11 b21)
		(follow b14 b20)
		(follow b16 b6)
		(follow b19 b7)
		(follow b2 b10)
		(follow b21 b13)
		(follow b5 b16)
		(follow b7 b2)
		(last b0 s34)
		(last b1 s13)
		(last b11 s12)
		(last b19 s15)
		(last b5 s23)
		(normal s15)
		(normal s23)
		(on b12 a1)
		(on b15 a4)
		(on b17 a4)
		(on b18 a3)
		(on b3 a5)
		(on b4 a5)
		(on b8 a2)
		(on b9 a2)
		(pop-updating s13)
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
		(connect a1 a5 s15)
		;; batch-atoms products
		(is-product b10 gasoleo)
		(is-product b21 rat-a)
		(is-product b17 oca1)
		(is-product b14 lco)
		(is-product b4 oc1b)
		(is-product b6 gasoleo)
		(is-product b15 gasoleo)
		(is-product b19 lco)
		(is-product b20 oca1)
		(is-product b13 gasoleo)
		(is-product b8 rat-a)
		(is-product b2 lco)
		(is-product b11 gasoleo)
		(is-product b5 oca1)
		(is-product b0 rat-a)
		(is-product b1 gasoleo)
		(is-product b18 oca1)
		(is-product b7 oca1)
		(is-product b9 oc1b)
		(is-product b12 gasoleo)
		(is-product b3 oc1b)
		(is-product b16 oca1)
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
    	(on b10 a3)
	(on b12 a1)
	
  ))
)

