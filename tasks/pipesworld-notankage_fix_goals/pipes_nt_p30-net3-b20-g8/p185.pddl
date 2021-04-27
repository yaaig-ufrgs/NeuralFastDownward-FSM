
(define (problem network3new_all_20_8_instance)
  (:domain pipesworld_strips)
  (:objects

    	b10 b17 b14 b4 b6 b15 b19 b13 b8 b2 b11 b5 b0 b1 b18 b7 b12 b9 b3 b16 - batch-atom
	a1 a2 a3 a4 - area
	s12 s13 s34 - pipe
	

  )
  		(:init
		(first b12 s13)
		(first b13 s34)
		(first b6 s12)
		(follow b4 b8)
		(follow b8 b6)
		(follow b9 b12)
		(last b13 s34)
		(last b4 s12)
		(last b9 s13)
		(normal s13)
		(on b0 a2)
		(on b1 a2)
		(on b10 a3)
		(on b11 a2)
		(on b14 a4)
		(on b15 a2)
		(on b16 a3)
		(on b17 a2)
		(on b18 a3)
		(on b19 a1)
		(on b2 a1)
		(on b3 a4)
		(on b5 a2)
		(on b7 a3)
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
		;; batch-atoms products
		(is-product b10 rat-a)
		(is-product b17 rat-a)
		(is-product b14 gasoleo)
		(is-product b4 rat-a)
		(is-product b6 lco)
		(is-product b15 gasoleo)
		(is-product b19 oc1b)
		(is-product b13 oca1)
		(is-product b8 rat-a)
		(is-product b2 oca1)
		(is-product b11 rat-a)
		(is-product b5 lco)
		(is-product b0 gasoleo)
		(is-product b1 rat-a)
		(is-product b18 rat-a)
		(is-product b7 rat-a)
		(is-product b12 oca1)
		(is-product b9 lco)
		(is-product b3 oca1)
		(is-product b16 gasoleo)
		;; batch-atoms initially located in areas
		;; batch-atoms initially located in pipes
		;; unitary pipeline segments
		(not-unitary s12)
		(not-unitary s13)
		(unitary s34)
		)
  (:goal (and
    	(on b10 a3)
	(on b4 a1)
	(on b6 a3)
	(on b13 a1)
	(on b2 a2)
	(on b11 a1)
	(on b18 a2)
	(on b12 a4)
	
  ))
)

