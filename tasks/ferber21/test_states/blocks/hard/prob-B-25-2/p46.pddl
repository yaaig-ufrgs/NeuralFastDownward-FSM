

(define (problem bw-rand-25)
(:domain blocks)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 )
(:init
	(clear b1)
	(clear b11)
	(clear b23)
	(clear b24)
	(clear b3)
	(clear b6)
	(clear b7)
	(handempty)
	(on b1 b9)
	(on b11 b15)
	(on b12 b8)
	(on b13 b16)
	(on b15 b13)
	(on b16 b12)
	(on b17 b21)
	(on b19 b18)
	(on b2 b20)
	(on b21 b2)
	(on b23 b19)
	(on b25 b10)
	(on b3 b25)
	(on b4 b17)
	(on b5 b4)
	(on b6 b14)
	(on b7 b22)
	(on b8 b5)
	(ontable b10)
	(ontable b14)
	(ontable b18)
	(ontable b20)
	(ontable b22)
	(ontable b24)
	(ontable b9)
)
(:goal
(and
(on b1 b2)
(on b2 b3)
(on b3 b4)
(on b4 b5)
(on b5 b6)
(on b6 b7)
(on b7 b8)
(on b8 b9)
(on b9 b10)
(on b10 b11)
(on b11 b12)
(on b12 b13)
(on b13 b14)
(on b14 b15)
(on b15 b16)
(on b16 b17)
(on b17 b18)
(on b18 b19)
(on b19 b20))
)
)



