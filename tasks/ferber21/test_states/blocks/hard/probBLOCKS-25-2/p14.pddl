

(define (problem bw-rand-25)
(:domain blocks)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 )
(:init
	(clear b11)
	(clear b12)
	(clear b20)
	(clear b5)
	(handempty)
	(on b1 b25)
	(on b10 b7)
	(on b11 b23)
	(on b12 b16)
	(on b13 b8)
	(on b14 b19)
	(on b15 b24)
	(on b16 b4)
	(on b17 b2)
	(on b19 b18)
	(on b2 b21)
	(on b20 b13)
	(on b21 b10)
	(on b23 b17)
	(on b24 b14)
	(on b25 b6)
	(on b4 b1)
	(on b5 b22)
	(on b6 b9)
	(on b8 b3)
	(on b9 b15)
	(ontable b18)
	(ontable b22)
	(ontable b3)
	(ontable b7)
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


