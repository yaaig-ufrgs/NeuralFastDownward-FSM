(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear e)
		(clear h)
		(handempty)
		(on a k)
		(on b j)
		(on d a)
		(on e g)
		(on f m)
		(on g f)
		(on h o)
		(on j i)
		(on k b)
		(on l c)
		(on m n)
		(on n l)
		(on o d)
		(ontable c)
		(ontable i)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
