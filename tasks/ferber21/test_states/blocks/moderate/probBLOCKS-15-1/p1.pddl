(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear b)
		(clear e)
		(clear j)
		(clear k)
		(handempty)
		(on a h)
		(on b d)
		(on d g)
		(on e m)
		(on f l)
		(on g c)
		(on h n)
		(on j i)
		(on k a)
		(on n o)
		(on o f)
		(ontable c)
		(ontable i)
		(ontable l)
		(ontable m)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)