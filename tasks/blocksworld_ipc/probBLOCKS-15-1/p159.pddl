(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear e)
		(clear g)
		(clear h)
		(clear l)
		(clear o)
		(handempty)
		(on a f)
		(on b j)
		(on d k)
		(on e c)
		(on g m)
		(on h n)
		(on j i)
		(on k b)
		(on n a)
		(on o d)
		(ontable c)
		(ontable f)
		(ontable i)
		(ontable l)
		(ontable m)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
