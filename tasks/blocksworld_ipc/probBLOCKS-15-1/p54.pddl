(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear e)
		(clear n)
		(clear o)
		(handempty)
		(on a k)
		(on b j)
		(on c f)
		(on d a)
		(on e c)
		(on f h)
		(on g m)
		(on h d)
		(on j i)
		(on k b)
		(on n g)
		(on o l)
		(ontable i)
		(ontable l)
		(ontable m)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
