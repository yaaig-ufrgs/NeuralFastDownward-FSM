(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear a)
		(clear e)
		(clear h)
		(clear m)
		(handempty)
		(on a k)
		(on b j)
		(on d b)
		(on e d)
		(on g c)
		(on h o)
		(on j i)
		(on k f)
		(on l n)
		(on n g)
		(on o l)
		(ontable c)
		(ontable f)
		(ontable i)
		(ontable m)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
