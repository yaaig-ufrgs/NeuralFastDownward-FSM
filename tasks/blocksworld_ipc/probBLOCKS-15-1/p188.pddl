(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear c)
		(clear g)
		(clear h)
		(handempty)
		(on a i)
		(on b f)
		(on c j)
		(on d l)
		(on f d)
		(on g a)
		(on h k)
		(on i b)
		(on j m)
		(on l n)
		(on m o)
		(on n e)
		(ontable e)
		(ontable k)
		(ontable o)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
