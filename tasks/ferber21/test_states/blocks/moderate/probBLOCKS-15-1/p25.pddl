(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear d)
		(clear f)
		(clear h)
		(handempty)
		(on a b)
		(on b j)
		(on c k)
		(on d n)
		(on f o)
		(on g m)
		(on h c)
		(on j i)
		(on l e)
		(on m l)
		(on n g)
		(on o a)
		(ontable e)
		(ontable i)
		(ontable k)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
