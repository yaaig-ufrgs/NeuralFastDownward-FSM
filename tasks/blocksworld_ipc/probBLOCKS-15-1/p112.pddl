(define (problem blocks-15-1)
(:domain blocks)
(:objects j b k a d h e n c f l m i o g )
		(:init
		(clear f)
		(clear m)
		(handempty)
		(on a k)
		(on b j)
		(on c o)
		(on d a)
		(on e l)
		(on f n)
		(on g c)
		(on h d)
		(on j i)
		(on k b)
		(on l h)
		(on m e)
		(on n g)
		(ontable i)
		(ontable o)
		)
(:goal (and (on d g) (on g f) (on f k) (on k j) (on j e) (on e m) (on m a)
            (on a b) (on b c) (on c n) (on n o) (on o i) (on i l) (on l h)))
)
