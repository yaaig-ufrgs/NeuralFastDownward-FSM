(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear g)
		(clear i)
		(clear l)
		(handempty)
		(on a c)
		(on b m)
		(on c q)
		(on e n)
		(on f e)
		(on g o)
		(on h d)
		(on i b)
		(on j a)
		(on k f)
		(on l j)
		(on n p)
		(on p h)
		(on q k)
		(ontable d)
		(ontable m)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
