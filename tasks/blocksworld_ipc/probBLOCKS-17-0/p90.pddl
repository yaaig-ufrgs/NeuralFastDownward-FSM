(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear d)
		(clear g)
		(clear l)
		(clear n)
		(holding c)
		(on a j)
		(on b m)
		(on e k)
		(on f e)
		(on g f)
		(on h o)
		(on i b)
		(on j i)
		(on l q)
		(on n p)
		(on p h)
		(on q a)
		(ontable d)
		(ontable k)
		(ontable m)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
