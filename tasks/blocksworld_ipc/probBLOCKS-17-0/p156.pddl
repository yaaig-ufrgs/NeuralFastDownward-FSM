(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear d)
		(clear e)
		(clear f)
		(holding o)
		(on a j)
		(on b m)
		(on d k)
		(on e p)
		(on f n)
		(on g l)
		(on h g)
		(on i b)
		(on j i)
		(on l q)
		(on n h)
		(on p c)
		(on q a)
		(ontable c)
		(ontable k)
		(ontable m)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
