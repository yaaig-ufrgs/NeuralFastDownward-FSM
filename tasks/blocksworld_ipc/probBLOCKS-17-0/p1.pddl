(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear b)
		(clear j)
		(clear m)
		(holding f)
		(on b e)
		(on c o)
		(on d c)
		(on e a)
		(on g d)
		(on h q)
		(on i p)
		(on j h)
		(on k n)
		(on m k)
		(on n i)
		(on p l)
		(on q g)
		(ontable a)
		(ontable l)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
