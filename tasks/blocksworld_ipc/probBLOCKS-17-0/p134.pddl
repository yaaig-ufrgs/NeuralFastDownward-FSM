(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear a)
		(clear l)
		(holding p)
		(on a i)
		(on b n)
		(on c o)
		(on d c)
		(on e f)
		(on f g)
		(on g d)
		(on h q)
		(on i b)
		(on k m)
		(on l h)
		(on m j)
		(on n k)
		(on q e)
		(ontable j)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
