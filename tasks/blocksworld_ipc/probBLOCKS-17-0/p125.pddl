(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear e)
		(clear h)
		(clear k)
		(clear l)
		(clear p)
		(clear q)
		(holding a)
		(on c o)
		(on d c)
		(on e j)
		(on f i)
		(on g d)
		(on h m)
		(on j f)
		(on n b)
		(on p n)
		(on q g)
		(ontable b)
		(ontable i)
		(ontable k)
		(ontable l)
		(ontable m)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
