(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear b)
		(clear f)
		(clear j)
		(clear m)
		(holding o)
		(on a l)
		(on b q)
		(on c n)
		(on d p)
		(on e k)
		(on f g)
		(on g d)
		(on h e)
		(on i h)
		(on j i)
		(on p c)
		(on q a)
		(ontable k)
		(ontable l)
		(ontable m)
		(ontable n)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
