(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear e)
		(clear f)
		(clear p)
		(holding d)
		(on a j)
		(on b m)
		(on e h)
		(on f g)
		(on g o)
		(on h l)
		(on i b)
		(on j i)
		(on k n)
		(on l k)
		(on n c)
		(on p q)
		(on q a)
		(ontable c)
		(ontable m)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
