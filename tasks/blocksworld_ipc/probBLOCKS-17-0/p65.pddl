(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear k)
		(clear n)
		(clear p)
		(clear q)
		(handempty)
		(on a j)
		(on b m)
		(on c o)
		(on d c)
		(on f e)
		(on g d)
		(on i b)
		(on j i)
		(on k f)
		(on l g)
		(on n l)
		(on p a)
		(on q h)
		(ontable e)
		(ontable h)
		(ontable m)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
