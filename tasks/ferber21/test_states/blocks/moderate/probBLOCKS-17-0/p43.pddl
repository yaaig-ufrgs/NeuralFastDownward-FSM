(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear c)
		(clear f)
		(clear g)
		(clear h)
		(clear l)
		(clear o)
		(handempty)
		(on a j)
		(on b m)
		(on c d)
		(on f q)
		(on g e)
		(on h k)
		(on i b)
		(on j i)
		(on l p)
		(on p n)
		(on q a)
		(ontable d)
		(ontable e)
		(ontable k)
		(ontable m)
		(ontable n)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
