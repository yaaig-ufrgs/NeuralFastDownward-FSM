(define (problem blocks-17-0)
(:domain blocks)
(:objects c d e f b i j a n o k m p h g l q )
		(:init
		(clear a)
		(clear c)
		(clear h)
		(clear i)
		(clear m)
		(clear p)
		(handempty)
		(on a d)
		(on b q)
		(on c o)
		(on d n)
		(on e k)
		(on f e)
		(on i b)
		(on l f)
		(on n g)
		(on p j)
		(on q l)
		(ontable g)
		(ontable h)
		(ontable j)
		(ontable k)
		(ontable m)
		(ontable o)
		)
(:goal (and (on q n) (on n l) (on l o) (on o j) (on j h) (on h c) (on c e)
            (on e m) (on m p) (on p a) (on a g) (on g b) (on b i) (on i k)
            (on k f) (on f d)))
)
