(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear h)
		(clear i)
		(clear j)
		(clear l)
		(clear o)
		(clear p)
		(holding n)
		(on b d)
		(on c k)
		(on d c)
		(on h g)
		(on i b)
		(on j f)
		(on k a)
		(on m e)
		(on p m)
		(ontable a)
		(ontable e)
		(ontable f)
		(ontable g)
		(ontable l)
		(ontable o)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
