(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear e)
		(clear f)
		(clear g)
		(clear l)
		(clear m)
		(holding p)
		(on b d)
		(on c k)
		(on d c)
		(on e n)
		(on f j)
		(on i b)
		(on k a)
		(on m o)
		(on n i)
		(on o h)
		(ontable a)
		(ontable g)
		(ontable h)
		(ontable j)
		(ontable l)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
