(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear e)
		(clear j)
		(clear n)
		(holding p)
		(on b d)
		(on c k)
		(on d c)
		(on e f)
		(on f g)
		(on g l)
		(on h o)
		(on i b)
		(on j m)
		(on k a)
		(on l h)
		(on n i)
		(ontable a)
		(ontable m)
		(ontable o)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
