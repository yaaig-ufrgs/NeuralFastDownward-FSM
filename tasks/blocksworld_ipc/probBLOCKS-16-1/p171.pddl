(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear c)
		(clear i)
		(clear k)
		(holding g)
		(on b h)
		(on c o)
		(on d b)
		(on e f)
		(on h l)
		(on i m)
		(on j p)
		(on k a)
		(on m j)
		(on n e)
		(on o d)
		(on p n)
		(ontable a)
		(ontable f)
		(ontable l)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
