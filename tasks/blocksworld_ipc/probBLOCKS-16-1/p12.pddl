(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear f)
		(clear j)
		(holding o)
		(on b d)
		(on c k)
		(on d c)
		(on f m)
		(on g e)
		(on h g)
		(on i b)
		(on j p)
		(on k a)
		(on l h)
		(on m l)
		(on n i)
		(on p n)
		(ontable a)
		(ontable e)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
