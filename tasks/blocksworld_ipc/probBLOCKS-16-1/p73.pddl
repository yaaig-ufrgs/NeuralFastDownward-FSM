(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear g)
		(clear j)
		(clear m)
		(clear o)
		(holding l)
		(on b d)
		(on c k)
		(on d c)
		(on e f)
		(on g e)
		(on i b)
		(on k a)
		(on m p)
		(on n i)
		(on o h)
		(on p n)
		(ontable a)
		(ontable f)
		(ontable h)
		(ontable j)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
