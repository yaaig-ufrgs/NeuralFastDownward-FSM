(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear g)
		(clear h)
		(clear l)
		(clear o)
		(holding p)
		(on b d)
		(on c k)
		(on d c)
		(on g f)
		(on h j)
		(on i b)
		(on j m)
		(on k a)
		(on l e)
		(on n i)
		(on o n)
		(ontable a)
		(ontable e)
		(ontable f)
		(ontable m)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
