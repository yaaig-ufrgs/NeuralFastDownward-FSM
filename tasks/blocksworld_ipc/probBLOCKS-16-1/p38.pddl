(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear f)
		(clear l)
		(clear o)
		(clear p)
		(holding n)
		(on b d)
		(on c k)
		(on d c)
		(on e h)
		(on f g)
		(on i b)
		(on j m)
		(on k a)
		(on l i)
		(on o j)
		(on p e)
		(ontable a)
		(ontable g)
		(ontable h)
		(ontable m)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
