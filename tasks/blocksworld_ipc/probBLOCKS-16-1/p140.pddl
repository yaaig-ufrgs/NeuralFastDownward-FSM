(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear i)
		(clear j)
		(clear l)
		(handempty)
		(on b d)
		(on c k)
		(on d c)
		(on f h)
		(on g n)
		(on h e)
		(on i b)
		(on j g)
		(on k a)
		(on l f)
		(on m p)
		(on n o)
		(on o m)
		(ontable a)
		(ontable e)
		(ontable p)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
