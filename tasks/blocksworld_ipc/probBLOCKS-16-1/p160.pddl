(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear e)
		(clear g)
		(clear j)
		(handempty)
		(on b d)
		(on c k)
		(on d c)
		(on e h)
		(on f l)
		(on g f)
		(on i b)
		(on j m)
		(on k a)
		(on l p)
		(on n i)
		(on o n)
		(on p o)
		(ontable a)
		(ontable h)
		(ontable m)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
