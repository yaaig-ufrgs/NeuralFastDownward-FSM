(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear b)
		(clear c)
		(clear f)
		(clear j)
		(clear o)
		(handempty)
		(on b d)
		(on c p)
		(on d n)
		(on e h)
		(on f k)
		(on g i)
		(on k a)
		(on l e)
		(on m l)
		(on n m)
		(on p g)
		(ontable a)
		(ontable h)
		(ontable i)
		(ontable j)
		(ontable o)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
