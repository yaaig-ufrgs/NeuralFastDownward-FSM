(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear c)
		(clear g)
		(clear m)
		(clear o)
		(handempty)
		(on a k)
		(on b d)
		(on c e)
		(on d i)
		(on e l)
		(on f p)
		(on h a)
		(on k f)
		(on l n)
		(on m b)
		(on n j)
		(on o h)
		(ontable g)
		(ontable i)
		(ontable j)
		(ontable p)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
