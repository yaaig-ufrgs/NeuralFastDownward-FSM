(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear b)
		(clear c)
		(clear d)
		(clear m)
		(clear p)
		(holding f)
		(on b e)
		(on c k)
		(on d n)
		(on h o)
		(on j i)
		(on k a)
		(on l j)
		(on n l)
		(on o g)
		(on p h)
		(ontable a)
		(ontable e)
		(ontable g)
		(ontable i)
		(ontable m)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
