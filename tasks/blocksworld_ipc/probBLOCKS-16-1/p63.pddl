(define (problem blocks-16-1)
(:domain blocks)
(:objects k c d b i n p j m l g e a o h f )
		(:init
		(clear f)
		(clear g)
		(clear j)
		(clear n)
		(clear p)
		(handempty)
		(on b d)
		(on c k)
		(on d c)
		(on f o)
		(on i b)
		(on j h)
		(on k a)
		(on l m)
		(on n i)
		(on o e)
		(on p l)
		(ontable a)
		(ontable e)
		(ontable g)
		(ontable h)
		(ontable m)
		)
(:goal (and (on d b) (on b p) (on p f) (on f g) (on g k) (on k i) (on i l)
            (on l j) (on j h) (on h a) (on a n) (on n e) (on e m) (on m c)
            (on c o)))
)
