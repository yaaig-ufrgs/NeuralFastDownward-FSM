(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear d)
		(clear e)
		(clear h)
		(clear j)
		(clear l)
		(handempty)
		(on a m)
		(on b k)
		(on c g)
		(on d a)
		(on e i)
		(on h n)
		(on i o)
		(on j b)
		(on k f)
		(on l c)
		(ontable f)
		(ontable g)
		(ontable m)
		(ontable n)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
