(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear a)
		(clear f)
		(clear j)
		(clear k)
		(clear m)
		(clear o)
		(handempty)
		(on a d)
		(on b e)
		(on c g)
		(on d n)
		(on f b)
		(on h i)
		(on j l)
		(on k h)
		(on l c)
		(ontable e)
		(ontable g)
		(ontable i)
		(ontable m)
		(ontable n)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
