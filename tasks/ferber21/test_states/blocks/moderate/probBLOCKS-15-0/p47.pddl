(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear g)
		(clear h)
		(clear i)
		(clear j)
		(clear k)
		(clear o)
		(handempty)
		(on a c)
		(on c f)
		(on e m)
		(on f n)
		(on g a)
		(on i d)
		(on k e)
		(on m l)
		(on o b)
		(ontable b)
		(ontable d)
		(ontable h)
		(ontable j)
		(ontable l)
		(ontable n)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
